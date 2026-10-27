"""发现已保存分类器，并在用户选择的测试集上进行只读推理。"""
import csv
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

import config
from services import storage
from services.frozen_scramble import sha256
from services.strong_original import _test_report
from src.modeling import build_model, IMAGENET_MEAN, IMAGENET_STD
from src.scrambling.arnold import scramble
from src.task_control import JobCancelled, check_cancelled, begin_commit, track_output


def model_catalog():
    records = {e['id']: e for e in storage.experiments()}
    found, bundled = [], set()
    for group in ('original_classifier', 'scrambled_classifier'):
        for settings_file in sorted((config.MODEL_DIR / group).glob('*/config.json')):
            path = settings_file.parent / 'model.pth'
            if not path.is_file():
                continue
            try:
                settings = json.loads(settings_file.read_text(encoding='utf-8'))
                metrics = json.loads((path.parent / 'metrics.json').read_text(encoding='utf-8'))
                experiment_id = settings['experiment_id']
                record = records.get(experiment_id, {})
                k = int(settings.get('trained_iterations', 0))
                augmentation = settings.get('augmentation') == 'pre_scramble_light_v1'
                name = '原图识别模型' if group == 'original_classifier' else f'Arnold {k} 次置乱模型（{"有" if augmentation else "无"}数据增强）'
                found.append(_entry(f'bundle:{group}:{experiment_id}', name, path, settings,
                                    record, metrics, k))
                bundled.add(experiment_id)
            except (KeyError, ValueError, OSError):
                continue
    for experiment_id, record in records.items():
        if record['status'] != 'done' or experiment_id in bundled or not record.get('results_dir'):
            continue
        result = Path(record['results_dir']).resolve()
        if not result.is_relative_to(config.RESULTS_DIR.resolve()):
            continue
        for path in sorted(result.glob('*/best.pth')):
            try:
                metrics = json.loads((path.parent / 'metrics.json').read_text(encoding='utf-8'))
                label = path.parent.name
                if label != 'original' and not label.isdigit():
                    continue
                settings = {**record.get('parameters', {}), **metrics}
                k = int(label) if label.isdigit() else 0
                name = f'普通训练 · {settings["arch"]} · ' + (f'Arnold {k} 次' if k else '原图')
                found.append(_entry(f'experiment:{experiment_id}:{label}', name, path, settings,
                                    record, metrics, k))
            except (KeyError, ValueError, OSError):
                continue
    return found


def _entry(model_id, name, path, settings, record, metrics, iterations):
    arch = settings.get('arch', 'resnet18')
    if arch not in ('resnet18', 'resnet50'):
        raise ValueError('不支持的模型结构')
    size = int(settings.get('image_size', 224))
    preprocess = settings.get('preprocess') or {
        'resize': [size, size], 'mean': IMAGENET_MEAN, 'std': IMAGENET_STD}
    return {'id': model_id, 'name': name, 'arch': arch,
            'experiment_id': record.get('id', settings.get('experiment_id')),
            'dataset_id': record.get('dataset_id'), 'dataset_name': record.get('dataset_name', '来源未记录'),
            'image_size': int(preprocess['resize'][0]), 'trained_iterations': iterations,
            'historical_accuracy': metrics.get('best_test_acc'),
            'selection_split': settings.get('selection_split', 'test'),
            '_path': path, '_preprocess': preprocess, '_classes': settings.get('classes', ['cat', 'dog'])}


def public_models():
    return [{k: v for k, v in entry.items() if not k.startswith('_')} for entry in model_catalog()]


def resolve_selection(model_id, dataset_id, conditions, iterations, input_source):
    model = next((m for m in model_catalog() if m['id'] == model_id), None)
    if model is None:
        raise ValueError('模型不存在或文件不完整，请刷新模型列表')
    dataset = next((d for d in storage.catalog()['datasets']
                    if d['id'] == dataset_id and d['status'] == 'ready'), None)
    if dataset is None or not (Path(dataset['raw_dir']) / 'test').is_dir():
        raise ValueError('请选择包含测试集的已保存数据集')
    if conditions != 'original' and not iterations:
        raise ValueError('请至少选择一个置乱次数')
    if conditions != 'original' and input_source == 'saved':
        if model['image_size'] != dataset.get('image_size', config.IMAGE_SIZE):
            raise ValueError('保存的置乱图尺寸与模型输入不一致，请选择临时无损置乱')
        root = dataset.get('scrambled_dir')
        for k in iterations:
            if not root or not (Path(root) / f'arnold_{k:02d}' / 'test').is_dir():
                raise ValueError(f'此数据集尚无完整的 Arnold {k} 次置乱版本，请先生成或选择临时无损置乱')
    return model, dataset


class EvaluationImages(Dataset):
    def __init__(self, folder, preprocess, iterations=0, saved=False):
        self.folder, self.iterations, self.saved = folder, iterations, saved
        self.size = tuple(preprocess['resize'])
        self.transform = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize(preprocess['mean'], preprocess['std'])])

    def __len__(self):
        return len(self.folder)

    def __getitem__(self, index):
        path, label = self.folder.samples[index]
        with Image.open(path) as image:
            image = image.convert('RGB')
            if self.saved:
                if image.size != self.size:
                    raise ValueError(f'保存的置乱图尺寸不匹配：{path}')
            else:
                image = transforms.functional.resize(image, list(self.size))
            pixels = np.asarray(image).copy()
        if self.iterations and not self.saved:
            pixels = scramble(pixels, self.iterations)
        return self.transform(pixels.copy()), label


def _audit_test(job, model, dataset, folder):
    """检查测试样本是否与已知训练／验证样本完全重复，包括模型初始化来源。"""
    catalog = {d['id']: d for d in storage.catalog()['datasets']}
    records = {e['id']: e for e in storage.experiments()}
    source_ids, visited = set(), set()
    current = model['experiment_id']
    while current and current not in visited:
        visited.add(current)
        record = records.get(current, {})
        if record.get('dataset_id'):
            source_ids.add(record['dataset_id'])
        current = record.get('parameters', {}).get('source_model_experiment')
    protected = set()
    checked_sources = []
    for source_id in source_ids:
        check_cancelled(job)
        source = catalog.get(source_id)
        if not source:
            continue
        manifest_path = source.get('split_manifest')
        if manifest_path and Path(manifest_path).is_file():
            manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
            protected.update(row['sha256'] for row in manifest if row['split'] in ('train', 'val'))
        else:
            raw = Path(source['raw_dir'])
            if not (raw / 'train').is_dir():
                continue
            for split in ('train', 'val'):
                for path in (raw / split).rglob('*'):
                    check_cancelled(job)
                    if path.is_file() and path.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp'):
                        protected.add(sha256(path))
        checked_sources.append(source_id)
    rows = []
    for index, (path, label) in enumerate(folder.samples):
        check_cancelled(job)
        digest = sha256(path)
        if digest in protected:
            raise ValueError('所选测试集含该模型已知训练或验证图片，请选择模型原测试集或其他独立数据集')
        rows.append({'path': path, 'label': folder.classes[label], 'sha256': digest})
        if index % 64 == 0:
            job.set_progress(1 + 7 * (index + 1) / len(folder), f'核对测试样本 {index + 1}/{len(folder)}')
    warnings = []
    if not checked_sources:
        warnings.append('模型来源数据不完整，未能核对训练样本重合情况')
    if model['selection_split'] != 'val':
        warnings.append('该模型原训练未使用独立验证集；原测试成绩可能用于选模型')
    return rows, {'method': 'exact_sha256_against_known_training_and_validation',
                  'checked_source_datasets': checked_sources, 'warnings': warnings,
                  'near_duplicates_checked': False}


def run(job, model_id, dataset_id, conditions='both', iterations=None, input_source='lossless', batch_size=32):
    started = time.time()
    iterations = sorted(set(iterations or []))
    model_info, dataset = resolve_selection(model_id, dataset_id, conditions, iterations, input_source)
    check_cancelled(job)
    folder = datasets.ImageFolder(str(Path(dataset['raw_dir']) / 'test'))
    if folder.classes != model_info['_classes']:
        raise ValueError('数据集类别或类别顺序与模型不一致')
    manifest, audit = _audit_test(job, model_info, dataset, folder)
    digest = sha256(model_info['_path'])
    model = build_model(len(folder.classes), model_info['arch'])
    model.load_state_dict(torch.load(model_info['_path'], map_location='cpu', weights_only=True))
    model.eval().requires_grad_(False)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    units = [('original', 0)] if conditions in ('original', 'both') else []
    if conditions in ('scrambled', 'both'):
        units += [(f'{k:02d}', k) for k in iterations]
    experiment_id = storage.new_id('experiment')
    root = storage.experiment_root() / experiment_id
    track_output(job, root)
    parameters = {'training_mode': 'saved_model_evaluation', 'arch': model_info['arch'],
                  'model_id': model_id, 'model_name': model_info['name'],
                  'source_experiment': model_info['experiment_id'], 'source_model_sha256': digest,
                  'image_size': model_info['image_size'], 'batch_size': batch_size,
                  'conditions': conditions, 'iterations': [k for _, k in units if k],
                  'input_source': input_source, 'preprocess': model_info['_preprocess'],
                  'weights_updated': False, 'evaluation_split': 'test', 'data_audit': audit}
    record = {'id': experiment_id, 'dataset_id': dataset_id, 'dataset_name': dataset['name'],
              'created_at': time.time(), 'status': 'running', 'algorithm': 'arnold',
              'parameters': parameters, 'counts': dataset['counts'], 'results_dir': str(root / 'arnold'),
              'variants': [label for label, _ in units]}
    storage.save_experiment(record)
    summary = []
    try:
        storage.write_json(root / 'test_manifest.json', manifest)
        with (root / 'test_predictions.csv').open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['condition', 'file', 'input_file', 'true_class', 'predicted_class', 'confidence'])
            for unit_index, (label, k) in enumerate(units):
                check_cancelled(job)
                selected_folder = folder
                saved = bool(k and input_source == 'saved')
                if saved:
                    selected_folder = datasets.ImageFolder(str(Path(dataset['scrambled_dir']) / f'arnold_{k:02d}' / 'test'))
                    keys = lambda f: [(Path(p).parent.name, Path(p).name, y) for p, y in f.samples]
                    if selected_folder.classes != folder.classes or keys(selected_folder) != keys(folder):
                        raise ValueError('保存的置乱测试集与原测试集的文件、标签不一致')
                loader = DataLoader(EvaluationImages(selected_folder, model_info['_preprocess'], k, saved),
                                    batch_size=batch_size, shuffle=False, num_workers=0)
                outputs, targets = [], []
                with torch.inference_mode():
                    for batch, (images, labels) in enumerate(loader):
                        check_cancelled(job)
                        outputs.append(model(images).cpu())
                        targets.append(labels)
                        job.set_progress(9 + 86 * (unit_index + (batch + 1) / len(loader)) / len(units),
                                         f'已保存模型测试 {"原图" if not k else f"Arnold {k} 次"}：{batch + 1}/{len(loader)} 批')
                logits, labels = torch.cat(outputs), torch.cat(targets)
                report = _test_report(logits, labels, folder.classes)
                for key in ('precision', 'recall', 'f1'):
                    report['macro_' + key] = sum(v[key] for v in report['per_class'].values()) / len(folder.classes)
                metrics = {'arch': model_info['arch'], 'best_test_acc': report['accuracy'],
                           'test_report': report, 'parameters': parameters, 'weights_updated': False}
                storage.write_json(root / 'arnold' / label / 'metrics.json', metrics)
                probabilities = logits.softmax(1)
                for index, (path, target) in enumerate(folder.samples):
                    prediction = int(logits[index].argmax())
                    writer.writerow([label, path, selected_folder.samples[index][0], folder.classes[target],
                                     folder.classes[prediction], float(probabilities[index, prediction])])
                summary.append({'label': label, 'iterations': k or None, 'test_acc': report['accuracy'],
                                'test_report': report})
                storage.write_json(root / 'evaluation_summary.json', {'parameters': parameters, 'rows': summary})
        if sha256(model_info['_path']) != digest:
            raise RuntimeError('评估期间模型文件发生变化')
        begin_commit(job)
        record.update(status='done', finished_at=time.time(), model_unchanged=True,
                      elapsed_seconds=time.time() - started)
        storage.save_experiment(record)
        storage.update_dataset(dataset_id, active_experiment=experiment_id)
        job.set_progress(100, '模型测试完成，指标和逐图预测已保存')
        return {'experiment_id': experiment_id, 'summary': summary, 'model_unchanged': True}
    except Exception as exc:
        record.update(status='cancelled' if isinstance(exc, JobCancelled) else 'error',
                      error=str(exc), finished_at=time.time())
        storage.save_experiment(record)
        raise
