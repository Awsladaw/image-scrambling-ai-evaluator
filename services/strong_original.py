"""预训练 ResNet-18 原图分类器：独立划分、验证集调参、一次最终测试及导出。"""
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import time
from importlib.metadata import version

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models

import config
from services import storage
from src.modeling import get_transforms
from src.task_control import JobCancelled, check_cancelled, begin_commit, track_output


def fuse_head(head, mean, scale):
    """把训练集特征标准化合并到分类层，导出普通 ResNet state_dict。"""
    fused = nn.Linear(head.in_features, head.out_features)
    with torch.no_grad():
        fused.weight.copy_(head.weight / scale)
        fused.bias.copy_(head.bias - (head.weight * mean / scale).sum(dim=1))
    return fused


def fit_head(features, targets, penalty, max_iter=80, cancel_callback=None):
    torch.manual_seed(config.SEED)
    head = nn.Linear(features.shape[1], 2)
    optimizer = torch.optim.LBFGS(head.parameters(), lr=1.0, max_iter=max_iter,
                                 history_size=10, line_search_fn="strong_wolfe")
    criterion = nn.CrossEntropyLoss()

    def closure():
        if cancel_callback:
            cancel_callback()
        optimizer.zero_grad()
        loss = criterion(head(features), targets) + penalty * head.weight.square().sum() / 2
        loss.backward()
        return loss

    optimizer.step(closure)
    return head.eval(), optimizer.state[head.weight].get("n_iter", 0)


def _prepare_dataset(job):
    catalog = storage.catalog()
    source = max(catalog["datasets"], key=lambda d: sum(sum(v.values()) for v in d["counts"].values()))
    source_raw = Path(source["raw_dir"])
    excluded = set()
    for filename in ("original-baseline-sample.json", "run-once-sample.json"):
        manifest = config.ROOT / "tmp" / filename
        if manifest.exists():
            for files in json.loads(manifest.read_text(encoding="utf-8"))["files"].values():
                excluded.update(Path(f).name for f in files)
    for previous in catalog["datasets"]:
        manifest = previous.get("split_manifest")
        if manifest and Path(manifest).exists():
            excluded.update(Path(row["source"]).name for row in
                            json.loads(Path(manifest).read_text(encoding="utf-8")))
    # 最初每类 60 张的子采样使用同一公开数据源和 seed=42，排除其源文件索引。
    early_indices = np.random.default_rng(42).choice(12500, size=60, replace=False)
    for cls in ("cat", "dog"):
        excluded.update(f"{cls}_{i:05d}.jpg" for i in early_indices)
    dataset_id = storage.new_id("dataset")
    root = config.DATA_ROOT / "datasets" / dataset_id
    track_output(job, root)
    raw = root / "raw"
    rng = random.Random(config.SEED)
    content_seen, visual_seen, records = set(), set(), []
    rejected = {"previous_experiment": 0, "duplicate": 0, "invalid": 0}
    sizes = {"train": 3000, "val": 500, "test": 1000}
    for cls in ("cat", "dog"):
        files = sorted(source_raw.glob(f"*/{cls}/*"))
        files = [f for f in files if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
        rng.shuffle(files)
        chosen = []
        for f in files:
            check_cancelled(job)
            if f.name in excluded:
                rejected["previous_experiment"] += 1
                continue
            digest = hashlib.sha256(f.read_bytes()).hexdigest()
            try:
                with Image.open(f) as im:
                    im = im.convert("RGB")
                    im.load()
                    small = np.asarray(im.convert("L").resize((9, 8)), dtype=np.int16)
                    visual = np.packbits(small[:, 1:] > small[:, :-1]).tobytes().hex()
            except Exception:
                rejected["invalid"] += 1
                continue
            if digest in content_seen or visual in visual_seen:
                rejected["duplicate"] += 1
                continue
            content_seen.add(digest)
            visual_seen.add(visual)
            chosen.append((f, digest, visual))
            if len(chosen) == sum(sizes.values()):
                break
        if len(chosen) < sum(sizes.values()):
            raise ValueError("可用的未使用图片不足，请先导入完整猫狗数据集")
        offset = 0
        for split, count in sizes.items():
            dst = raw / split / cls
            dst.mkdir(parents=True, exist_ok=True)
            for index, (f, digest, visual) in enumerate(chosen[offset:offset + count]):
                check_cancelled(job)
                target = dst / f"{cls}_{index:05d}{f.suffix.lower()}"
                shutil.copy2(f, target)
                records.append({"split": split, "class": cls, "source": str(f),
                                "path": str(target), "sha256": digest, "dhash": visual})
            offset += count
            job.set_progress(2 + len(records) / 9000 * 8, f"准备原图数据 {len(records)}/9000")
    dataset = {"id": dataset_id, "name": "original-transfer-9000",
               "created_at": time.time(), "status": "ready", "raw_dir": str(raw),
               "counts": storage.counts(raw), "scrambled_dir": None, "scramble_id": None,
               "active_experiment": None, "image_size": 224, "seed": config.SEED,
               "source_dataset": source["id"], "split_manifest": str(root / "split_manifest.json"),
               "deduplication": "SHA256 plus exact 64-bit dHash match; not exhaustive near-duplicate detection",
               "rejected": rejected, "previous_experiment_files_excluded": len(excluded)}
    storage.write_json(root / "split_manifest.json", records)
    storage.write_json(root / "dataset.json", dataset)
    check_cancelled(job)
    return dataset


@torch.inference_mode()
def _features(backbone, loader, job, start, end, label):
    features, targets = [], []
    for index, (images, labels) in enumerate(loader):
        check_cancelled(job)
        features.append(backbone(images).cpu())
        targets.append(labels)
        job.set_progress(start + (end - start) * (index + 1) / len(loader),
                         f"{label} {index + 1}/{len(loader)} 批")
    return torch.cat(features), torch.cat(targets)


def _score(head, features, labels):
    with torch.no_grad():
        logits = head(features)
        return ((logits.argmax(1) == labels).float().mean().item(),
                nn.functional.cross_entropy(logits, labels).item())


def _test_report(logits, targets, classes):
    predictions = logits.argmax(1)
    correct = int((predictions == targets).sum())
    n, p, z = len(targets), correct / len(targets), 1.959963984540054
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    matrix = torch.bincount(targets * 2 + predictions, minlength=4).reshape(2, 2).tolist()
    per_class = {}
    for i, cls in enumerate(classes):
        tp = matrix[i][i]
        precision = tp / max(1, sum(row[i] for row in matrix))
        recall = tp / max(1, sum(matrix[i]))
        per_class[cls] = {"precision": precision, "recall": recall,
                          "f1": 2 * precision * recall / max(1e-12, precision + recall),
                          "support": sum(matrix[i])}
    return {"accuracy": p, "correct": correct, "total": n,
            "accuracy_95pct_wilson_interval": [center - half, center + half],
            "confusion_matrix": matrix, "confusion_matrix_order": classes,
            "per_class": per_class}


def run(job):
    started = time.time()
    torch.set_num_threads(min(8, torch.get_num_threads()))
    torch.manual_seed(config.SEED)
    job.set_progress(0, "准备独立原图数据，排除此前实验使用的图片…")
    dataset = _prepare_dataset(job)
    experiment_id = storage.new_id("experiment")
    root = storage.experiment_root() / experiment_id
    track_output(job, root)
    output = root / "arnold" / "original"
    output.mkdir(parents=True)
    parameters = {"training_mode": "transfer_linear_probe", "pretrained": True,
                  "pretrained_weights": "ResNet18_Weights.IMAGENET1K_V1", "arch": "resnet18",
                  "pretrained_weights_url": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
                  "torch_version": str(torch.__version__), "torchvision_version": version("torchvision"),
                  "backbone_frozen": True, "image_size": 224, "batch_size": 64,
                  "solver": "LBFGS", "max_iterations": 80, "head_learning_rate": 1.0,
                  "penalties": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0],
                  "selection_split": "val", "seed": config.SEED,
                  "preprocess": {"resize": [224, 224], "mean": [.485, .456, .406],
                                 "std": [.229, .224, .225]}, "augmentation": "none"}
    record = {"id": experiment_id, "dataset_id": dataset["id"], "dataset_name": dataset["name"],
              "counts": dataset["counts"], "created_at": time.time(), "status": "running",
              "algorithm": "original", "parameters": parameters, "results_dir": str(root / "arnold"),
              "variants": ["original"]}
    storage.save_experiment(record)
    try:
        torch.hub.set_dir(str(config.DATA_ROOT / "cache" / "torch"))
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        backbone = models.resnet18(weights=weights).eval()
        backbone.fc = nn.Identity()
        for parameter in backbone.parameters():
            parameter.requires_grad_(False)
        transform = get_transforms(224)["eval"]
        image_sets = {split: datasets.ImageFolder(str(Path(dataset["raw_dir"]) / split), transform=transform)
                      for split in ("train", "val", "test")}
        loaders = {split: DataLoader(images, batch_size=64, shuffle=False, num_workers=0)
                   for split, images in image_sets.items()}
        train_x, train_y = _features(backbone, loaders["train"], job, 10, 61, "提取训练特征")
        val_x, val_y = _features(backbone, loaders["val"], job, 61, 70, "提取验证特征")
        # inference_mode 张量需复制为普通张量，才能参与分类头的梯度计算。
        train_x, val_x = train_x.clone(), val_x.clone()
        train_y, val_y = train_y.clone(), val_y.clone()
        mean = train_x.mean(0)
        scale = train_x.std(0).clamp_min(1e-6)
        train_z, val_z = (train_x - mean) / scale, (val_x - mean) / scale
        best, trials = None, []
        for index, penalty in enumerate(parameters["penalties"]):
            job.set_progress(70 + index / len(parameters["penalties"]) * 15,
                             f"验证集调参 {index + 1}/{len(parameters['penalties'])}，正则系数 {penalty}")
            head, iterations = fit_head(train_z, train_y, penalty,
                                        cancel_callback=lambda: check_cancelled(job))
            accuracy, loss = _score(head, val_z, val_y)
            train_accuracy, _ = _score(head, train_z, train_y)
            trial = {"penalty": penalty, "iterations": iterations,
                     "train_accuracy": train_accuracy, "validation_accuracy": accuracy,
                     "validation_loss": loss}
            trials.append(trial)
            storage.write_json(root / "tuning_trials.json", trials)
            if best is None or (accuracy, -loss) > (best[0], -best[1]):
                best = (accuracy, loss, head, trial)
                saved_head = fuse_head(head, mean, scale)
                saved_weights = dict(backbone.state_dict())
                saved_weights.update({"fc." + key: value for key, value in saved_head.state_dict().items()})
                torch.save(saved_weights, output / "best.pth")
        validation_accuracy, _, head, selected = best
        fused = fuse_head(head, mean, scale).eval()
        parameters["selected_penalty"] = selected["penalty"]
        # 确定所有参数后才读取测试图像特征。
        test_x, test_y = _features(backbone, loaders["test"], job, 85, 97, "独立测试")
        with torch.no_grad():
            logits = fused(test_x)
        report = _test_report(logits, test_y, image_sets["train"].classes)
        backbone.fc = fused
        torch.save(backbone.state_dict(), output / "best.pth")
        metrics = {"arch": "resnet18", "classes": image_sets["train"].classes,
                   "best_test_acc": report["accuracy"], "best_validation_acc": validation_accuracy,
                   "selection_split": "val", "test_evaluated_once": True,
                   "training_mode": "transfer_linear_probe", "pretrained": True,
                   "parameters": parameters, "selected_trial": selected, "tuning_trials": trials,
                   "counts": dataset["counts"], "test_report": report,
                   "elapsed_seconds": time.time() - started}
        storage.write_json(output / "metrics.json", metrics)
        with (root / "test_predictions.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["file", "true_class", "predicted_class", "confidence"])
            probabilities = logits.softmax(1)
            for index, (path, label) in enumerate(image_sets["test"].samples):
                prediction = int(logits[index].argmax())
                writer.writerow([path, image_sets["test"].classes[label],
                                 image_sets["test"].classes[prediction], float(probabilities[index, prediction])])
        bundle = config.MODEL_DIR / "original_classifier" / experiment_id
        track_output(job, bundle)
        bundle.mkdir(parents=True)
        shutil.copy2(output / "best.pth", bundle / "model.pth")
        storage.write_json(bundle / "config.json", {**parameters, "classes": image_sets["train"].classes,
                                                     "experiment_id": experiment_id})
        storage.write_json(bundle / "metrics.json", metrics)
        (bundle / "README.md").write_text(
            "# 原图猫狗分类器\n\n预训练 ResNet-18 固定特征，分类头使用验证集选择正则系数。\n"
            "\n输入转换：RGB，Resize(224,224)，ToTensor，ImageNet mean/std 标准化。\n"
            f"\n独立测试准确率：{report['accuracy']:.2%}，共 {report['total']} 张。\n"
            "\n识别新图片：`.venv\\Scripts\\python.exe tools\\predict_original.py 图片路径`。\n",
            encoding="utf-8")
        begin_commit(job)
        storage.publish_dataset(dataset)
        storage.write_json(config.MODEL_DIR / "original_classifier.json", {"bundle_dir": str(bundle),
                           "experiment_id": experiment_id, "accuracy": report["accuracy"]})
        record.update(status="done", finished_at=time.time(), parameters=parameters,
                      model_bundle=str(bundle), test_accuracy=report["accuracy"])
        storage.save_experiment(record)
        storage.update_dataset(dataset["id"], active_experiment=experiment_id)
        job.set_progress(100, f"原图模型已保存，独立测试准确率 {report['accuracy']:.2%}")
        return {"experiment_id": experiment_id, "model_bundle": str(bundle),
                "test_accuracy": report["accuracy"], "validation_accuracy": validation_accuracy}
    except Exception as exc:
        record.update(status="cancelled" if isinstance(exc, JobCancelled) else "error",
                      error=str(exc), finished_at=time.time())
        storage.save_experiment(record)
        raise
