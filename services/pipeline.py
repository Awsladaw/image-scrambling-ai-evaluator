"""数据导入、置乱、训练编排；每次执行创建独立版本，完成后发布。"""
import json
import time
from pathlib import Path
import config
from src.data import prepare
from src.scrambling import batch_scramble
from src import train
from services import storage
from src.task_control import JobCancelled, check_cancelled, begin_commit, track_output

ALGORITHMS = {"arnold": "arnold"}


def _dataset():
    dataset = storage.active_dataset()
    if not dataset:
        raise ValueError("请先导入完整的数据集")
    return dataset


def _scrambled_variants(dataset=None):
    dataset = dataset or storage.active_dataset()
    if not dataset or not dataset.get("scrambled_dir"):
        return []
    return [(d.name.replace("arnold_", ""), d)
            for d in sorted(Path(dataset["scrambled_dir"]).glob("arnold_*")) if d.is_dir()]


def get_scrambled_labels():
    return [label for label, _ in _scrambled_variants()]


def get_evaluated_labels():
    return [r["label"] for r in get_metrics()["rows"]]


def import_dataset(zip_path, max_per_class=None, job=None, source_name=None, validation_ratio=0.0):
    check_cancelled(job)
    storage.catalog()
    dataset_id = storage.new_id("dataset")
    root = config.DATA_ROOT / "datasets" / dataset_id
    track_output(job, root)
    raw = prepare.import_dataset(
        zip_path, data_root=root, max_per_class=max_per_class,
        progress_callback=job.set_progress if job else None, validation_ratio=validation_ratio,
        cancel_callback=lambda: check_cancelled(job))
    if not storage.is_ready(raw):
        raise ValueError("导入后训练集或测试集为空；请增加样本量并检查图片是否损坏")
    dataset = {"id": dataset_id, "name": source_name or Path(zip_path).name,
               "created_at": time.time(), "status": "ready", "raw_dir": str(raw),
               "counts": storage.counts(raw), "scrambled_dir": None,
               "scramble_id": None, "active_experiment": None,
               "max_per_class": max_per_class, "image_size": config.IMAGE_SIZE,
               "seed": config.SEED, "split_ratio": config.SPLIT_RATIO,
               "validation_ratio": validation_ratio}
    begin_commit(job)
    storage.write_json(root / "dataset.json", dataset)
    storage.publish_dataset(dataset)
    return {"raw": str(raw), "dataset_id": dataset_id, "counts": dataset["counts"]}


def run_scramble(algorithm="arnold", iterations=None, job=None):
    check_cancelled(job)
    dataset = _dataset()
    scramble_id = storage.new_id("scramble")
    out_root = config.DATA_ROOT / "datasets" / dataset["id"] / "scrambles" / scramble_id
    track_output(job, out_root)
    meta = batch_scramble.scramble_dataset(
        raw_dir=dataset["raw_dir"], out_root=out_root,
        iterations=iterations or config.ARNOLD_ITERATIONS, algorithm=algorithm,
        progress_callback=job.set_progress if job else None,
        cancel_callback=lambda: check_cancelled(job))
    for k in meta["iterations"]:
        if storage.counts(out_root / f"{algorithm}_{k:02d}") != dataset["counts"]:
            raise RuntimeError("置乱图像数量与原始数据不一致，旧版本仍保留")
    begin_commit(job)
    storage.update_dataset(dataset["id"], scrambled_dir=str(out_root),
                           scramble_id=scramble_id, active_experiment=None)
    return {**meta, "scramble_id": scramble_id}


def run_evaluate(arch=None, epochs=None, batch_size=None, lr=None,
                 optimizer=None, job=None, include_baseline=True, image_size=None):
    dataset = _dataset()
    arch, epochs = arch or config.ARCH, epochs if epochs is not None else config.EPOCHS
    batch_size, lr = batch_size or config.BATCH_SIZE, lr or config.LEARNING_RATE
    optimizer = optimizer or config.OPTIMIZER
    image_size = image_size or config.IMAGE_SIZE
    units = [("original", Path(dataset["raw_dir"]))] if include_baseline else []
    units += _scrambled_variants(dataset)
    if not units:
        raise ValueError("没有可评估数据，请先执行置乱")
    experiment_id = storage.new_id("experiment")
    root = storage.experiment_root() / experiment_id
    track_output(job, root)
    record = {"id": experiment_id, "dataset_id": dataset["id"],
              "dataset_name": dataset["name"], "counts": dataset["counts"],
              "scramble_id": dataset.get("scramble_id"), "created_at": time.time(),
              "status": "running", "algorithm": "arnold", "results_dir": str(root / "arnold"),
              "parameters": {"arch": arch, "epochs": epochs, "batch_size": batch_size,
                             "lr": lr, "optimizer": optimizer, "device": config.DEVICE,
                             "image_size": image_size, "seed": config.SEED,
                             "selection_split": "val" if "val" in dataset["counts"] else "test"},
              "variants": [label for label, _ in units]}
    storage.save_experiment(record)
    summary = []
    try:
        for index, (label, data_dir) in enumerate(units):
            check_cancelled(job)
            def cb(percent, message, index=index, label=label):
                if job:
                    job.set_progress((index + percent / 100) / len(units) * 100,
                                     f"[{label}] {message}")
            metrics = train.run(data_dir=data_dir, out_dir=root / "arnold" / label,
                                arch=arch, epochs=epochs, batch_size=batch_size,
                                lr=lr, optimizer=optimizer, image_size=image_size, progress_callback=cb,
                                cancel_callback=lambda: check_cancelled(job))
            summary.append({"label": label, "test_acc": metrics["best_test_acc"]})
        _save_summary(summary, root / "arnold_summary.csv")
        begin_commit(job)
        record.update(status="done", finished_at=time.time())
        storage.save_experiment(record)
        storage.update_dataset(dataset["id"], active_experiment=experiment_id)
    except Exception as exc:
        record.update(status="cancelled" if isinstance(exc, JobCancelled) else "error",
                      error=str(exc), finished_at=time.time())
        storage.save_experiment(record)
        raise
    return {"experiment_id": experiment_id, "summary": summary, "ranking": _ranking(summary)}


def _save_summary(summary, path):
    Path(path).write_text("label,recognition_rate\n" + "".join(
        f"{r['label']},{r['test_acc']:.4f}\n" for r in summary), encoding="utf-8")


def _ranking(summary):
    return sorted((r for r in summary if r["label"] != "original"),
                  key=lambda r: abs(r["test_acc"] - 0.5))


def get_metrics(experiment_id=None):
    dataset = storage.active_dataset()
    experiment_id = experiment_id or (dataset or {}).get("active_experiment")
    experiment = storage.experiment(experiment_id) if experiment_id else None
    if experiment_id and not experiment:
        raise ValueError("实验不存在")
    rows = []
    if experiment:
        for mf in sorted(Path(experiment["results_dir"]).glob("*/metrics.json")):
            m = json.loads(mf.read_text(encoding="utf-8"))
            label = mf.parent.name
            report = m.get('test_report', {})
            per_class = list(report.get('per_class', {}).values())
            def macro(key):
                return report.get('macro_' + key, sum(v[key] for v in per_class) / len(per_class) if per_class else None)
            rows.append({"label": label, "iterations": int(label) if label.isdigit() else None,
                         "test_acc": m.get("best_test_acc"), "arch": m.get("arch"),
                         'precision': macro('precision'), 'recall': macro('recall'), 'f1': macro('f1'),
                         'total': report.get('total'), 'confusion_matrix': report.get('confusion_matrix'),
                         'classes': report.get('confusion_matrix_order')})
    rows.sort(key=lambda r: -1 if r["iterations"] is None else r["iterations"])
    return {"rows": rows, "ranking": _ranking(rows), "baseline": 0.5,
            "algorithm": "arnold", "experiment": experiment}
