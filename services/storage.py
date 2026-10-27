"""数据集版本及实验目录；已有工作区登记为旧版数据集，不移动原文件。"""
import json
import threading
import time
import uuid
from pathlib import Path
import config

_lock = threading.RLock()


def new_id(prefix):
    return f"{prefix}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def counts(raw_dir):
    raw_dir = Path(raw_dir)
    return {split: {cls: sum(1 for p in (raw_dir / split / cls).glob("*")
                            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"))
                    for cls in ("cat", "dog")}
            for split in (("train", "val", "test") if (raw_dir / "val").exists() else ("train", "test"))}


def is_ready(raw_dir):
    return all(n > 0 for split in counts(raw_dir).values() for n in split.values())


def _catalog():
    path = config.DATA_ROOT / "catalog.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    catalog = {"active_dataset": None, "datasets": []}
    if is_ready(config.RAW_DIR):
        dataset = {"id": new_id("legacy"), "name": "已有猫狗数据集",
                   "created_at": time.time(), "status": "ready",
                   "raw_dir": str(config.RAW_DIR), "counts": counts(config.RAW_DIR),
                   "scrambled_dir": str(config.SCRAMBLED_DIR), "scramble_id": "legacy",
                   "active_experiment": None}
        if list((config.RESULTS_DIR / "arnold").glob("*/metrics.json")):
            experiment = {"id": new_id("legacy-exp"), "dataset_id": dataset["id"],
                          "dataset_name": dataset["name"], "created_at": time.time(),
                          "status": "done", "algorithm": "arnold",
                          "results_dir": str(config.RESULTS_DIR / "arnold"),
                          "parameters": {"note": "旧版实验，未完整记录训练参数"}}
            save_experiment(experiment)
            dataset["active_experiment"] = experiment["id"]
        catalog["datasets"].append(dataset)
        catalog["active_dataset"] = dataset["id"]
    write_json(path, catalog)
    return catalog


def catalog():
    with _lock:
        return _catalog()


def active_dataset():
    c = catalog()
    return next((d for d in c["datasets"] if d["id"] == c["active_dataset"]), None)


def publish_dataset(dataset):
    with _lock:
        c = _catalog()
        c["datasets"].append(dataset)
        c["active_dataset"] = dataset["id"]
        write_json(config.DATA_ROOT / "catalog.json", c)


def update_dataset(dataset_id, **changes):
    with _lock:
        c = _catalog()
        dataset = next(d for d in c["datasets"] if d["id"] == dataset_id)
        dataset.update(changes)
        write_json(config.DATA_ROOT / "catalog.json", c)
        return dataset


def select_dataset(dataset_id):
    with _lock:
        c = _catalog()
        if not any(d["id"] == dataset_id and d["status"] == "ready" for d in c["datasets"]):
            raise ValueError("数据集不存在或尚未准备完成")
        c["active_dataset"] = dataset_id
        write_json(config.DATA_ROOT / "catalog.json", c)
        return {"dataset_id": dataset_id}


def experiment_root():
    return config.RESULTS_DIR / "experiments"


def save_experiment(experiment):
    write_json(experiment_root() / experiment["id"] / "experiment.json", experiment)


def experiments():
    catalog()
    rows = [json.loads(f.read_text(encoding="utf-8"))
            for f in experiment_root().glob("*/experiment.json")]
    return sorted(rows, key=lambda e: e["created_at"], reverse=True)


def experiment(experiment_id):
    return next((e for e in experiments() if e["id"] == experiment_id), None)
