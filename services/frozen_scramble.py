"""使用已保存原图分类器，对同一测试集作纯像素置乱后推理。"""
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms

import config
from services import storage
from src.task_control import JobCancelled, check_cancelled, begin_commit, track_output
from services.strong_original import _test_report
from src.scrambling.arnold import scramble, period


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ScrambledTestSet(Dataset):
    """先解码为 RGB，置乱后直接标准化，不经过 JPEG 再压缩。"""
    def __init__(self, folder, preprocess, iterations, pre_scramble=None):
        self.folder, self.iterations = folder, iterations
        self.pre_scramble = pre_scramble
        self.size = tuple(preprocess["resize"])
        self.transform = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize(preprocess["mean"], preprocess["std"])])

    def __len__(self):
        return len(self.folder)

    def __getitem__(self, index):
        path, target = self.folder.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            if image.size != self.size:
                raise ValueError(f"测试图尺寸与保存模型不一致：{path}")
            if self.pre_scramble is not None:
                image = self.pre_scramble(image)
                if image.size != self.size:
                    raise ValueError("增强后的图片尺寸必须与模型输入一致")
            pixels = np.asarray(image)
        if self.iterations:
            pixels = scramble(pixels, self.iterations)
        return self.transform(pixels.copy()), target


@torch.inference_mode()
def predict(model, loader, job, condition, offset, total_batches):
    logits, targets = [], []
    for index, (images, labels) in enumerate(loader):
        logits.append(model(images).cpu())
        targets.append(labels)
        job.set_progress(3 + 94 * (offset + index + 1) / total_batches,
                         f"固定模型识别 {condition}：{index + 1}/{len(loader)} 批")
    return torch.cat(logits), torch.cat(targets)


def run(job, iterations=None):
    started = time.time()
    iterations = sorted(set(iterations or config.ARNOLD_ITERATIONS))
    pointer_path = config.MODEL_DIR / "original_classifier.json"
    if not pointer_path.exists():
        raise ValueError("尚无已保存的原图模型，请先训练原图分类器")
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    bundle = Path(pointer["bundle_dir"])
    saved = json.loads((bundle / "config.json").read_text(encoding="utf-8"))
    source = storage.experiment(pointer["experiment_id"])
    if not source or source["status"] != "done":
        raise ValueError("找不到模型对应的已完成实验")
    dataset = next(d for d in storage.catalog()["datasets"] if d["id"] == source["dataset_id"])
    folder = datasets.ImageFolder(str(Path(dataset["raw_dir"]) / "test"))
    if folder.classes != saved["classes"]:
        raise ValueError("测试集类别顺序与保存模型不一致")
    expected_total = sum(dataset["counts"]["test"].values())
    if len(folder) != expected_total:
        raise ValueError("测试集数量发生变化")
    # 验证原训练划分中的测试文件未被替换，避免比较两批不同图片。
    manifest = json.loads(Path(dataset["split_manifest"]).read_text(encoding="utf-8"))
    expected = {str(Path(row["path"])): row["sha256"] for row in manifest if row["split"] == "test"}
    if set(expected) != {str(Path(path)) for path, _ in folder.samples}:
        raise ValueError("测试集文件与原训练划分不一致")
    for path, _ in folder.samples:
        check_cancelled(job)
        if sha256(path) != expected[str(Path(path))]:
            raise ValueError(f"测试文件内容已改变：{path}")
    model_path = bundle / "model.pth"
    model_digest = sha256(model_path)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(saved["classes"]))
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval().requires_grad_(False)
    conditions = [("original", 0)] + [(f"{k:02d}", k) for k in iterations]
    experiment_id = storage.new_id("experiment")
    root = storage.experiment_root() / experiment_id
    track_output(job, root)
    parameters = {"training_mode": "frozen_model_scramble", "arch": saved["arch"],
                  "image_size": saved["image_size"], "batch_size": 64,
                  "iterations": iterations, "preprocess": saved["preprocess"],
                  "source_model_experiment": pointer["experiment_id"],
                  "model_bundle": str(bundle), "model_sha256": model_digest,
                  "weights_updated": False, "test_only": True,
                  "scramble_storage": "in_memory_lossless_pixel_permutation",
                  "arnold_period": period(saved["image_size"])}
    record = {"id": experiment_id, "dataset_id": dataset["id"], "dataset_name": dataset["name"],
              "counts": dataset["counts"], "created_at": time.time(), "status": "running",
              "algorithm": "arnold", "parameters": parameters,
              "results_dir": str(root / "arnold"), "variants": [label for label, _ in conditions]}
    storage.save_experiment(record)
    storage.write_json(root / "test_manifest.json", [
        {"file": path, "true_class": folder.classes[target], "sha256": expected[str(Path(path))]}
        for path, target in folder.samples])
    summary = []
    try:
        batches = (len(folder) + 63) // 64
        with (root / "test_predictions.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["condition", "iterations", "file", "true_class", "predicted_class",
                             "confidence", "probability_cat", "probability_dog", "correct"])
            for condition_index, (label, k) in enumerate(conditions):
                loader = DataLoader(ScrambledTestSet(folder, saved["preprocess"], k),
                                    batch_size=64, shuffle=False, num_workers=0)
                logits, targets = predict(model, loader, job, label, condition_index * batches,
                                          len(conditions) * batches)
                report = _test_report(logits, targets, folder.classes)
                if k == 0 and report["accuracy"] != source["test_accuracy"]:
                    raise ValueError("原图复测与保存结果不一致，停止本次比较")
                probs = logits.softmax(1)
                predictions = logits.argmax(1)
                report["predicted_class_counts"] = {
                    cls: int((predictions == i).sum()) for i, cls in enumerate(folder.classes)}
                report["mean_confidence"] = float(probs.max(1).values.mean())
                metrics = {"arch": saved["arch"], "classes": folder.classes,
                           "training_mode": "frozen_model_scramble", "iterations": k,
                           "best_test_acc": report["accuracy"],  # 与现有展示接口兼容，未选择最佳轮次
                           "test_accuracy": report["accuracy"], "test_report": report,
                           "model_sha256": model_digest, "weights_updated": False,
                           "elapsed_seconds": time.time() - started}
                storage.write_json(root / "arnold" / label / "metrics.json", metrics)
                for index, (path, target) in enumerate(folder.samples):
                    predicted = int(predictions[index])
                    writer.writerow([label, k, path, folder.classes[target], folder.classes[predicted],
                                     float(probs[index, predicted]), *probs[index].tolist(), predicted == target])
                stream.flush()
                summary.append({"label": label, "iterations": k, "test_acc": report["accuracy"],
                                "correct": report["correct"], "total": report["total"],
                                "confusion_matrix": report["confusion_matrix"],
                                "accuracy_95pct_wilson_interval": report["accuracy_95pct_wilson_interval"]})
                storage.write_json(root / "summary.json", summary)
        if sha256(model_path) != model_digest:
            raise RuntimeError("模型文件在识别过程中发生变化")
        _plot(summary, root / "comparison.png")
        with (root / "summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["condition", "iterations", "accuracy", "correct", "total"])
            writer.writerows([r["label"], r["iterations"], r["test_acc"], r["correct"], r["total"]] for r in summary)
        (root / "README.md").write_text(
            "# 固定原图模型识别 Arnold 置乱测试集\n\n"
            f"模型：{pointer['experiment_id']}，SHA256：{model_digest}。\n\n"
            f"对原训练实验同一批 {len(folder)} 张独立测试图进行识别，未使用训练/验证集，未更新模型参数。"
            "RGB 解码后在内存中置乱，不进行 JPEG 再压缩。每个条件使用相同样本及标签顺序。\n\n"
            "识别率下降反映该原图分类器面对置乱输入的性能变化；不能据此证明密码安全，"
            "也不能代表在置乱数据上重新训练或掌握逆变换的攻击者。\n",
            encoding="utf-8")
        begin_commit(job)
        record.update(status="done", finished_at=time.time(), model_unchanged=True,
                      elapsed_seconds=time.time() - started)
        storage.save_experiment(record)
        storage.update_dataset(dataset["id"], active_experiment=experiment_id)
        job.set_progress(100, "固定模型置乱测试完成，结果已保存")
        return {"experiment_id": experiment_id, "summary": summary, "model_unchanged": True}
    except Exception as exc:
        record.update(status="cancelled" if isinstance(exc, JobCancelled) else "error",
                      error=str(exc), finished_at=time.time())
        storage.save_experiment(record)
        raise


def _plot(summary, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    accuracy = [r["test_acc"] * 100 for r in summary]
    lower = [100 * (r["test_acc"] - r["accuracy_95pct_wilson_interval"][0]) for r in summary]
    upper = [100 * (r["accuracy_95pct_wilson_interval"][1] - r["test_acc"]) for r in summary]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.errorbar(range(len(summary)), accuracy, yerr=[lower, upper], marker="o", capsize=4)
    ax.axhline(50, color="red", linestyle="--", label="Balanced-class chance baseline")
    for index, value in enumerate(accuracy):
        ax.annotate(f"{value:.2f}%", (index, value), xytext=(0, 12), textcoords="offset points", ha="center")
    ax.set(xticks=range(len(summary)), xticklabels=["Original"] + [f"Arnold {r['iterations']}" for r in summary[1:]],
           ylim=(0, 108), ylabel="Accuracy (%)", title="Saved ResNet-18 | same 2,000 test images | no retraining")
    ax.grid(alpha=.2)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
