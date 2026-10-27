"""在固定次数的 Arnold 置乱训练集上微调全部 ResNet-18 层。"""
import csv
import json
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

import config
from services import storage
from src.task_control import JobCancelled, check_cancelled, begin_commit, track_output
from services.frozen_scramble import ScrambledTestSet, sha256
from services.strong_original import _test_report


def save_checkpoint(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def build_optimizer(model, backbone_lr, head_lr, weight_decay):
    """全部参数可训练，分类层使用独立学习率。"""
    model.requires_grad_(True)
    backbone = [p for name, p in model.named_parameters() if not name.startswith("fc.")]
    return torch.optim.AdamW([
        {"params": backbone, "lr": backbone_lr, "name": "backbone"},
        {"params": list(model.fc.parameters()), "lr": head_lr, "name": "head"},
    ], weight_decay=weight_decay)


def build_augmentation(size):
    """只增强原图，再由 Dataset 作置乱；不改变验证/测试输入。"""
    return transforms.Compose([
        transforms.RandomResizedCrop(tuple(size), scale=(.85, 1.), ratio=(.9, 1.1)),
        transforms.RandomHorizontalFlip(p=.5),
        transforms.ColorJitter(brightness=.1, contrast=.1, saturation=.1),
    ])


def matching_reference(dataset_id, source_digest, manifest_digest, iterations, epochs, batch_size, backbone_lr, head_lr):
    for previous in storage.experiments():
        p = previous.get("parameters", {})
        if (previous["status"] == "done" and previous["dataset_id"] == dataset_id
            and p.get("training_mode") == "adaptive_scramble_finetune"
            and p.get("augmentation") == "none" and p.get("source_model_sha256") == source_digest
            and p.get("source_split_manifest_sha256") == manifest_digest
            and p.get("iterations") == [iterations] and p.get("epochs") == epochs
            and p.get("batch_size") == batch_size and p.get("lr") == backbone_lr
            and p.get("head_lr") == head_lr and p.get("seed") == config.SEED):
            return previous
    return None


def validation_promotion(score, parameters):
    """默认模型仅由同任务的验证成绩决定，测试准确率不参与。"""
    pointer_path = config.MODEL_DIR / "scrambled_classifier.json"
    if not pointer_path.exists():
        return True, None
    current = json.loads(pointer_path.read_text(encoding="utf-8"))
    metrics = json.loads((Path(current["bundle_dir"]) / "metrics.json").read_text(encoding="utf-8"))
    previous = metrics.get("parameters", {})
    if any(previous.get(key) != parameters.get(key) for key in
           ("source_model_sha256", "source_split_manifest_sha256", "iterations")):
        return True, current["experiment_id"]
    loss = metrics.get("best_validation_loss")
    if loss is None:
        history = json.loads((storage.experiment_root() / current["experiment_id"] / "training_history.json").read_text(encoding="utf-8"))
        loss = next(r["validation_loss"] for r in history if r["epoch"] == metrics["best_epoch"])
    return score >= (metrics["best_validation_acc"], -loss), current["experiment_id"]


def train_epoch(model, loader, optimizer, callback, use_bf16=True):
    model.train()
    loss_sum, correct, total = 0., 0, 0
    for batch, (images, labels) in enumerate(loader):
        images = images.contiguous(memory_format=torch.channels_last)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=use_bf16):
            logits = model(images)
            loss = nn.functional.cross_entropy(logits, labels, label_smoothing=.05)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        loss_sum += float(loss.detach()) * len(labels)
        correct += int((logits.detach().argmax(1) == labels).sum())
        total += len(labels)
        callback(batch + 1, len(loader), correct / total, loss_sum / total)
    return {"accuracy": correct / total, "loss": loss_sum / total}


@torch.inference_mode()
def evaluate(model, loader, callback=None):
    model.eval()
    logits, targets, loss_sum = [], [], 0.
    for batch, (images, labels) in enumerate(loader):
        output = model(images.contiguous(memory_format=torch.channels_last)).float()
        logits.append(output.cpu())
        targets.append(labels)
        loss_sum += float(nn.functional.cross_entropy(output, labels, reduction="sum"))
        if callback:
            callback(batch + 1, len(loader))
    logits, targets = torch.cat(logits), torch.cat(targets)
    return logits, targets, loss_sum / len(targets)


def run(job, iterations=2, epochs=10, batch_size=32, backbone_lr=3e-4, head_lr=1e-3, augmentation=False):
    started = time.time()
    torch.set_num_threads(8)
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)
    pointer = json.loads((config.MODEL_DIR / "original_classifier.json").read_text(encoding="utf-8"))
    bundle = Path(pointer["bundle_dir"])
    source_model = bundle / "model.pth"
    source_digest = sha256(source_model)
    settings = json.loads((bundle / "config.json").read_text(encoding="utf-8"))
    source = storage.experiment(pointer["experiment_id"])
    if not source or source["status"] != "done":
        raise ValueError("原图模型的训练记录不存在或未完成")
    dataset = next(d for d in storage.catalog()["datasets"] if d["id"] == source["dataset_id"])
    raw = Path(dataset["raw_dir"])
    manifest = json.loads(Path(dataset["split_manifest"]).read_text(encoding="utf-8"))
    # 必须继承原始划分；验证每张文件，禁止同一原图跨训练/验证/测试集。
    seen = {}
    for index, row in enumerate(manifest):
        check_cancelled(job)
        path, split, digest = Path(row["path"]), row["split"], row["sha256"]
        if split not in ("train", "val", "test") or path.parent.parent != raw / split:
            raise ValueError("原始划分清单无效")
        if digest in seen and seen[digest] != split:
            raise ValueError("同一原图跨数据划分，无法训练")
        seen[digest] = split
        if sha256(path) != digest:
            raise ValueError(f"源图片内容发生变化：{path}")
        if index % 300 == 0:
            job.set_progress(2 * (index + 1) / len(manifest), f"核对原训练划分 {index + 1}/{len(manifest)}")
    folders = {split: datasets.ImageFolder(str(raw / split)) for split in ("train", "val")}
    for split, folder in folders.items():
        expected = {str(Path(r["path"])) for r in manifest if r["split"] == split}
        if folder.classes != settings["classes"] or {str(Path(p)) for p, _ in folder.samples} != expected:
            raise ValueError("训练/验证集与原模型划分不一致")
    generator = torch.Generator().manual_seed(config.SEED)
    loaders = {split: DataLoader(ScrambledTestSet(folder, settings["preprocess"], iterations,
                                pre_scramble=build_augmentation(settings["preprocess"]["resize"])
                                if augmentation and split == "train" else None),
                                batch_size=batch_size, shuffle=split == "train", num_workers=0,
                                generator=generator if split == "train" else None)
               for split, folder in folders.items()}
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(settings["classes"]))
    model.load_state_dict(torch.load(source_model, map_location="cpu", weights_only=True))
    model = model.to(memory_format=torch.channels_last)
    optimizer = build_optimizer(model, backbone_lr, head_lr, .01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=.3, patience=1, threshold=.002, threshold_mode="abs", min_lr=1e-6)
    experiment_id = storage.new_id("experiment")
    root = storage.experiment_root() / experiment_id
    track_output(job, root)
    output = root / "arnold" / f"{iterations:02d}"
    parameters = {"training_mode": "adaptive_scramble_finetune", "arch": "resnet18",
                  "iterations": [iterations], "epochs": epochs, "batch_size": batch_size,
                  "lr": backbone_lr, "head_lr": head_lr, "optimizer": "adamw", "weight_decay": .01,
                  "label_smoothing": .05, "gradient_clip_norm": 5., "all_layers_trainable": True,
                  "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                  "pretrained": True, "initialization": "saved_original_classifier",
                  "source_model_experiment": source["id"], "source_model_sha256": source_digest,
                  "source_split_manifest_sha256": sha256(dataset["split_manifest"]),
                  "image_size": settings["image_size"], "preprocess": settings["preprocess"],
                  "precision": "cpu_bfloat16_training_float32_validation_and_test", "num_threads": 8,
                  "memory_format": "channels_last", "selection_split": "val",
                  "selection_criterion": "highest_accuracy_then_lowest_cross_entropy",
                  "scheduler": "ReduceLROnPlateau(val_accuracy, factor=0.3, patience=1)",
                  "early_stopping_patience": 3, "early_stopping_min_delta": .002, "minimum_epochs": 4,
                  "augmentation": "pre_scramble_light_v1" if augmentation else "none", "seed": config.SEED,
                  "scramble_storage": "in_memory_lossless_pixel_permutation", "inverse_transform_used": False}
    if augmentation:
        parameters["augmentation_settings"] = {
            "order": ["random_resized_crop", "horizontal_flip", "color_jitter", "arnold", "normalize"],
            "crop_scale": [.85, 1.], "crop_ratio": [.9, 1.1], "horizontal_flip_probability": .5,
            "brightness": .1, "contrast": .1, "saturation": .1,
            "applied_to": "train_only", "validation_and_test_augmented": False}
    reference = matching_reference(dataset["id"], source_digest, parameters["source_split_manifest_sha256"],
                                   iterations, epochs, batch_size, backbone_lr, head_lr) if augmentation else None
    reference_digest = sha256(Path(reference["model_bundle"]) / "model.pth") if reference else None
    parameters["comparison_reference"] = reference["id"] if reference else None
    record = {"id": experiment_id, "dataset_id": dataset["id"], "dataset_name": dataset["name"],
              "counts": dataset["counts"], "created_at": time.time(), "status": "running",
              "algorithm": "arnold", "parameters": parameters, "results_dir": str(root / "arnold"),
              "variants": [f"{iterations:02d}"]}
    storage.save_experiment(record)
    shutil.copy2(dataset["split_manifest"], root / "split_manifest.json")
    history, best, early_best, stale = [], None, -1., 0
    training_span = 80 if augmentation else 87
    try:
        for epoch in range(1, epochs + 1):
            epoch_started = time.time()
            rates = {g["name"]: g["lr"] for g in optimizer.param_groups}
            def progress(batch, total_batches, accuracy, loss):
                job.set_progress(3 + training_span * ((epoch - 1) + .88 * batch / total_batches) / epochs,
                                 f"Arnold {iterations} 次微调：第 {epoch}/{epochs} 轮，"
                                 f"训练 {batch}/{total_batches} 批，准确率 {accuracy:.1%}，损失 {loss:.3f}")
            train = train_epoch(model, loaders["train"], optimizer, progress)
            def validation_progress(batch, total_batches):
                job.set_progress(3 + training_span * ((epoch - 1) + .88 + .12 * batch / total_batches) / epochs,
                                 f"第 {epoch}/{epochs} 轮：验证 {batch}/{total_batches} 批")
            logits, labels, val_loss = evaluate(model, loaders["val"], validation_progress)
            val_accuracy = int((logits.argmax(1) == labels).sum()) / len(labels)
            row = {"epoch": epoch, "train_accuracy": train["accuracy"], "train_loss": train["loss"],
                   "validation_accuracy": val_accuracy, "validation_loss": val_loss,
                   "learning_rates": rates, "elapsed_seconds": time.time() - epoch_started}
            history.append(row)
            score = (val_accuracy, -val_loss)
            if best is None or score > best[0]:
                best = (score, epoch)
                save_checkpoint(output / "best.pth", model.state_dict())
            if val_accuracy > early_best + .002:
                early_best, stale = val_accuracy, 0
            else:
                stale += 1
            scheduler.step(val_accuracy)
            storage.write_json(root / "training_history.json", history)
            save_checkpoint(root / "last_checkpoint.pth", {
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "epoch": epoch, "history": history,
                "torch_rng_state": torch.get_rng_state(), "shuffle_rng_state": generator.get_state(),
                "parameters": parameters})
            record.update(completed_epochs=epoch, best_epoch=best[1], best_validation_accuracy=best[0][0])
            storage.save_experiment(record)
            job.set_progress(3 + training_span * epoch / epochs,
                             f"第 {epoch} 轮完成：训练 {train['accuracy']:.2%}，验证 {val_accuracy:.2%}，"
                             f"最佳验证 {best[0][0]:.2%}（第 {best[1]} 轮）")
            if epoch >= parameters["minimum_epochs"] and stale >= parameters["early_stopping_patience"]:
                break
        # 此前只用训练/验证数据更新参数及选模型；确定权重后才建立测试加载器。
        model.load_state_dict(torch.load(output / "best.pth", map_location="cpu", weights_only=True))
        comparison = None
        if augmentation:
            fixed_train = DataLoader(ScrambledTestSet(folders["train"], settings["preprocess"], iterations),
                                     batch_size=batch_size, shuffle=False, num_workers=0)
            train_logits, train_targets, train_loss = evaluate(model, fixed_train, lambda batch, total:
                job.set_progress(84 + 3 * batch / total, f"固定训练图评估增强模型：{batch}/{total} 批"))
            comparison = {"only_training_augmentation_changed": True,
                          "reference_experiment_id": reference["id"] if reference else None,
                          "evaluation_input": "same_fixed_unaugmented_scrambled_images",
                          "augmented": {"validation_accuracy": best[0][0],
                              "validation_loss": -best[0][1],
                              "fixed_train_accuracy": int((train_logits.argmax(1) == train_targets).sum()) / len(train_targets),
                              "fixed_train_loss": train_loss}}
            if reference:
                reference_path = Path(reference["model_bundle"]) / "model.pth"
                reference_model = models.resnet18(weights=None)
                reference_model.fc = nn.Linear(reference_model.fc.in_features, len(settings["classes"]))
                reference_model.load_state_dict(torch.load(reference_path, map_location="cpu", weights_only=True))
                reference_model = reference_model.to(memory_format=torch.channels_last)
                ref_logits, ref_targets, ref_loss = evaluate(reference_model, fixed_train, lambda batch, total:
                    job.set_progress(87 + 3 * batch / total, f"固定训练图评估原方案：{batch}/{total} 批"))
                ref_metrics = json.loads((reference_path.parent / "metrics.json").read_text(encoding="utf-8"))
                ref_history = json.loads((storage.experiment_root() / reference["id"] / "training_history.json").read_text(encoding="utf-8"))
                ref_best = next(r for r in ref_history if r["epoch"] == ref_metrics["best_epoch"])
                comparison["without_augmentation"] = {
                    "validation_accuracy": ref_metrics["best_validation_acc"], "validation_loss": ref_best["validation_loss"],
                    "test_accuracy": ref_metrics["test_accuracy"],
                    "fixed_train_accuracy": int((ref_logits.argmax(1) == ref_targets).sum()) / len(ref_targets),
                    "fixed_train_loss": ref_loss, "model_sha256": reference_digest}
                if sha256(reference_path) != reference_digest:
                    raise RuntimeError("原置乱模型文件在对照实验中发生变化")
                del reference_model, ref_logits, ref_targets
            for key in ("augmented", "without_augmentation"):
                if key in comparison:
                    c = comparison[key]
                    c["fixed_train_validation_gap"] = c["fixed_train_accuracy"] - c["validation_accuracy"]
            storage.write_json(root / "augmentation_comparison.json", comparison)
            del train_logits, train_targets
        folder = datasets.ImageFolder(str(raw / "test"))
        expected = {str(Path(r["path"])) for r in manifest if r["split"] == "test"}
        if folder.classes != settings["classes"] or {str(Path(p)) for p, _ in folder.samples} != expected:
            raise ValueError("测试集与原模型划分不一致")
        loader = DataLoader(ScrambledTestSet(folder, settings["preprocess"], iterations),
                            batch_size=batch_size, shuffle=False, num_workers=0)
        logits, labels, test_loss = evaluate(model, loader, lambda batch, total:
            job.set_progress(91 + 6 * batch / total, f"最终测试：{batch}/{total} 批，仅评估验证集选定的模型"))
        report = _test_report(logits, labels, folder.classes)
        promoted, default_reference = validation_promotion(best[0], parameters) if augmentation else (True, None)
        if comparison:
            comparison["augmented"]["test_accuracy"] = report["accuracy"]
            comparison["augmented"]["experiment_id"] = experiment_id
            comparison["promoted_by_validation"] = promoted
            comparison["default_model_reference"] = default_reference
            storage.write_json(root / "augmentation_comparison.json", comparison)
        report["macro_f1"] = sum(r["f1"] for r in report["per_class"].values()) / len(folder.classes)
        probabilities = logits.softmax(1)
        predictions = logits.argmax(1)
        with (root / "test_predictions.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["file", "true_class", "predicted_class", "confidence", "probability_cat", "probability_dog", "correct"])
            for index, (path, label) in enumerate(folder.samples):
                prediction = int(predictions[index])
                writer.writerow([path, folder.classes[label], folder.classes[prediction],
                                 float(probabilities[index, prediction]), *probabilities[index].tolist(), prediction == label])
        baselines = [e for e in storage.experiments() if e["status"] == "done"
                     and e["dataset_id"] == dataset["id"]
                     and e["parameters"].get("training_mode") == "frozen_model_scramble"
                     and e["parameters"].get("model_sha256") == source_digest]
        baseline = None
        for previous in baselines:
            path = Path(previous["results_dir"]) / f"{iterations:02d}" / "metrics.json"
            if path.exists():
                baseline = {"experiment_id": previous["id"], "accuracy": json.loads(path.read_text(encoding="utf-8"))["test_accuracy"]}
                break
        if sha256(source_model) != source_digest:
            raise RuntimeError("原图模型文件在微调过程中发生变化")
        metrics = {"arch": "resnet18", "classes": folder.classes, "training_mode": parameters["training_mode"],
                   "parameters": parameters, "best_test_acc": report["accuracy"], "test_accuracy": report["accuracy"],
                   "best_validation_acc": best[0][0], "best_validation_loss": -best[0][1],
                   "best_epoch": best[1], "completed_epochs": len(history),
                   "selection_split": "val", "test_evaluated_once": True, "test_loss": test_loss,
                   "test_report": report, "original_model_on_same_scrambled_test": baseline,
                   "augmentation_comparison": comparison,
                   "elapsed_seconds": time.time() - started}
        storage.write_json(output / "metrics.json", metrics)
        exported = config.MODEL_DIR / "scrambled_classifier" / experiment_id
        track_output(job, exported)
        exported.mkdir(parents=True)
        shutil.copy2(output / "best.pth", exported / "model.pth")
        storage.write_json(exported / "config.json", {**parameters, "classes": folder.classes,
                           "experiment_id": experiment_id, "algorithm": "arnold", "trained_iterations": iterations,
                           "input_contract": "RGB decoded 224x224 pixels, already scrambled; normalize only"})
        storage.write_json(exported / "metrics.json", metrics)
        storage.write_json(exported / "verification.json", {
            "source_model_unchanged": True, "source_model_sha256": source_digest,
            "exported_model_sha256": sha256(exported / "model.pth"), "checkpoint_sha256": sha256(output / "best.pth"),
            "test_samples": len(folder), "training_samples": len(folders["train"]), "validation_samples": len(folders["val"])})
        _plot(history, baseline, report, root / "training_summary.png", comparison)
        shutil.copy2(root / "training_summary.png", exported / "training_summary.png")
        description = (f"# Arnold {iterations} 次置乱图像分类器\n\n"
                       "在既有猫狗划分上，以已保存原图模型初始化，直接使用置乱训练图微调全部网络层；未做逆变换。\n\n"
                       f"最佳验证：{best[0][0]:.2%}（第 {best[1]} 轮）；最终测试：{report['accuracy']:.2%}，共 {len(folder)} 张。\n\n"
                       "输入：RGB，224×224、已做指定 Arnold 置乱的像素图；ToTensor 后使用 config.json 中的 mean/std。\n"
                       "输入已置乱图时不应再做一次置乱。若输入原图，由预测工具先执行指定次数置乱。\n\n"
                       "这是一种已知固定变换的监督适应实验；结果不代表对任意置乱次数或任意算法均有效。\n")
        if augmentation:
            description += ("\n训练时先轻微裁剪/翻转/颜色增强，再执行置乱；验证和测试保持固定。"
                            "增强训练准确率与无增强训练准确率不直接可比，过拟合差距另用固定训练图评估。\n")
        (exported / "README.md").write_text(description, encoding="utf-8")
        (root / "README.md").write_text(description, encoding="utf-8")
        new_pointer = {
            "bundle_dir": str(exported), "experiment_id": experiment_id, "algorithm": "arnold",
            "iterations": iterations, "accuracy": report["accuracy"]}
        begin_commit(job)
        if augmentation:
            storage.write_json(config.MODEL_DIR / "scrambled_classifier_augmented.json", new_pointer)
        if promoted:
            storage.write_json(config.MODEL_DIR / "scrambled_classifier.json", new_pointer)
        record.update(status="done", finished_at=time.time(), model_bundle=str(exported),
                      test_accuracy=report["accuracy"], original_model_unchanged=True,
                      augmentation_comparison=comparison, promoted_by_validation=promoted,
                      elapsed_seconds=time.time() - started)
        storage.save_experiment(record)
        storage.update_dataset(dataset["id"], active_experiment=experiment_id)
        job.set_progress(100, f"置乱模型已保存：测试 {report['accuracy']:.2%}")
        return {"experiment_id": experiment_id, "model_bundle": str(exported),
                "test_accuracy": report["accuracy"], "validation_accuracy": best[0][0],
                "best_epoch": best[1], "completed_epochs": len(history), "baseline": baseline,
                "augmentation_comparison": comparison, "promoted_by_validation": promoted}
    except Exception as exc:
        record.update(status="cancelled" if isinstance(exc, JobCancelled) else "error",
                      error=str(exc), finished_at=time.time())
        storage.save_experiment(record)
        raise


def _plot(history, baseline, report, path, comparison=None):
    os.environ.setdefault("MPLCONFIGDIR", str(config.ROOT / "tmp" / "matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    paired = comparison and "without_augmentation" in comparison
    fig, axes = plt.subplots(1, 3 if paired else 2, figsize=(16 if paired else 11, 4.5))
    epochs = [r["epoch"] for r in history]
    axes[0].plot(epochs, [100*r["train_accuracy"] for r in history], label="Training")
    axes[0].plot(epochs, [100*r["validation_accuracy"] for r in history], label="Validation", marker="o")
    axes[0].set(xlabel="Epoch", ylabel="Accuracy (%)", title="All-layer fine-tuning on scrambled images", ylim=(40, 101))
    axes[0].legend()
    if comparison and "without_augmentation" in comparison:
        old, new = comparison["without_augmentation"], comparison["augmented"]
        positions = np.arange(2)
        axes[1].bar(positions - .18, [100 * old["fixed_train_accuracy"], 100 * new["fixed_train_accuracy"]],
                    width=.36, label="Fixed training images")
        axes[1].bar(positions + .18, [100 * old["validation_accuracy"], 100 * new["validation_accuracy"]],
                    width=.36, label="Validation")
        axes[1].set(xticks=positions, xticklabels=["No augmentation", "Augmentation"],
                    ylabel="Accuracy (%)", ylim=(0, 105), title="Best checkpoints: same fixed inputs")
        axes[1].legend()
        values = [100 * old["test_accuracy"], 100 * report["accuracy"]]
        axes[2].bar(["No augmentation", "Augmentation"], values, color=["#9ca3af", "#2563eb"])
        for i, value in enumerate(values):
            axes[2].text(i, value + 1, f"{value:.2f}%", ha="center")
        axes[2].set(ylabel="Test accuracy (%)", title=f"Same {report['total']} test images", ylim=(0, 105))
        axes[0].set_title("Augmented training / fixed validation")
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        return
    if baseline:
        values = [100 * baseline["accuracy"], 100 * report["accuracy"]]
        axes[1].bar(["Original model", "Adapted model"], values, color=["#9ca3af", "#2563eb"])
        for i, value in enumerate(values):
            axes[1].text(i, value + 1, f"{value:.2f}%", ha="center")
    axes[1].set(ylabel="Test accuracy (%)", title=f"Same {report['total']} scrambled test images", ylim=(0, 105))
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
