"""ResNet 二分类训练脚本：从零训练 + 测试集识别率评估。

用法（在项目根目录下执行）：
    python -m src.train --data data/raw --out models/arnold_02 \
        --arch resnet50 --epochs 10 --batch-size 16 --lr 1e-3
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from src.modeling import build_model, evaluate, get_dataloaders, get_validation_loader  # noqa: E402


def run(data_dir, out_dir, arch=None, epochs=None, batch_size=None, lr=None,
        optimizer=None, device=None, image_size=None, num_workers=None,
        progress_callback=None, cancel_callback=None):
    """训练并评估，返回 metrics 字典（含 best_test_acc 与历史曲线）。

    progress_callback : callable(percent: float, message: str) | None
    """
    arch = arch or config.ARCH
    epochs = epochs if epochs is not None else config.EPOCHS
    batch_size = batch_size or config.BATCH_SIZE
    lr = lr or config.LEARNING_RATE
    optimizer = optimizer or config.OPTIMIZER
    device = device or config.DEVICE
    image_size = image_size or config.IMAGE_SIZE
    num_workers = num_workers if num_workers is not None else config.NUM_WORKERS
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    started = time.time()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    print(f"[train] 数据: {data_dir} | 模型: {arch} | 设备: {device} | "
          f"epochs={epochs} batch={batch_size} lr={lr} opt={optimizer}")

    train_loader, test_loader, classes = get_dataloaders(
        data_dir, image_size, batch_size, num_workers)
    validation_loader = get_validation_loader(data_dir, image_size, batch_size, num_workers)
    selection_loader = validation_loader if validation_loader is not None else test_loader
    selection_split = "val" if validation_loader is not None else "test"
    print(f"[train] 类别: {classes} | 训练批数={len(train_loader)} 测试批数={len(test_loader)}")

    model = build_model(num_classes=2, arch=arch).to(device)
    criterion = nn.CrossEntropyLoss()
    if optimizer == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError(f"未知优化器: {optimizer}")

    history = {"train_acc": [], "train_loss": [], "test_acc": []}
    if validation_loader is not None:
        history["val_acc"] = []
    best_selection_acc, best_epoch = -1.0, None

    for epoch in range(1, epochs + 1):
        if cancel_callback:
            cancel_callback()
        model.train()
        run_loss = 0.0
        correct = total = 0
        last_update = 0.0
        for batch_index, (x, y) in enumerate(train_loader):
            if cancel_callback:
                cancel_callback()
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()
            run_loss += loss.item() * y.size(0)
            correct += (logits.detach().argmax(1) == y).sum().item()
            total += y.size(0)
            now = time.monotonic()
            if progress_callback and now - last_update >= 1:
                progress_callback((epoch - 1 + (batch_index + 1) / len(train_loader)) / epochs * 100,
                                  f"epoch {epoch}/{epochs} batch {batch_index + 1}/{len(train_loader)}")
                last_update = now

        train_acc = correct / total
        selection_acc = evaluate(model, selection_loader, device, cancel_callback=cancel_callback)
        history["train_acc"].append(round(train_acc, 4))
        history["train_loss"].append(round(run_loss / total, 4))
        history[selection_split + "_acc"].append(round(selection_acc, 4))
        print(f"  epoch {epoch:>2}/{epochs}  train_acc={train_acc:.4f}  {selection_split}_acc={selection_acc:.4f}", flush=True)

        if selection_acc > best_selection_acc:
            best_selection_acc, best_epoch = selection_acc, epoch
            torch.save(model.state_dict(), out_dir / "best.pth")
        (out_dir / "history.json").write_text(json.dumps({"epoch": epoch, "epochs": epochs,
            "history": history, "selection_split": selection_split,
            "best_epoch": best_epoch}, indent=2), encoding="utf-8")

        if progress_callback:
            progress_callback(epoch * 100.0 / epochs,
                              f"epoch {epoch}/{epochs} train_acc={train_acc:.4f} "
                              f"{selection_split}_acc={selection_acc:.4f}")

    if validation_loader is not None:
        model.load_state_dict(torch.load(out_dir / "best.pth", map_location=device, weights_only=True))
        best_test_acc = evaluate(model, test_loader, device, cancel_callback=cancel_callback)
        history["test_acc"] = [round(best_test_acc, 4)]
    else:
        best_test_acc = best_selection_acc

    if cancel_callback:
        cancel_callback()
    metrics = {
        "arch": arch,
        "data": str(data_dir),
        "classes": classes,
        "epochs": epochs,
        "best_test_acc": best_test_acc,
        "selection_split": selection_split,
        "best_epoch": best_epoch,
        "best_validation_acc": best_selection_acc if validation_loader is not None else None,
        "test_evaluated_once": validation_loader is not None,
        "seed": config.SEED, "image_size": image_size,
        "batch_size": batch_size, "lr": lr, "optimizer": optimizer,
        "elapsed_seconds": time.time() - started,
        "history": history,
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    result_label = "独立测试识别率" if validation_loader is not None else "最佳测试识别率"
    print(f"[train] 完成 | {result_label} = {best_test_acc:.4f} | 已保存 {out_dir / 'metrics.json'}", flush=True)
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ResNet 二分类训练与评估")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arch", default=config.ARCH)
    ap.add_argument("--epochs", type=int, default=config.EPOCHS)
    ap.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    ap.add_argument("--optimizer", default=config.OPTIMIZER)
    ap.add_argument("--device", default=config.DEVICE)
    ap.add_argument("--image-size", type=int, default=config.IMAGE_SIZE)
    args = ap.parse_args()
    run(data_dir=args.data, out_dir=args.out, arch=args.arch, epochs=args.epochs,
        batch_size=args.batch_size, lr=args.lr, optimizer=args.optimizer,
        device=args.device, image_size=args.image_size)
