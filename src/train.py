"""ResNet 二分类训练脚本：从零训练 + 测试集识别率评估。

用法（在项目根目录下执行）：
    python -m src.train --data data/raw --out models/arnold_02 \
        --arch resnet50 --epochs 10 --batch-size 16 --lr 1e-3
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from src.modeling import build_model, evaluate, get_dataloaders  # noqa: E402


def run(data_dir, out_dir, arch=None, epochs=None, batch_size=None, lr=None,
        optimizer=None, device=None, image_size=None, num_workers=None,
        progress_callback=None):
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

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    print(f"[train] 数据: {data_dir} | 模型: {arch} | 设备: {device} | "
          f"epochs={epochs} batch={batch_size} lr={lr} opt={optimizer}")

    train_loader, test_loader, classes = get_dataloaders(
        data_dir, image_size, batch_size, num_workers)
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
    best_test_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        run_loss = 0.0
        correct = total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()
            run_loss += loss.item() * y.size(0)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)

        train_acc = correct / total
        test_acc = evaluate(model, test_loader, device)
        history["train_acc"].append(round(train_acc, 4))
        history["train_loss"].append(round(run_loss / total, 4))
        history["test_acc"].append(round(test_acc, 4))
        print(f"  epoch {epoch:>2}/{epochs}  train_acc={train_acc:.4f}  test_acc={test_acc:.4f}")

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), out_dir / "best.pth")

        if progress_callback:
            progress_callback(epoch * 100.0 / epochs,
                              f"epoch {epoch}/{epochs} train_acc={train_acc:.4f} "
                              f"test_acc={test_acc:.4f}")

    metrics = {
        "arch": arch,
        "data": str(data_dir),
        "classes": classes,
        "epochs": epochs,
        "best_test_acc": best_test_acc,
        "history": history,
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[train] 完成 | 最佳测试识别率 = {best_test_acc:.4f} | 已保存 {out_dir / 'metrics.json'}")
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
