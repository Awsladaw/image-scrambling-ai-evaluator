"""加载已训练模型，在测试集上独立复评识别率。

用法：
    python -m src.evaluate --data data/scrambled/arnold_06 --weights models/arnold_06/best.pth
"""

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from src.modeling import build_model, evaluate, get_dataloaders  # noqa: E402


def run(data_dir, weights, arch=None, device=None, image_size=None):
    arch = arch or config.ARCH
    device = device or config.DEVICE
    image_size = image_size or config.IMAGE_SIZE
    device = torch.device(device)

    model = build_model(num_classes=2, arch=arch).to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
    _, test_loader, classes = get_dataloaders(data_dir, image_size, config.BATCH_SIZE)
    acc = evaluate(model, test_loader, device)
    print(f"[evaluate] 数据={data_dir} 类别={classes} 测试识别率={acc:.4f}")
    return acc


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="模型复评")
    ap.add_argument("--data", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--arch", default=config.ARCH)
    ap.add_argument("--device", default=config.DEVICE)
    ap.add_argument("--image-size", type=int, default=config.IMAGE_SIZE)
    args = ap.parse_args()
    run(args.data, args.weights, args.arch, args.device, args.image_size)
