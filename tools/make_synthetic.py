"""生成合成二分类图像（水平条纹 vs 垂直条纹），用于离线验证整条管线。

类别 0：水平条纹；类别 1：垂直条纹。二者像素值分布相同、仅空间排列不同，
原始图可被 CNN 100% 区分；经 Arnold 置乱后空间信息被破坏，识别率应趋近 50%。

用法（在项目根目录下执行）：
    python tools/make_synthetic.py --per-class 40 --size 224 --out data/synthetic
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402


def make_image(cls_id, size=224, stripe=8, seed=None):
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size, 3), dtype=np.uint8)
    n_stripes = size // stripe
    vals = rng.integers(0, 256, size=n_stripes)
    if cls_id == 0:  # 水平条纹
        for i in range(size):
            img[i, :, :] = vals[i // stripe]
    else:            # 垂直条纹
        for j in range(size):
            img[:, j, :] = vals[j // stripe]
    noise = rng.integers(-6, 7, size=(size, size, 1))
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


def run(per_class=40, size=None, out_dir=None, seed=None):
    size = size or config.IMAGE_SIZE
    out_dir = Path(out_dir) if out_dir else config.DATA_ROOT / "synthetic"
    seed = seed if seed is not None else config.SEED
    rng = np.random.default_rng(seed)

    n_train = int(round(per_class * config.SPLIT_RATIO))
    split_counts = {"train": n_train, "test": per_class - n_train}
    idx = 0
    for cls_id, cls_name in ((0, "class0"), (1, "class1")):
        for split, cnt in split_counts.items():
            d = out_dir / split / cls_name
            d.mkdir(parents=True, exist_ok=True)
            for _ in range(cnt):
                img = make_image(cls_id, size=size, seed=int(rng.integers(2 ** 31)))
                Image.fromarray(img).save(d / f"{cls_name}_{idx:04d}.png")
                idx += 1
    print(f"[synthetic] 已生成合成二分类数据 -> {out_dir}")
    return out_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="生成合成二分类验证数据")
    ap.add_argument("--per-class", type=int, default=40)
    ap.add_argument("--size", type=int, default=config.IMAGE_SIZE)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    run(per_class=args.per_class, size=args.size, out_dir=args.out)
