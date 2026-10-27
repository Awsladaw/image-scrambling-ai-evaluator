"""批量置乱：把规范化后的数据集按多组参数梯度复制并置乱。

输入目录结构（由 prepare 生成）：
    data/raw/{train,test}/{cat,dog}/*.jpg

输出目录结构（每个参数梯度一份完整副本）：
    data/scrambled/arnold_02/{train,test}/{cat,dog}/*.jpg
    data/scrambled/arnold_04/...
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from src.scrambling.arnold import period, scramble  # noqa: E402


def _load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"))


def scramble_dataset(raw_dir=None, out_root=None, iterations=None,
                     algorithm="arnold", image_size=None, progress_callback=None, cancel_callback=None):
    """对 raw_dir 下的 train/test 数据集，按每个迭代参数生成置乱副本。

    progress_callback : callable(percent: float, message: str) | None
    返回 metadata 字典。
    """
    raw_dir = Path(raw_dir) if raw_dir is not None else config.RAW_DIR
    out_root = Path(out_root) if out_root is not None else config.SCRAMBLED_DIR
    iterations = iterations if iterations is not None else config.ARNOLD_ITERATIONS
    image_size = image_size or config.IMAGE_SIZE

    if algorithm != "arnold":
        raise NotImplementedError(f"尚未实现算法: {algorithm}")

    N = image_size
    T = period(N)
    print(f"[batch_scramble] {algorithm} 变换 | N={N} | 周期 T={T} | 迭代参数 {iterations}")

    # 统计源图片总数，用于进度计算
    src_files = []
    splits = ("train", "val", "test") if (raw_dir / "val").is_dir() else ("train", "test")
    for split in splits:
        src_split = raw_dir / split
        if src_split.is_dir():
            for cls_dir in src_split.iterdir():
                if cls_dir.is_dir():
                    src_files.extend(sorted(cls_dir.glob("*")))
    total_work = len(src_files) * len(iterations)
    done = 0

    meta = {
        "algorithm": algorithm,
        "image_size": N,
        "period": T,
        "iterations": list(iterations),
        "source": str(raw_dir),
    }

    for k in iterations:
        if k >= T:
            print(f"  [警告] 迭代 {k} >= 周期 {T}，等效为 {k % T} 次，"
                  f"已接近/越过还原点，建议使用 {T // 2} 以内的值。")
        dst_root = out_root / f"{algorithm}_{k:02d}"
        total = 0
        for split in splits:
            src_split = raw_dir / split
            if not src_split.is_dir():
                continue
            for cls_dir in sorted(p for p in src_split.iterdir() if p.is_dir()):
                dst_cls = dst_root / split / cls_dir.name
                dst_cls.mkdir(parents=True, exist_ok=True)
                for f in sorted(cls_dir.glob("*")):
                    if cancel_callback:
                        cancel_callback()
                    try:
                        arr = _load_rgb(f)
                        out = scramble(arr, k)
                        Image.fromarray(out).save(dst_cls / f.name)
                        total += 1
                    except Exception as exc:  # 跳过损坏图片
                        print(f"  [跳过] {f}: {exc}")
                    done += 1
                    if progress_callback and total_work:
                        progress_callback(done * 100.0 / total_work,
                                          f"迭代 {k} 置乱 {done}/{total_work}")
        print(f"  迭代 {k:>2}: 生成 {total} 张 -> {dst_root}")

    if cancel_callback:
        cancel_callback()
    meta_path = out_root / "metadata.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[batch_scramble] 元信息已写入 {meta_path}")
    return meta


if __name__ == "__main__":
    scramble_dataset()
