"""二分类猫狗数据集：下载 / 本地导入、规范化、8:2 划分。

对外提供两个入口：
  * import_dataset(source, ...)  —— 从本地 zip 或目录导入（供 Web 系统调用，带进度回调）
  * run(...)                     —— 命令行入口（支持自动下载）

流程：定位 Cat/Dog 目录 -> 过滤损坏图片 -> 每类可选子采样 -> 8:2 划分 -> 缩放保存。
"""

import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _download(url, dest):
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[prepare] 已存在 {dest}，跳过下载")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[prepare] 下载数据集（约 800MB）...\n  {url}\n  -> {dest}")

    def _hook(count, block_size, total_size):
        if total_size > 0:
            pct = int(count * block_size * 100 / total_size)
            if pct % 20 == 0:
                print(f"    {pct}%")

    urllib.request.urlretrieve(url, dest, reporthook=_hook)
    print("[prepare] 下载完成")
    return dest


def _extract(zip_path, dest_dir, cancel_callback=None):
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"[prepare] 解压到 {dest_dir} ...")
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            if cancel_callback:
                cancel_callback()
            zf.extract(member, dest_dir)


def _find_class_dirs(root, cancel_callback=None):
    """在目录中定位 Cat/Dog 图片目录（大小写不敏感）。"""
    cat = dog = None
    for d in Path(root).rglob("*"):
        if cancel_callback:
            cancel_callback()
        if d.is_dir():
            if d.name.lower() == "cat":
                cat = d
            elif d.name.lower() == "dog":
                dog = d
    if cat is None or dog is None:
        raise FileNotFoundError(f"在 {root} 中未找到 cat/dog 图片目录")
    return cat, dog


def _list_images(d):
    return sorted(p for p in Path(d).iterdir()
                  if p.is_file() and p.suffix.lower() in _IMAGE_EXTS)


def _subsample(files, max_per_class, seed):
    if not max_per_class or len(files) <= max_per_class:
        return files
    rng = np.random.default_rng(seed)
    idx = sorted(rng.choice(len(files), size=max_per_class, replace=False))
    return [files[i] for i in idx]


def _resize_save(src, dst, size):
    """解码并缩放到 size×size 保存，损坏图片返回 False。"""
    try:
        with Image.open(src) as im:
            im = im.convert("RGB")
            im.load()  # 触发解码，暴露截断/损坏文件
            im = im.resize((size, size), Image.BILINEAR)
            im.save(dst)
        return True
    except Exception:
        return False


def import_dataset(source, data_root=None, max_per_class=None, image_size=None,
                   split_ratio=None, seed=None, progress_callback=None, force=False,
                   validation_ratio=0.0, cancel_callback=None):
    """从本地 zip 或目录导入猫狗数据集，返回 raw_dir 路径。

    source : str | Path —— zip 文件路径，或含 Cat/Dog 子目录的文件夹
    progress_callback : callable(percent: float, message: str) | None
    """
    data_root = Path(data_root) if data_root is not None else config.DATA_ROOT
    max_per_class = max_per_class if max_per_class is not None else config.MAX_PER_CLASS
    image_size = image_size or config.IMAGE_SIZE
    split_ratio = split_ratio if split_ratio is not None else config.SPLIT_RATIO
    seed = seed if seed is not None else config.SEED
    if not 0 <= validation_ratio < split_ratio < 1:
        raise ValueError("验证集比例应小于训练及验证集总比例，且测试集比例大于零")

    raw_dir = data_root / "raw"
    if force and raw_dir.exists():
        shutil.rmtree(raw_dir)

    source = Path(source)
    if source.is_dir():
        extract_dir = source
    else:
        extract_dir = data_root / "extracted"
        if force and extract_dir.exists():
            shutil.rmtree(extract_dir)
        _extract(source, extract_dir, cancel_callback)

    cat_dir, dog_dir = _find_class_dirs(extract_dir, cancel_callback)
    cats = _subsample(_list_images(cat_dir), max_per_class, seed)
    dogs = _subsample(_list_images(dog_dir), max_per_class, seed)
    print(f"[prepare] 有效图片: Cat={len(cats)}, Dog={len(dogs)}"
          + (f"（每类子采样 {max_per_class}）" if max_per_class else "（全量）"))

    if not cats or not dogs:
        raise RuntimeError("没有可用图片，请检查数据源")

    rng = np.random.default_rng(seed)
    total = len(cats) + len(dogs)
    done = 0
    saved = {"train": 0, "test": 0}
    if validation_ratio:
        saved["val"] = 0
    for cls_name, files in (("cat", cats), ("dog", dogs)):
        order = rng.permutation(len(files))
        n_development = int(round(len(files) * split_ratio))
        n_val = int(round(len(files) * validation_ratio))
        n_train = n_development - n_val
        split_ids = {"train": order[:n_train], "test": order[n_development:]}
        if validation_ratio:
            split_ids["val"] = order[n_train:n_development]
        for split, ids in split_ids.items():
            out_dir = raw_dir / split / cls_name
            out_dir.mkdir(parents=True, exist_ok=True)
            for i in ids:
                if cancel_callback:
                    cancel_callback()
                dst = out_dir / f"{cls_name}_{int(i):05d}.jpg"
                if _resize_save(files[int(i)], dst, image_size):
                    saved[split] += 1
                done += 1
                if progress_callback:
                    progress_callback(done * 100.0 / total,
                                      f"处理 {done}/{total} 张图片")

    print(f"[prepare] 完成 -> {raw_dir}（train={saved['train']}, test={saved['test']}）")
    return raw_dir


def run(data_root=None, max_per_class=None, image_size=None, split_ratio=None,
        seed=None, download=True, source=None, force=False):
    """命令行入口：可选下载后导入。"""
    data_root = Path(data_root) if data_root is not None else config.DATA_ROOT

    if source:
        return import_dataset(source, data_root=data_root, max_per_class=max_per_class,
                              image_size=image_size, split_ratio=split_ratio,
                              seed=seed, force=force)

    raw_dir = data_root / "raw"
    if any(raw_dir.rglob("*.jpg")) or any(raw_dir.rglob("*.png")):
        if not force:
            print(f"[prepare] {raw_dir} 已存在，跳过。如需重新生成请加 --force")
            return raw_dir

    zip_path = data_root / "cache" / "catsdogs.zip"
    if download:
        _download(config.DATASET_URL, zip_path)
    return import_dataset(zip_path, data_root=data_root, max_per_class=max_per_class,
                          image_size=image_size, split_ratio=split_ratio,
                          seed=seed, force=force)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="猫狗二分类数据集下载与规范化")
    ap.add_argument("--max-per-class", type=int, default=None, help="每类图片数上限")
    ap.add_argument("--image-size", type=int, default=config.IMAGE_SIZE)
    ap.add_argument("--split-ratio", type=float, default=config.SPLIT_RATIO)
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--source", default=None, help="本地 zip 或目录（跳过下载）")
    ap.add_argument("--no-download", action="store_true", help="不下载，仅用已有缓存")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    run(max_per_class=args.max_per_class, image_size=args.image_size,
        split_ratio=args.split_ratio, seed=args.seed, source=args.source,
        download=not args.no_download, force=args.force)
