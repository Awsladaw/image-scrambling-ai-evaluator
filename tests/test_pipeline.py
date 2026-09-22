"""services 层端到端自检（不依赖 FastAPI）。

流程：生成临时 Cat/Dog zip -> 导入 -> 置乱 -> AI评估 -> 读取指标。
用于验证「数据导入 / 置乱 / 评估 / 指标」四大模块解耦后能正确串起来。
"""

import json
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from services import pipeline  # noqa: E402


class DummyJob:
    """模拟 Web 层的任务对象，只需 set_progress 接口。"""
    def set_progress(self, p, m):
        print(f"    [{p:5.1f}%] {m}")


def make_stripes(cls_id, size=224, stripe=8, seed=None):
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size, 3), dtype=np.uint8)
    n = size // stripe
    vals = rng.integers(0, 256, size=n)
    if cls_id == 0:      # 水平条纹
        for i in range(size):
            img[i, :, :] = vals[i // stripe]
    else:                # 垂直条纹
        for j in range(size):
            img[:, j, :] = vals[j // stripe]
    return img


def build_zip(n=30):
    src = config.DATA_ROOT / "test_upload"
    for cls_name, cls_id in (("Cat", 0), ("Dog", 1)):
        d = src / cls_name
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            Image.fromarray(make_stripes(cls_id, seed=i)).save(d / f"{cls_name}_{i:03d}.jpg")
    zpath = config.DATA_ROOT / "test_upload.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        for p in src.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(src))
    return zpath


def main():
    zpath = build_zip()
    print("== ① 数据导入 ==")
    pipeline.import_dataset(zpath, max_per_class=30, job=DummyJob())

    print("== ② 图像置乱 ==")
    pipeline.run_scramble(iterations=[2, 6], job=DummyJob())

    print("== ③ AI 评估 ==")
    pipeline.run_evaluate(arch="resnet18", epochs=2, job=DummyJob())

    print("== ④ 指标读取 ==")
    print(json.dumps(pipeline.get_metrics(), ensure_ascii=False, indent=2))

    print("services 层自检完成 (OK)")


if __name__ == "__main__":
    main()
