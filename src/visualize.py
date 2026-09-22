"""识别率结果汇总与可视化。

读取 results/<algo>/*/metrics.json 中每个参数梯度的最佳测试识别率，
汇总为 CSV，并在 matplotlib 可用时绘制「迭代次数—识别率」曲线。

用法：
    python -m src.visualize --algo arnold
"""

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402


def collect(results_dir, algo):
    """返回 [(iterations, best_test_acc), ...]，iterations=0 表示原始未置乱。"""
    rows = []
    base = Path(results_dir) / algo
    if not base.is_dir():
        print(f"[visualize] 未找到 {base}")
        return rows
    for sub in sorted(base.iterdir()):
        if not sub.is_dir():
            continue
        mf = sub / "metrics.json"
        if not mf.exists():
            continue
        metrics = json.loads(mf.read_text(encoding="utf-8"))
        label = sub.name
        try:
            iters = int(label)
        except ValueError:
            iters = label
        rows.append((iters, metrics.get("best_test_acc", float("nan"))))
    # 按迭代次数排序（0 原始图排最前）
    rows.sort(key=lambda r: (r[0] if isinstance(r[0], int) else -1))
    return rows


def save_csv(rows, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["iterations", "recognition_rate"])
        for iters, acc in rows:
            w.writerow([iters, f"{acc:.4f}"])
    print(f"[visualize] 已写 CSV: {out_path}")


def plot(rows, out_path, algo, random_baseline=0.5):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [str(r[0]) for r in rows]
    ys = [r[1] for r in rows]
    plt.figure(figsize=(8, 5))
    plt.plot(xs, ys, marker="o", linewidth=2)
    plt.axhline(random_baseline, color="red", linestyle="--",
                label="随机猜测基线 50%")
    plt.xlabel("Arnold 迭代次数（0=原始图）")
    plt.ylabel("ResNet 识别率")
    plt.ylim(0, 1.05)
    plt.title(f"{algo} 置乱参数 — 大模型识别率")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"[visualize] 已绘曲线: {out_path}")


def run(results_dir=None, algo="arnold"):
    results_dir = Path(results_dir) if results_dir else config.RESULTS_DIR
    rows = collect(results_dir, algo)
    if not rows:
        print("[visualize] 无结果可汇总")
        return
    print(f"[visualize] {algo} 识别率汇总:")
    for iters, acc in rows:
        print(f"  迭代 {iters}: {acc:.4f}")

    save_csv(rows, results_dir / f"{algo}_summary.csv")
    try:
        plot(rows, results_dir / f"{algo}_curve.png", algo)
    except ImportError:
        print("[visualize] 未安装 matplotlib，跳过绘图（已生成 CSV）")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="识别率汇总可视化")
    ap.add_argument("--algo", default="arnold")
    args = ap.parse_args()
    run(algo=args.algo)
