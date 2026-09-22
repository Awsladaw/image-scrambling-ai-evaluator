"""业务编排层：把「数据导入 / 置乱 / AI评估 / 指标」四个模块组装成统一流程。

本层不依赖任何 Web 框架，仅调用 src/ 下的算法模块，可独立测试。
前端 / API 层只负责调用本层函数，从而实现「界面与算法解耦」。
"""

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from src.data import prepare  # noqa: E402
from src.scrambling import batch_scramble  # noqa: E402
from src import train  # noqa: E402

# 支持的置乱算法名 -> 内部标识
ALGORITHMS = {"arnold": "arnold"}


# ---------------- 基础状态查询 ----------------

def _scrambled_variants():
    """返回已生成的置乱变体 [(label, dir), ...]，label 为迭代次数字符串。"""
    variants = []
    if config.SCRAMBLED_DIR.exists():
        for d in sorted(config.SCRAMBLED_DIR.glob("arnold_*")):
            if d.is_dir():
                variants.append((d.name.replace("arnold_", ""), d))
    return variants


def get_scrambled_labels():
    return [label for label, _ in _scrambled_variants()]


def get_evaluated_labels():
    """返回已评估（存在 metrics.json）的标签列表。"""
    labels = []
    results_dir = config.RESULTS_DIR / "arnold"
    if results_dir.exists():
        for sub in sorted(results_dir.iterdir()):
            if (sub / "metrics.json").exists():
                labels.append(sub.name)
    return labels


# ---------------- 四大业务动作 ----------------

def reset_workspace():
    """清空当前工作区（原始/置乱/结果/模型），导入新数据集前调用。"""
    for p in (config.RAW_DIR, config.SCRAMBLED_DIR, config.RESULTS_DIR, config.MODEL_DIR):
        if p.exists():
            shutil.rmtree(p)


def import_dataset(zip_path, max_per_class=None, job=None):
    """动作①：导入数据集。job 需提供 set_progress(percent, message)。"""
    reset_workspace()

    def cb(p, m):
        if job:
            job.set_progress(p, m)

    raw = prepare.import_dataset(zip_path, max_per_class=max_per_class,
                                 progress_callback=cb, force=True)
    return {"raw": str(raw)}


def run_scramble(algorithm="arnold", iterations=None, job=None):
    """动作②：批量置乱。"""
    iterations = iterations or config.ARNOLD_ITERATIONS

    def cb(p, m):
        if job:
            job.set_progress(p, m)

    meta = batch_scramble.scramble_dataset(iterations=iterations, algorithm=algorithm,
                                           progress_callback=cb)
    return meta


def run_evaluate(arch=None, epochs=None, batch_size=None, lr=None,
                 optimizer=None, job=None, include_baseline=True):
    """动作③：AI 评估。训练「原始基线 + 各置乱变体」，汇总识别率。"""
    arch = arch or config.ARCH
    epochs = epochs if epochs is not None else config.EPOCHS
    batch_size = batch_size or config.BATCH_SIZE
    lr = lr or config.LEARNING_RATE
    optimizer = optimizer or config.OPTIMIZER

    algo = "arnold"
    units = []
    if include_baseline:
        units.append(("original", config.RAW_DIR))
    units += _scrambled_variants()

    if not units:
        raise RuntimeError("没有可评估的数据，请先导入数据集并执行置乱")

    total_epochs = len(units) * epochs
    done_epochs = [0]  # 用可变容器累计已完成的 epoch 数

    summary = []
    for label, data_dir in units:
        def cb(p, m):
            frac = (done_epochs[0] + p / 100.0) / total_epochs
            if job:
                job.set_progress(frac * 100.0, f"[{label}] {m}")

        out_dir = config.RESULTS_DIR / algo / label
        metrics = train.run(data_dir=data_dir, out_dir=out_dir, arch=arch,
                            epochs=epochs, batch_size=batch_size, lr=lr,
                            optimizer=optimizer, progress_callback=cb)
        done_epochs[0] += epochs
        summary.append({"label": label, "test_acc": metrics["best_test_acc"]})

    # 保存汇总
    _save_summary(summary)
    return {"summary": summary, "ranking": _ranking(summary)}


def _save_summary(summary):
    out = config.RESULTS_DIR / "arnold_summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        f.write("label,recognition_rate\n")
        for row in summary:
            f.write(f"{row['label']},{row['test_acc']:.4f}\n")


def _ranking(summary):
    """按「识别率距离 50% 越近越安全」排序（排除原始基线）。"""
    scrambled = [r for r in summary if r["label"] != "original"]
    return sorted(scrambled, key=lambda r: abs(r["test_acc"] - 0.5))


# ---------------- 动作④：指标查询 ----------------

def get_metrics():
    """读取已评估结果，返回曲线数据 + 安全性排名。"""
    algo = "arnold"
    results_dir = config.RESULTS_DIR / algo
    rows = []
    if results_dir.exists():
        for sub in sorted(results_dir.iterdir()):
            mf = sub / "metrics.json"
            if not mf.exists():
                continue
            m = json.loads(mf.read_text(encoding="utf-8"))
            label = sub.name
            try:
                iters = int(label)
            except ValueError:
                iters = None  # 原始基线 original
            rows.append({
                "label": label,
                "iterations": iters,
                "test_acc": m.get("best_test_acc"),
                "arch": m.get("arch"),
            })

    # 排序：原始基线（iterations=None）在前，其余按迭代次数升序
    rows.sort(key=lambda r: (r["iterations"] is not None,
                             r["iterations"] if r["iterations"] is not None else -1))

    scrambled = [r for r in rows if r["iterations"] is not None]
    ranking = sorted(scrambled, key=lambda r: abs(r["test_acc"] - 0.5))

    return {"rows": rows, "ranking": ranking, "baseline": 0.5,
            "algorithm": algo}
