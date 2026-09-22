"""步骤一全流程驱动：Arnold 变换置乱算法安全性评估（一键）。

流程：数据准备 -> 原始图基线训练 -> 各参数批量置乱 -> 逐参数训练 -> 汇总可视化。

用法（在项目根目录下执行）：
    python run_step1.py --max-per-class 60 --arch resnet18 --epochs 3
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from src.data import prepare  # noqa: E402
from src.scrambling import batch_scramble  # noqa: E402
from src import train, visualize  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="步骤一：Arnold 变换安全性评估")
    ap.add_argument("--max-per-class", type=int, default=config.MAX_PER_CLASS,
                    help="每类图片数上限（小样本跑通建议 60）")
    ap.add_argument("--arch", default=config.ARCH, help="resnet50 / resnet18")
    ap.add_argument("--epochs", type=int, default=config.EPOCHS)
    ap.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    ap.add_argument("--optimizer", default=config.OPTIMIZER)
    ap.add_argument("--device", default=config.DEVICE)
    ap.add_argument("--iterations", default=None,
                    help="逗号分隔的迭代列表，默认取 config.ARNOLD_ITERATIONS")
    ap.add_argument("--skip-download", action="store_true")
    args = ap.parse_args()

    iterations = (config.ARNOLD_ITERATIONS if not args.iterations
                  else [int(x) for x in args.iterations.split(",")])

    # 1. 数据准备
    prepare.run(max_per_class=args.max_per_class, seed=config.SEED,
                download=not args.skip_download)

    # 2. 批量置乱
    batch_scramble.scramble_dataset(iterations=iterations)

    algo = "arnold"
    results_root = config.RESULTS_DIR / algo

    # 3. 原始图基线（迭代=0）
    print("\n=== 基线：原始未置乱图像 ===")
    train.run(data_dir=config.RAW_DIR, out_dir=results_root / "original",
              arch=args.arch, epochs=args.epochs, batch_size=args.batch_size,
              lr=args.lr, optimizer=args.optimizer, device=args.device)

    # 4. 各参数梯度逐一训练
    for k in iterations:
        data_dir = config.SCRAMBLED_DIR / f"{algo}_{k:02d}"
        print(f"\n=== Arnold 迭代 {k} 次置乱 ===")
        train.run(data_dir=data_dir, out_dir=results_root / f"{k:02d}",
                  arch=args.arch, epochs=args.epochs, batch_size=args.batch_size,
                  lr=args.lr, optimizer=args.optimizer, device=args.device)

    # 5. 汇总 + 可视化
    print("\n=== 汇总 ===")
    visualize.run(algo=algo)


if __name__ == "__main__":
    main()
