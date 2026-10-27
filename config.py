"""项目全局配置 —— 图像置乱算法 AI 安全评价平台。

所有脚本/模块从本文件读取默认参数；也可通过命令行参数覆盖。
"""

from pathlib import Path

# ---------------- 路径 ----------------
ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data"                 # 数据根目录
RAW_DIR = DATA_ROOT / "raw"               # 规范化后的原始猫狗数据（train/test 划分后）
SCRAMBLED_DIR = DATA_ROOT / "scrambled"   # 各算法/参数梯度置乱数据
RESULTS_DIR = ROOT / "results"            # 识别率结果与曲线
MODEL_DIR = ROOT / "models"               # 训练好的模型权重
UPLOAD_DIR = DATA_ROOT / "uploads"        # Web 上传文件暂存目录

# ---------------- 数据集 ----------------
IMAGE_SIZE = 224          # ResNet 标准输入尺寸，也是 Arnold 变换方阵边长（必须为方形）
SPLIT_RATIO = 0.8         # 训练/测试划分比例 8:2
MAX_PER_CLASS = None      # 每类图片数上限；None=全量；小样本跑通可设为 60
SEED = 42                 # 随机种子（子采样与划分可复现）

# 猫狗数据集免登录下载源（微软官方原始链接，即 Kaggle cats-vs-dogs 的出处）
DATASET_URL = (
    "https://download.microsoft.com/download/3/E/1/"
    "3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip"
)

# ---------------- Arnold 置乱参数 ----------------
# 重要：Arnold 映射在 N×N 上存在周期 T(N)，T(224)=24。
# 迭代满 24 次图像会完全还原。选取半周期内的参数作比较，
# 识别难度不保证随迭代次数单调变化，应以实测结果为准。
ARNOLD_ITERATIONS = [2, 4, 6, 8, 10, 12]

# ---------------- 训练 ----------------
ARCH = "resnet50"         # 代表性大模型：resnet50 / resnet18（快速验证可用 resnet18）
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 1e-3
OPTIMIZER = "adam"        # adam / sgd
NUM_WORKERS = 0           # Windows 下 DataLoader 多进程必须为 0
DEVICE = "cpu"            # 或 "cuda"

# ---------------- Web 系统 ----------------
WEB_HOST = "127.0.0.1"
WEB_PORT = 8000
