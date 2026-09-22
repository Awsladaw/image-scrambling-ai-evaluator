# 图像置乱算法 AI 安全评价系统

> **Image Scrambling AI Evaluator** — Deep-learning-based security evaluation of image scrambling algorithms: train ResNet on scrambled images and use recognition rate (closer to 50% = more secure) as the quantitative security metric.

以「大模型识别率是否接近 50%」为核心判据，对图像置乱算法进行量化安全性评价：先对二分类图像数据集进行置乱，再训练 ResNet 对置乱图像分类；识别率越接近 50%（随机猜测水平），说明置乱越彻底、安全性越高。

<!-- 替换为你的网页界面截图：![系统界面](docs/screenshot.png) -->

## 核心思想

二分类场景下，随机猜测的准确率为 **50%**。因此：

| 大模型识别率 | 含义 |
|---|---|
| ≈ 50% | 模型无法区分置乱图像的原始类别 → 置乱成功掩盖类别信息 → **安全性高** |
| 明显 > 50% | 置乱图仍残留可被模型提取的类别特征 → **安全性低** |

把大模型（ResNet）当作「唯密文攻击者」：只给它看置乱图像（密文），若它猜不出原始类别，说明该置乱算法有效抵抗了模式识别攻击。

## 技术栈

| 环节 | 技术 |
|---|---|
| 置乱算法 | Python + **numpy / PIL**（Arnold 变换等） |
| 深度学习 | **PyTorch** + **ResNet**（ResNet-18 / ResNet-50） |
| Web 系统 | **FastAPI** + 原生前端（HTML/JS/CSS，无第三方 CDN） |

## 系统架构（三层解耦）

```
① src/        领域算法层（无 Web 依赖）   ← 置乱 / 训练 / 评估 / 可视化
② services/   编排层（无 Web 依赖）       ← 组装「导入/置乱/评估/指标」
③ app/        表现层                      ← FastAPI + 前端，只做转发
```

新增置乱算法只需改 `src/scrambling/`，复用其余模块，重复「置乱 → 训练 → 记录识别率」三步即可。

## 功能特性

- **数据导入**：上传猫狗 zip → 自动规范化 224×224 → 8:2 划分
- **图像置乱**：多参数梯度批量置乱（带进度条）
- **AI 评估**：ResNet 从零训练 + 识别率评估（带进度条）
- **指标显示**：「迭代次数—识别率」曲线 + 安全性排名表

## 快速开始

### 环境

已在 Python 3.11 + torch 2.12.0+cpu + torchvision 0.27.0+cpu 环境验证。Windows 下统一使用 `py -3.11`（本机 `python` 指向商店占位符，不可用）。

### 1. 安装依赖

```powershell
py -3.11 -m pip install -r requirements.txt
```

### 2. 启动 Web 系统

```powershell
cd dachuang
py -3.11 run_web.py
```

浏览器打开 **http://127.0.0.1:8000**，依次操作：

1. **数据导入**：上传猫狗数据集 zip（内含 `Cat/`、`Dog/` 文件夹）；
2. **图像置乱**：勾选参数梯度 → 点「开始置乱」；
3. **AI 评估**：选模型/轮数 → 点「开始 AI 评估」；
4. **指标显示**：自动出曲线 + 安全性排名。

### 3. 命令行方式（可选）

```powershell
# 一键跑完整流程（下载数据→置乱→训练→出曲线）
py -3.11 run_step1.py --max-per-class 60 --arch resnet18 --epochs 3

# 分步执行
py -3.11 -m src.data.prepare --max-per-class 60          # 下载/规范化数据
py -3.11 -m src.scrambling.batch_scramble                # 批量置乱
py -3.11 -m src.train --data data/scrambled/arnold_06 --out models/arnold_06  # 训练
py -3.11 -m src.visualize --algo arnold                  # 汇总可视化
```

## 实验结果示例

本地合成二分类数据（水平条纹 vs 垂直条纹）验证：

| 数据 | 识别率 |
|---|---|
| 原始图（未置乱） | 100% |
| Arnold 2 次置乱 | 66.7% |
| Arnold 6 次置乱 | 50%（随机猜测水平） |

识别率随置乱强度下降并趋近 50%，符合「越置乱越安全」的预期。

## 关键技术发现

**Arnold 变换周期 T(224) = 24**：迭代满 24 次图像完全还原，越过半周期（12）后开始反向还原。因此「迭代次数越多越安全」对 Arnold 不成立，参数梯度取 `[2, 4, 6, 8, 10, 12]`（半周期单调区间）。

## Roadmap

- [x] Arnold 变换置乱与评估
- [ ] Logistic 混沌置乱
- [ ] Lorenz 混沌置乱
- [ ] 元胞自动机置乱
- [ ] ViT 模型对比验证
- [ ] 多数据集泛化验证

## License

[MIT](./LICENSE)
