# 图像置乱算法 AI 安全评价系统

> **Image Scrambling AI Evaluator** — Evaluate how learnable image categories remain after scrambling through controlled ResNet training and saved-model inference.

对二分类图像数据集进行置乱，通过原图模型、固定模型测试和置乱后微调等对照实验，分析类别信息在特定模型及训练条件下的可识别程度。系统结合准确率、宏平均精确率、召回率、F1 和混淆矩阵评估结果。

<!-- 替换为你的网页界面截图：![系统界面](docs/screenshot.png) -->

## 核心思想

在类别均衡的二分类测试集上，均匀随机猜测的期望准确率为 **50%**：

| 识别结果 | 含义 |
|---|---|
| ≈ 50% | 需结合混淆矩阵检查类别偏向，并通过适应性训练判断该模型能否学习置乱后的类别信息 |
| 明显 > 50% | 在当前实验条件下，模型仍可利用置乱图中的类别信息 |

将 ResNet 用作特定的模式识别评估模型。固定原图模型在置乱图上表现下降可能来自输入分布变化；50% 的准确率也可能来自几乎只预测一个类别。识别率下降不能单独证明密码安全，仍需考虑适应性训练、其他模型及逆变换等条件。

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
- **已保存模型测试**：在 AI 评估中选择已有模型和测试数据集，对原图、指定 Arnold 次数或二者对照进行只读推理；支持临时无损置乱和读取已保存的置乱图片。自动沿用模型输入尺寸及标准化参数，检查测试图与已知训练／验证图的 SHA256 重合，分别保存准确率、宏平均精确率／召回率／F1、混淆矩阵和逐图预测，并提供 JSON／CSV 下载。原始测试集曾用于选模型的旧模型会明确提示，完整重复检查不等于近重复检查。
- **指标显示**：「迭代次数—识别率」曲线、各项分类指标及混淆矩阵
- **任务互斥**：导入、置乱、训练和数据集切换串行执行，上传期间也禁止冲突操作；刷新页面自动恢复进度。
- **停止当前任务**：顶部按钮可停止上传、导入、置乱及训练。等待当前图片或批次结束后，自动清理该任务新建的上传文件、数据版本、模型及实验输出；此前完成的数据和模型保留。清理期间禁止启动新任务，完成后显示清理结果。任务列表保留停止记录，但被清理的实验不再出现在结果列表中。最终发布结果期间按钮暂不可用，已停止的任务不会自动续跑；此规则不追溯删除改动前已停止任务的文件。
- **数据集版本**：每次导入生成独立版本，成功后切换为当前数据集；失败时保留此前数据集。
- **历史实验**：置乱和训练均使用独立目录，支持切换数据集、查看历史实验及对应参数，不覆盖旧权重。

数据集保存在 `data/datasets/<版本>/`，训练记录保存在 `results/experiments/<实验编号>/`。
已有 `data/raw`、`data/scrambled` 和 `results/arnold` 会在首次启动时登记，原文件保留。
任务状态保存在 `data/jobs.json`；服务重启后，未完成任务会标为中断，可查看已保存的部分实验结果，但不会自动续训。
Web 服务使用单进程运行；执行后台任务时不要同时用命令行修改相同数据。

## 快速开始

### 环境

当前本地验证环境为 Python 3.12 + torch 2.14.1+cpu + torchvision 0.29.1+cpu。以下 Windows 命令使用 Python 3.12 创建独立虚拟环境；当前训练服务主要使用 CPU，尚未完成 CUDA 适配。

### 1. 安装依赖

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 2. 启动 Web 系统

```powershell
cd dachuang
.\.venv\Scripts\python.exe run_web.py
```

本项目已配置的虚拟环境可直接启动：

```powershell
.\.venv\Scripts\python.exe run_web.py
```

浏览器打开 **http://127.0.0.1:8000**，依次操作：

1. **数据导入**：上传猫狗数据集 zip（内含 `Cat/`、`Dog/` 文件夹）；
2. **图像置乱**：勾选参数梯度 → 点「开始置乱」；
3. **AI 评估**：选择「使用已保存模型测试」或「重新训练并评估」，配置模型、数据集及测试条件后开始；新安装尚无模型时先训练；
4. **指标显示**：查看曲线、分类指标和混淆矩阵；已保存模型测试可下载 JSON 汇总和 CSV 逐图预测。

### 3. 命令行方式（可选）

```powershell
# 一键跑完整流程（下载数据→置乱→训练→出曲线）
.\.venv\Scripts\python.exe run_step1.py --max-per-class 60 --arch resnet18 --epochs 3

# 分步执行
.\.venv\Scripts\python.exe -m src.data.prepare --max-per-class 60          # 下载/规范化数据
.\.venv\Scripts\python.exe -m src.scrambling.batch_scramble                # 批量置乱
.\.venv\Scripts\python.exe -m src.train --data data/scrambled/arnold_06 --out models/arnold_06  # 训练
.\.venv\Scripts\python.exe -m src.visualize --algo arnold                  # 汇总可视化
```

## 实验结果示例

本地合成二分类数据（水平条纹 vs 垂直条纹）验证：

| 数据 | 识别率 |
|---|---|
| 原始图（未置乱） | 100% |
| Arnold 2 次置乱 | 66.7% |
| Arnold 6 次置乱 | 50%（随机猜测水平） |

以上是早期小样本运行示例，不能用于判断密码安全；正式结论应参考对应实验的样本量及评估方式。

## 关键技术发现

**Arnold 变换周期 T(224) = 24**：迭代满 24 次图像完全还原。参数梯度取 `[2, 4, 6, 8, 10, 12]` 作比较，识别难度不保证随次数单调变化。

## 高识别率原图模型

原图模型训练服务使用 ImageNet 预训练 ResNet-18 固定特征，
在独立的 6000 张训练图、1000 张验证图上比较分类头正则参数，最后评估 2000 张测试图。
这是迁移学习实验，与从零训练的置乱评估分开记录；此前实验使用过的图片会从候选数据中排除。
完整划分清单、调参结果及测试预测会保存到对应实验目录。

模型导出到 `models/original_classifier/<实验编号>/`，包括 `model.pth`、`config.json`、`metrics.json`。
默认模型位置由 `models/original_classifier.json` 记录。使用保存的参数识别一张新图片：

```powershell
.\.venv\Scripts\python.exe tools\predict_original.py "D:\path\image.jpg"
```

`--bundle` 可指定其他导出目录。预测时无需下载预训练权重或再次训练。

固定模型对照服务直接加载上述模型，使用其原实验的同一批独立测试图。
按已勾选的次数在内存中作 Arnold 像素置乱，避免 JPEG 再压缩干扰比较；不训练、不调参。
同时复测原图基线，并核对测试文件和模型的 SHA256。结果按新实验保存，包含
`summary.csv`、`test_predictions.csv`、各条件的混淆矩阵和 `comparison.png`。

固定原图模型的准确率下降只反映它面对置乱输入的性能变化，不能证明密码安全，
也不能代表掌握逆变换或使用置乱样本重新训练的攻击者。

## 在置乱图片上微调

Arnold 2 次置乱模型训练服务沿用原模型的数据划分，以原图模型初始化，
用置乱训练图片微调全部 ResNet-18 层。验证集用于降低学习率、选最佳轮次和早停，
模型确定后只进行一次最终测试。所有变体留在原图所属划分中，不做逆变换或 JPEG 再压缩。
训练采用 CPU BF16，验证及测试采用 FP32；每轮保存训练曲线、最佳权重及最后一轮训练状态。

置乱模型单独保存到 `models/scrambled_classifier/<实验编号>/`，默认位置记录在
`models/scrambled_classifier.json`，不会替换原图模型。预测已有 224×224 置乱图：

```powershell
.\.venv\Scripts\python.exe tools\predict_scrambled.py "D:\path\scrambled.png"
```

若输入的是原图，添加 `--from-original`，工具会先执行模型对应的置乱变换，再分类。
仅对模型训练时的固定变换进行验证，不代表对其他置乱算法或次数具有相同效果。

数据增强对照模型只改变训练增强：原图先 RandomResizedCrop（面积 0.85–1.0、
宽高比 0.9–1.1）、50% 水平翻转、轻微颜色调整（亮度/对比度/饱和度各 0.1），
然后 Arnold 置乱。验证和测试不增强，划分和其余训练设置沿用无增强基线。
`augmentation_comparison.json` 记录在相同固定训练图上评估的两个模型及验证、测试成绩；
增强训练过程中的准确率与无增强训练准确率不直接比较。新模型总会单独导出，
只有验证表现更优或相同才替换默认置乱模型；增强模型位置另存于
`models/scrambled_classifier_augmented.json`，测试成绩不参与默认模型选择。

上述专项训练服务保留在后端，页面已统一为常规训练和已保存模型测试两个入口。
代码仓库不包含数据集、模型权重及本地实验产物；已有模型测试需先在本地生成模型和数据记录。

## 回归检查

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

工作区回归检查使用独立临时目录，覆盖任务停止与清理、版本管理、模型复用、指标导出及数据重合校验。`tests/test_pipeline.py` 是另行运行的端到端脚本，会生成合成数据并执行训练。

## Roadmap

- [x] Arnold 变换置乱与评估
- [ ] Logistic 混沌置乱
- [ ] Lorenz 混沌置乱
- [ ] 元胞自动机置乱
- [ ] ViT 模型对比验证
- [ ] 多数据集泛化验证

## License

[MIT](./LICENSE)
