"""模型构建、数据加载与评估 —— 供 train/evaluate 共用。"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_model(num_classes=2, arch="resnet50"):
    """构建从零初始化（无预训练权重）的分类模型。"""
    if arch == "resnet50":
        model = models.resnet50(weights=None)
    elif arch == "resnet18":
        model = models.resnet18(weights=None)
    else:
        raise ValueError(f"不支持的模型架构: {arch}")
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def get_transforms(image_size):
    """训练与评估统一使用最小化变换，保证各算法横向可比。"""
    base = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    return {
        "train": transforms.Compose(base),
        "eval": transforms.Compose(base),
    }


def get_dataloaders(data_dir, image_size, batch_size, num_workers=0):
    """读取 data_dir/{train,test} 两个类目录，返回 loader 与类名。"""
    data_dir = str(data_dir)
    tr = get_transforms(image_size)
    train_ds = datasets.ImageFolder(data_dir + "/train", transform=tr["train"])
    test_ds = datasets.ImageFolder(data_dir + "/test", transform=tr["eval"])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers)
    return train_loader, test_loader, train_ds.classes


@torch.no_grad()
def evaluate(model, loader, device):
    """返回模型在 loader 上的分类准确率（识别率）。"""
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.size(0)
    return correct / total if total else 0.0
