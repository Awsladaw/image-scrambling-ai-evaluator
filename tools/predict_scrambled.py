"""加载置乱识别模型；默认输入已置乱图，--from-original 可先置乱原图再预测。"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
import torch
from torchvision import models, transforms
import config
from src.scrambling.arnold import scramble


def predict(image_path, bundle=None, from_original=False):
    torch.set_num_threads(8)
    if bundle is None:
        bundle = json.loads((config.MODEL_DIR / "scrambled_classifier.json").read_text(encoding="utf-8"))["bundle_dir"]
    bundle = Path(bundle)
    settings = json.loads((bundle / "config.json").read_text(encoding="utf-8"))
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(settings["classes"]))
    model.load_state_dict(torch.load(bundle / "model.pth", map_location="cpu", weights_only=True))
    model.eval()
    prep = settings["preprocess"]
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        if from_original:
            image = transforms.Resize(tuple(prep["resize"]))(image)
            pixels = scramble(np.asarray(image), settings["trained_iterations"])
            image = Image.fromarray(pixels)
        elif image.size != tuple(prep["resize"]):
            raise ValueError("已置乱图片须与训练尺寸一致；直接缩放会改变置乱结构")
        tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize(prep["mean"], prep["std"])])(image)
    with torch.inference_mode():
        probabilities = model(tensor.unsqueeze(0)).softmax(1)[0]
    values = {cls: float(p) for cls, p in zip(settings["classes"], probabilities)}
    return {"prediction": max(values, key=values.get), "probabilities": values,
            "experiment_id": settings["experiment_id"], "trained_iterations": settings["trained_iterations"],
            "input_was_original": from_original}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("--bundle")
    parser.add_argument("--from-original", action="store_true")
    args = parser.parse_args()
    print(json.dumps(predict(args.image, args.bundle, args.from_original), ensure_ascii=False, indent=2))
