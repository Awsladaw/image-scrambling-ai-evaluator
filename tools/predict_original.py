"""加载保存的原图猫狗分类器识别一张图片。"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from PIL import Image
from torchvision import models, transforms
import config


def predict(image_path, bundle=None):
    if bundle is None:
        bundle = json.loads((config.MODEL_DIR / "original_classifier.json").read_text(encoding="utf-8"))["bundle_dir"]
    bundle = Path(bundle)
    settings = json.loads((bundle / "config.json").read_text(encoding="utf-8"))
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(settings["classes"]))
    model.load_state_dict(torch.load(bundle / "model.pth", map_location="cpu", weights_only=True))
    model.eval()
    prep = settings["preprocess"]
    transform = transforms.Compose([transforms.Resize(tuple(prep["resize"])), transforms.ToTensor(),
                                    transforms.Normalize(prep["mean"], prep["std"])])
    with Image.open(image_path) as image:
        tensor = transform(image.convert("RGB")).unsqueeze(0)
    with torch.inference_mode():
        probabilities = model(tensor).softmax(1)[0]
    result = {cls: float(p) for cls, p in zip(settings["classes"], probabilities)}
    return {"prediction": max(result, key=result.get), "probabilities": result,
            "experiment_id": settings["experiment_id"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("--bundle")
    args = parser.parse_args()
    print(json.dumps(predict(args.image, args.bundle), ensure_ascii=False, indent=2))
