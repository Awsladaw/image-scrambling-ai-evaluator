"""API 路由：接收前端请求，转发给 services 层，返回 JSON。

本层只做「参数校验 + 提交后台任务 + 查询结果」，不包含任何算法逻辑。
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from fastapi import APIRouter, File, Form, HTTPException, UploadFile  # noqa: E402
from pydantic import BaseModel  # noqa: E402

import config  # noqa: E402
from services import pipeline  # noqa: E402
from services.job_manager import job_manager  # noqa: E402

router = APIRouter(prefix="/api")


class ScrambleRequest(BaseModel):
    algorithm: str = "arnold"
    iterations: list = []


class EvaluateRequest(BaseModel):
    arch: str = "resnet18"
    epochs: int = 3
    batch_size: int = 16
    lr: float = 1e-3
    optimizer: str = "adam"


@router.get("/status")
def status():
    """整体状态：已导入 / 已置乱 / 已评估 的进度。"""
    return {
        "imported": (config.RAW_DIR / "train").exists(),
        "scrambled": pipeline.get_scrambled_labels(),
        "evaluated": pipeline.get_evaluated_labels(),
        "algorithms": list(pipeline.ALGORITHMS.keys()),
        "config": {
            "iterations": config.ARNOLD_ITERATIONS,
            "image_size": config.IMAGE_SIZE,
            "archs": ["resnet18", "resnet50"],
        },
    }


@router.get("/config")
def get_config():
    return {
        "iterations": config.ARNOLD_ITERATIONS,
        "image_size": config.IMAGE_SIZE,
        "archs": ["resnet18", "resnet50"],
        "defaults": {
            "epochs": config.EPOCHS,
            "batch_size": config.BATCH_SIZE,
            "lr": config.LEARNING_RATE,
            "optimizer": config.OPTIMIZER,
        },
    }


@router.post("/dataset/import")
async def dataset_import(file: UploadFile = File(...),
                         max_per_class: int = Form(None)):
    """上传 zip（含 Cat/Dog 子目录），后台导入并规范化。"""
    upload_dir = config.UPLOAD_DIR
    upload_dir.mkdir(parents=True, exist_ok=True)
    filename = Path(file.filename or "dataset.zip").name
    zip_path = upload_dir / filename
    with open(zip_path, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            f.write(chunk)

    job = job_manager.submit(
        "import", {"filename": filename, "max_per_class": max_per_class},
        lambda j: pipeline.import_dataset(zip_path, max_per_class=max_per_class, job=j))
    return {"job_id": job.id}


@router.post("/scramble")
def scramble(req: ScrambleRequest):
    """按算法 + 参数梯度执行批量置乱。"""
    iterations = req.iterations or config.ARNOLD_ITERATIONS
    job = job_manager.submit(
        "scramble", {"algorithm": req.algorithm, "iterations": iterations},
        lambda j: pipeline.run_scramble(algorithm=req.algorithm,
                                        iterations=iterations, job=j))
    return {"job_id": job.id}


@router.post("/evaluate")
def evaluate(req: EvaluateRequest):
    """AI 评估：训练原始基线 + 各置乱变体，计算识别率。"""
    job = job_manager.submit(
        "evaluate", {"arch": req.arch, "epochs": req.epochs},
        lambda j: pipeline.run_evaluate(arch=req.arch, epochs=req.epochs,
                                        batch_size=req.batch_size, lr=req.lr,
                                        optimizer=req.optimizer, job=j))
    return {"job_id": job.id}


@router.get("/jobs")
def jobs():
    return job_manager.list()


@router.get("/jobs/{job_id}")
def get_job(job_id: str):
    job = job_manager.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="任务不存在")
    return job.to_dict()


@router.get("/metrics")
def metrics():
    """识别率曲线数据 + 安全性排名。"""
    return pipeline.get_metrics()
