"""API 路由：接收前端请求，转发给 services 层，返回 JSON。

本层只做「参数校验 + 提交后台任务 + 查询结果」，不包含任何算法逻辑。
"""

import sys
from pathlib import Path
from typing import Literal
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, Request, Query  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402
from fastapi.responses import FileResponse

import config  # noqa: E402
from services import pipeline, storage  # noqa: E402
from services.job_manager import BusyError, job_manager  # noqa: E402
from services import strong_original, frozen_scramble, adaptive_scramble  # noqa: E402
from services import saved_evaluation
from src.task_control import JobCancelled  # noqa: E402

router = APIRouter(prefix="/api")


class ScrambleRequest(BaseModel):
    algorithm: Literal["arnold"] = "arnold"
    iterations: list[int] = Field(default_factory=list, max_length=24)


class EvaluateRequest(BaseModel):
    arch: Literal["resnet18", "resnet50"] = "resnet18"
    epochs: int = Field(default=3, ge=1)
    batch_size: int = Field(default=16, ge=1)
    lr: float = Field(default=1e-3, gt=0)
    optimizer: Literal["adam", "sgd"] = "adam"
    image_size: int = Field(default=config.IMAGE_SIZE, ge=32, le=512)


class DatasetSelection(BaseModel):
    dataset_id: str


class SavedEvaluationRequest(BaseModel):
    model_id: str
    dataset_id: str
    conditions: Literal['original', 'scrambled', 'both'] = 'both'
    iterations: list[int] = Field(default_factory=list, max_length=23)
    input_source: Literal['lossless', 'saved'] = 'lossless'
    batch_size: int = Field(default=32, ge=1, le=128)


class AdaptiveTrainRequest(BaseModel):
    iterations: int = Field(default=2, ge=1, le=23)
    epochs: int = Field(default=10, ge=1, le=30)
    batch_size: int = Field(default=32, ge=2, le=128)
    backbone_lr: float = Field(default=3e-4, gt=0, le=.01)
    head_lr: float = Field(default=1e-3, gt=0, le=.1)
    augmentation: bool = False


def reserve(kind, params):
    try:
        return job_manager.reserve(kind, params)
    except BusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


def submit(kind, params, target):
    return job_manager.start(reserve(kind, params), target)


@router.get("/status")
def status():
    """整体状态：已导入 / 已置乱 / 已评估 的进度。"""
    active = job_manager.active()
    dataset = storage.active_dataset()
    importing = active and active["kind"] == "import"
    changing = active and active["kind"] in ("import", "scramble")
    jobs = job_manager.list()
    return {
        "imported": bool(dataset) and not importing,
        "scrambled": [] if changing else pipeline.get_scrambled_labels(),
        "evaluated": [] if changing else pipeline.get_evaluated_labels(),
        "dataset": dataset,
        "busy": bool(active), "active_job": active,
        "latest_job": jobs[-1] if jobs else None,
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
                         max_per_class: int = Form(None, ge=3),
                         validation_ratio: float = Form(0.0, ge=0, lt=0.8)):
    """上传 zip（含 Cat/Dog 子目录），后台导入并规范化。"""
    filename = Path(file.filename or "dataset.zip").name
    if not filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="请选择 ZIP 数据集")
    job = reserve("import", {"filename": filename, "max_per_class": max_per_class,
                             "validation_ratio": validation_ratio})
    zip_path = None
    try:
        config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        zip_path = config.UPLOAD_DIR / (uuid.uuid4().hex[:12] + "_" + filename)
        job.track_output(zip_path)
        job.set_progress(0, "正在接收数据集…")
        with open(zip_path, "wb") as f:
            while chunk := await file.read(1024 * 1024):
                job.check_cancelled()
                f.write(chunk)
        job_manager.start(job, lambda j: pipeline.import_dataset(
            zip_path, max_per_class=max_per_class, job=j, source_name=filename,
            validation_ratio=validation_ratio))
    except JobCancelled as exc:
        job_manager.finish(job, error=exc)
    except BaseException as exc:
        job_manager.finish(job, error=f"上传失败：{exc}")
        raise
    finally:
        await file.close()
    return {"job_id": job.id}


@router.post("/dataset/upload")
async def dataset_upload(request: Request, filename: str = Query(...),
                         max_per_class: int | None = Query(None, ge=3),
                         validation_ratio: float = Query(0.0, ge=0, lt=0.8)):
    """直接接收 ZIP 流：上传开始即登记任务，允许上传期间停止。"""
    filename = Path(filename).name
    if not filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="请选择 ZIP 数据集")
    job = reserve("import", {"filename": filename, "max_per_class": max_per_class,
                             "validation_ratio": validation_ratio})
    zip_path = config.UPLOAD_DIR / (uuid.uuid4().hex[:12] + "_" + filename)
    try:
        job.track_output(zip_path)
        config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        job.set_progress(0, "正在接收数据集…")
        received = 0
        with zip_path.open("wb") as output:
            async for chunk in request.stream():
                job.check_cancelled()
                output.write(chunk)
                received += len(chunk)
                job.set_progress(0, f"正在接收数据集：{received / 1024 / 1024:.1f} MB")
        job_manager.start(job, lambda j: pipeline.import_dataset(
            zip_path, max_per_class=max_per_class, job=j, source_name=filename,
            validation_ratio=validation_ratio))
    except BaseException as exc:
        job_manager.finish(job, error=exc if isinstance(exc, JobCancelled) else f"上传中断：{exc}")
        if job.status != "cancelled":
            zip_path.unlink(missing_ok=True)
        if not isinstance(exc, JobCancelled) and job.status != "cancelled":
            raise
    return {"job_id": job.id}


@router.post("/scramble")
def scramble(req: ScrambleRequest):
    """按算法 + 参数梯度执行批量置乱。"""
    iterations = req.iterations or config.ARNOLD_ITERATIONS
    if any(k < 1 or k > 1000 for k in iterations):
        raise HTTPException(status_code=422, detail="迭代次数应在 1 到 1000 之间")
    job = submit(
        "scramble", {"algorithm": req.algorithm, "iterations": iterations},
        lambda j: pipeline.run_scramble(algorithm=req.algorithm,
                                        iterations=iterations, job=j))
    return {"job_id": job.id}


@router.post("/evaluate")
def evaluate(req: EvaluateRequest):
    """AI 评估：训练原始基线 + 各置乱变体，计算识别率。"""
    job = submit(
        "evaluate", req.model_dump(),
        lambda j: pipeline.run_evaluate(arch=req.arch, epochs=req.epochs,
                                        batch_size=req.batch_size, lr=req.lr,
                                        optimizer=req.optimizer, image_size=req.image_size, job=j))
    return {"job_id": job.id}


@router.get("/jobs")
def jobs():
    return job_manager.list()


@router.get('/models')
def saved_models():
    return {'models': saved_evaluation.public_models()}


@router.post('/evaluate/saved')
def evaluate_saved(req: SavedEvaluationRequest):
    if any(k < 1 or k > 23 for k in req.iterations):
        raise HTTPException(status_code=422, detail='Arnold 置乱次数应在 1 到 23 之间')
    try:
        saved_evaluation.resolve_selection(req.model_id, req.dataset_id, req.conditions,
                                           req.iterations, req.input_source)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    params = {**req.model_dump(), 'training_mode': 'saved_model_evaluation'}
    job = submit('evaluate', params, lambda j: saved_evaluation.run(j, **req.model_dump()))
    return {'job_id': job.id}


@router.get('/evaluations/{experiment_id}/download/{kind}')
def download_evaluation(experiment_id: str, kind: Literal['summary', 'predictions']):
    record = storage.experiment(experiment_id)
    if not record or record['status'] != 'done' or record['parameters'].get('training_mode') != 'saved_model_evaluation':
        raise HTTPException(status_code=404, detail='此模型测试结果不存在或尚未完成')
    root = storage.experiment_root().resolve() / record['id']
    name = 'evaluation_summary.json' if kind == 'summary' else 'test_predictions.csv'
    path = (root / name).resolve()
    if not path.is_relative_to(storage.experiment_root().resolve()) or not path.is_file():
        raise HTTPException(status_code=404, detail='结果文件不存在')
    return FileResponse(path, filename=f'{experiment_id}_{name}',
                        media_type='application/json' if kind == 'summary' else 'text/csv')


@router.post("/original/train-strong")
def train_strong_original():
    """完整原图数据上训练预训练特征分类器，调参和测试独立。"""
    job = submit("evaluate", {"arch": "resnet18", "image_size": 224, "batch_size": 64,
                              "training_mode": "transfer_linear_probe"}, strong_original.run)
    return {"job_id": job.id}


@router.get("/jobs/{job_id}")
def get_job(job_id: str):
    job = job_manager.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="任务不存在")
    return job.to_dict()


@router.post("/jobs/{job_id}/stop")
def stop_job(job_id: str):
    try:
        return job_manager.request_stop(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="任务不存在") from exc
    except BusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/original/evaluate-scrambled")
def evaluate_saved_original(req: ScrambleRequest):
    iterations = sorted(set(req.iterations or config.ARNOLD_ITERATIONS))
    if any(k < 1 or k > 1000 for k in iterations):
        raise HTTPException(status_code=422, detail="迭代次数应在 1 到 1000 之间")
    job = submit("evaluate", {"training_mode": "frozen_model_scramble", "iterations": iterations},
                 lambda j: frozen_scramble.run(j, iterations))
    return {"job_id": job.id}


@router.get("/metrics")
def metrics(experiment_id: str | None = None):
    """识别率曲线数据 + 安全性排名。"""
    active = job_manager.active()
    if not experiment_id and active and active["kind"] in ("import", "scramble"):
        return {"rows": [], "ranking": [], "baseline": 0.5,
                "algorithm": "arnold", "experiment": None}
    try:
        return pipeline.get_metrics(experiment_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/scrambled/train-adaptive")
def train_adaptive_scrambled(req: AdaptiveTrainRequest):
    params = {**req.model_dump(), "training_mode": "adaptive_scramble_finetune",
              "arch": "resnet18", "image_size": 224, "lr": req.backbone_lr}
    job = submit("evaluate", params, lambda j: adaptive_scramble.run(j, **req.model_dump()))
    return {"job_id": job.id}


@router.get("/datasets")
def datasets():
    return storage.catalog()


@router.post("/datasets/select")
def select_dataset(req: DatasetSelection):
    job = submit("select", {"dataset_id": req.dataset_id},
                 lambda j: storage.select_dataset(req.dataset_id))
    return {"job_id": job.id}


@router.get("/experiments")
def experiments():
    return {"experiments": storage.experiments()}
