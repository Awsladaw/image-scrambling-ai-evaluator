"""FastAPI 应用入口。

启动方式（项目根目录）：
    py -3.11 run_web.py
然后浏览器访问 http://127.0.0.1:8000
"""

import sys
from pathlib import Path
from contextlib import asynccontextmanager

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from app import routes  # noqa: E402
from services import storage  # noqa: E402


@asynccontextmanager
async def lifespan(app):
    storage.catalog()
    for experiment in storage.experiments():
        if experiment["status"] == "running":
            experiment.update(status="error", error="服务重启，实验中断；部分结果仍保留")
            storage.save_experiment(experiment)
    yield


app = FastAPI(title="图像置乱算法 AI 安全评价系统", version="1.1", lifespan=lifespan)

# 先注册 API 路由，再把静态前端挂载到根路径（/api/* 优先匹配）
app.include_router(routes.router)

STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
