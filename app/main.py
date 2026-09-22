"""FastAPI 应用入口。

启动方式（项目根目录）：
    py -3.11 run_web.py
然后浏览器访问 http://127.0.0.1:8000
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from app import routes  # noqa: E402

app = FastAPI(title="图像置乱算法 AI 安全评价系统", version="1.0")

# 先注册 API 路由，再把静态前端挂载到根路径（/api/* 优先匹配）
app.include_router(routes.router)

STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
