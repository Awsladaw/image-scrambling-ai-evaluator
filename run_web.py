"""启动 Web 系统（FastAPI + 前端）。

用法：
    py -3.11 run_web.py
浏览器访问 http://127.0.0.1:8000
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=config.WEB_HOST, port=config.WEB_PORT)
