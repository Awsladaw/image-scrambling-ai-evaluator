"""后台任务管理：任务状态、进度、结果。

用于 Web 系统把「导入 / 置乱 / AI评估」等耗时操作放到后台线程执行，
前端通过轮询 GET /api/jobs/{id} 获取实时进度。
"""

import threading
import time
import uuid


class Job:
    """一个后台任务。状态机：pending -> running -> done / error。"""

    def __init__(self, kind, params):
        self.id = uuid.uuid4().hex[:12]
        self.kind = kind
        self.params = params
        self.status = "pending"   # pending / running / done / error
        self.progress = 0.0
        self.message = "排队中…"
        self.result = None
        self.error = None
        self.created_at = time.time()
        self._lock = threading.Lock()

    def set_progress(self, percent, message=""):
        with self._lock:
            self.progress = max(0.0, min(100.0, float(percent)))
            if message:
                self.message = message

    def to_dict(self):
        with self._lock:
            return {
                "id": self.id,
                "kind": self.kind,
                "status": self.status,
                "progress": round(self.progress, 1),
                "message": self.message,
                "result": self.result,
                "error": self.error,
            }


class JobManager:
    def __init__(self, keep=50):
        self._jobs = {}
        self._lock = threading.Lock()
        self._keep = keep

    def submit(self, kind, params, target):
        """提交后台任务。target(job) 在工作线程中执行，通过 job.set_progress 更新进度。"""
        job = Job(kind, params)
        with self._lock:
            self._jobs[job.id] = job

        def _run():
            job.status = "running"
            try:
                job.result = target(job)
                job.status = "done"
                job.set_progress(100.0, "完成")
            except Exception as exc:  # 任务内任何异常都转为 error 状态
                job.status = "error"
                job.error = f"{type(exc).__name__}: {exc}"
                job.set_progress(100.0, "失败")

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        return job

    def get(self, job_id):
        with self._lock:
            return self._jobs.get(job_id)

    def list(self):
        with self._lock:
            return [j.to_dict() for j in self._jobs.values()]


# 模块级单例，供 Web 层直接使用
job_manager = JobManager()
