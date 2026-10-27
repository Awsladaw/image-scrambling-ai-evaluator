"""串行后台任务及持久化状态，上传阶段也持有任务锁。"""
import json
import shutil
import threading
import time
import uuid
from pathlib import Path
import config
from src.task_control import JobCancelled


class BusyError(RuntimeError):
    pass


class Job:
    def __init__(self, kind, params, on_change=None):
        self.id = uuid.uuid4().hex[:12]
        self.kind, self.params = kind, params
        self.status, self.progress, self.message = "pending", 0.0, "准备中…"
        self.result = self.error = None
        self.created_at, self.finished_at = time.time(), None
        self._lock = threading.Lock()
        self._on_change = on_change
        self._stop = threading.Event()
        self._finalizing = False
        self._outputs = {}
        self.cleanup_complete = None
        self.cleanup_errors = []

    def check_cancelled(self):
        if self._stop.is_set():
            raise JobCancelled("用户已停止任务，正在清理本次任务生成的文件。")

    def track_output(self, path):
        candidate = Path(path).resolve()
        roots = [config.DATA_ROOT / 'datasets', config.UPLOAD_DIR,
                 config.RESULTS_DIR / 'experiments', config.MODEL_DIR / 'original_classifier',
                 config.MODEL_DIR / 'scrambled_classifier']
        allowed = next((root.resolve() for root in roots
                        if candidate != root.resolve() and candidate.is_relative_to(root.resolve())), None)
        if allowed is None:
            raise ValueError(f"清理目标不在任务输出目录内：{candidate}")
        with self._lock:
            self.check_cancelled()
            if candidate in self._outputs:
                return
            if candidate.exists() or candidate.is_symlink():
                raise ValueError(f"不能将已有文件或目录登记为本次任务输出：{candidate}")
            self._outputs[candidate] = allowed

    def cleanup_outputs(self):
        errors = []
        # 仅删除任务开始后新建且登记过的目标，删除前再次核对绝对路径边界。
        for path, root in self._outputs.items():
            try:
                resolved = path.resolve()
                if resolved != path or resolved == root or not resolved.is_relative_to(root):
                    raise ValueError("清理目标的实际路径发生变化")
                if path.is_symlink():
                    raise ValueError("清理目标不能是符号链接")
                if path.is_dir():
                    shutil.rmtree(path)
                elif path.exists():
                    path.unlink()
            except Exception as exc:
                errors.append(f"{path}: {exc}")
        return errors

    def begin_commit(self):
        with self._lock:
            self.check_cancelled()
            self._finalizing = True

    def set_progress(self, percent, message=""):
        with self._lock:
            self.check_cancelled()
            self.progress = max(0.0, min(100.0, float(percent)))
            if message:
                self.message = message
        if self._on_change:
            self._on_change(False)

    def to_dict(self):
        with self._lock:
            row = {k: getattr(self, k) for k in (
                "id", "kind", "params", "status", "progress", "message",
                "result", "error", "created_at", "finished_at", "cleanup_complete", "cleanup_errors")}
            row["can_stop"] = (self.kind in ("import", "scramble", "evaluate") and
                               self.status in ("pending", "running") and not self._finalizing)
            return row


class JobManager:
    def __init__(self, keep=50, state_file=None):
        self._jobs = {}
        self._lock = threading.RLock()
        self._keep, self._last_save = keep, 0.0
        self._state_file = Path(state_file) if state_file else None
        if self._state_file and self._state_file.exists():
            for row in json.loads(self._state_file.read_text(encoding="utf-8")):
                job = Job(row["kind"], row.get("params", {}), self._persist)
                for key, value in row.items():
                    if not key.startswith("_") and key != "can_stop":
                        setattr(job, key, value)
                if job.status in ("pending", "running", "stopping"):
                    job.status = "error"
                    job.error = "服务重启，任务已中断；已保存的文件仍保留，请重新提交任务。"
                    job.message, job.finished_at = "任务中断", time.time()
                self._jobs[job.id] = job
            self._persist(True)

    def _persist(self, force=False):
        if not self._state_file:
            return
        with self._lock:
            now = time.monotonic()
            if not force and now - self._last_save < 1:
                return
            self._state_file.parent.mkdir(parents=True, exist_ok=True)
            temporary = self._state_file.with_suffix(".tmp")
            temporary.write_text(json.dumps(
                [j.to_dict() for j in self._jobs.values()], ensure_ascii=False,
                indent=2), encoding="utf-8")
            temporary.replace(self._state_file)
            self._last_save = now

    def reserve(self, kind, params):
        """先原子占用工作区，再上传或启动线程。"""
        with self._lock:
            active = self.active()
            if active:
                name = {"import": "数据导入", "scramble": "图像置乱",
                        "evaluate": "AI 评估", "select": "切换数据集"}.get(active["kind"], "后台任务")
                raise BusyError(f"{name}正在执行，请等待完成后再操作。")
            job = Job(kind, params, self._persist)
            self._jobs[job.id] = job
            while len(self._jobs) > self._keep:
                oldest = next(iter(self._jobs))
                if oldest == job.id:
                    break
                del self._jobs[oldest]
            self._persist(True)
            return job

    def request_stop(self, job_id):
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                raise KeyError(job_id)
            with job._lock:
                if job.status == "stopping":
                    return {"job_id": job.id, "status": "stopping"}
                if (job.status not in ("pending", "running") or job._finalizing or
                        job.kind not in ("import", "scramble", "evaluate")):
                    raise BusyError("任务已结束或正在保存最终结果，当前无法停止")
                job._stop.set()
                job.status, job.message = "stopping", "正在停止，请等待当前图片或批次结束…"
            self._persist(True)
            return {"job_id": job.id, "status": "stopping"}

    def finish(self, job, result=None, error=None):
        with self._lock:
            cancelled = job._stop.is_set() or isinstance(error, JobCancelled)
            if not cancelled:
                with job._lock:
                    job._finalizing = True
            if cancelled:
                with job._lock:
                    job.status, job.message = "stopping", "正在清理本次任务生成的文件…"
                self._persist(True)
        # 清理时仍保持 stopping，状态查询可用，其他修改任务仍被互斥锁阻止。
        cleanup_errors = job.cleanup_outputs() if cancelled else []
        with self._lock:
            with job._lock:
                job.status = "cancelled" if cancelled else "error" if error else "done"
                if isinstance(error, Exception):
                    error = str(error)
                job.result, job.error = (None if cancelled else result), (None if cancelled else error)
                if cancelled:
                    job.cleanup_errors, job.cleanup_complete = cleanup_errors, not cleanup_errors
                    job.message = "已停止，本次任务生成的文件已清理" if not cleanup_errors else "已停止，部分文件清理失败"
                else:
                    job.message = "失败" if error else "完成"
                if not error and not cancelled:
                    job.progress = 100.0
                job.finished_at = time.time()
            self._persist(True)

    def start(self, job, target):
        with self._lock:
            if self._jobs.get(job.id) is not job:
                raise RuntimeError("任务未预留或已经启动")
            if job._stop.is_set():
                self.finish(job, error=JobCancelled("用户已停止任务"))
                return job
            if job.status != "pending":
                raise RuntimeError("任务未预留或已经启动")
            with job._lock:
                job.status, job.message = "running", "开始执行…"
            self._persist(True)

        def run():
            try:
                job.check_cancelled()
                self.finish(job, result=target(job))
            except JobCancelled as exc:
                self.finish(job, error=exc)
            except Exception as exc:
                self.finish(job, error=f"{type(exc).__name__}: {exc}")
        try:
            threading.Thread(target=run, daemon=True).start()
        except Exception as exc:
            self.finish(job, error=str(exc))
            raise
        return job

    def submit(self, kind, params, target):
        return self.start(self.reserve(kind, params), target)

    def get(self, job_id):
        with self._lock:
            return self._jobs.get(job_id)

    def list(self):
        with self._lock:
            return [j.to_dict() for j in self._jobs.values()]

    def active(self):
        with self._lock:
            return next((j.to_dict() for j in self._jobs.values()
                         if j.status in ("pending", "running", "stopping")), None)


job_manager = JobManager(state_file=config.DATA_ROOT / "jobs.json")
