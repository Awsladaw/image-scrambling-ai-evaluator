"""可选的协作式取消检查；命令行调用不依赖 Web 任务管理器。"""


class JobCancelled(Exception):
    """用户要求停止任务。"""


def check_cancelled(job):
    callback = getattr(job, "check_cancelled", None)
    if callback:
        callback()


def begin_commit(job):
    """发布结果前最后检查取消；发布期间拒绝新的停止请求。"""
    callback = getattr(job, "begin_commit", None)
    if callback:
        callback()


def track_output(job, path):
    """在创建新目录或文件前登记，只用于停止任务时清理。"""
    callback = getattr(job, "track_output", None)
    if callback:
        callback(path)
