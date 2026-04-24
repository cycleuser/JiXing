__version__ = "1.0.6"

from .api import ToolResult

__all__ = ["__version__", "ToolResult"]


def __getattr__(name: str):
    if name == "run_ollama":
        from .api import run_ollama

        return run_ollama
    if name == "run_moxing":
        from .api import run_moxing

        return run_moxing
    if name == "query_sessions":
        from .api import query_sessions

        return query_sessions
    if name == "get_session":
        from .api import get_session

        return get_session
    if name == "run_long_running_task":
        from .api import run_long_running_task

        return run_long_running_task
    if name == "get_task_checkpoints":
        from .api import get_task_checkpoints

        return get_task_checkpoints
    if name == "list_task_results":
        from .api import list_task_results

        return list_task_results
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
