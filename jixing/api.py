from dataclasses import dataclass, field
from importlib import import_module
from typing import Any, Optional

from .core import ModelRunner, SessionManager, SystemInfo
from .db import Database


def _get_version():
    return import_module("jixing").__version__


@dataclass
class ToolResult:
    success: bool
    data: Any = None
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "metadata": self.metadata,
        }


def run_ollama(
    *,
    model: str,
    prompt: str,
    session_id: Optional[str] = None,
    system_info: Optional[dict] = None,
    **kwargs,
) -> ToolResult:
    try:
        manager = SessionManager.get_instance()
        session = None
        if session_id:
            session = manager.get_session(session_id)

        if session is None:
            session = manager.create_session(
                model_provider="ollama",
                model_name=model,
                system_info=system_info or SystemInfo.collect().to_dict(),
            )

        result_session, response_text, metrics = ModelRunner.run(
            provider="ollama",
            model_name=model,
            prompt=prompt,
            session=session,
            **kwargs,
        )

        return ToolResult(
            success=True,
            data={
                "session_id": result_session.id,
                "response": response_text,
                "metrics": metrics,
            },
            metadata={"version": _get_version()},
        )

    except Exception as e:
        return ToolResult(success=False, error=str(e), metadata={"version": _get_version()})


def run_moxing(
    *,
    model: str,
    prompt: str,
    session_id: Optional[str] = None,
    system_info: Optional[dict] = None,
    **kwargs,
) -> ToolResult:
    try:
        manager = SessionManager.get_instance()
        session = None
        if session_id:
            session = manager.get_session(session_id)

        if session is None:
            session = manager.create_session(
                model_provider="moxing",
                model_name=model,
                system_info=system_info or SystemInfo.collect().to_dict(),
            )

        result_session, response_text, metrics = ModelRunner.run(
            provider="moxing",
            model_name=model,
            prompt=prompt,
            session=session,
            **kwargs,
        )

        return ToolResult(
            success=True,
            data={
                "session_id": result_session.id,
                "response": response_text,
                "metrics": metrics,
            },
            metadata={"version": _get_version()},
        )

    except Exception as e:
        return ToolResult(success=False, error=str(e), metadata={"version": _get_version()})


def query_sessions(
    *,
    model_provider: Optional[str] = None,
    model_name: Optional[str] = None,
    limit: int = 100,
) -> ToolResult:
    try:
        manager = SessionManager.get_instance()
        sessions = manager.list_sessions(
            model_provider=model_provider,
            model_name=model_name,
            limit=limit,
        )
        return ToolResult(
            success=True,
            data=[s.to_dict() for s in sessions],
            metadata={"version": _get_version(), "count": len(sessions)},
        )
    except Exception as e:
        return ToolResult(success=False, error=str(e), metadata={"version": _get_version()})


def get_session(*, session_id: str) -> ToolResult:
    try:
        manager = SessionManager.get_instance()
        session = manager.get_session(session_id)
        if session is None:
            return ToolResult(
                success=False,
                error=f"Session {session_id} not found",
                metadata={"version": _get_version()},
            )
        return ToolResult(
            success=True,
            data=session.to_dict(),
            metadata={"version": _get_version()},
        )
    except Exception as e:
        return ToolResult(success=False, error=str(e), metadata={"version": _get_version()})


def delete_session(*, session_id: str) -> ToolResult:
    try:
        manager = SessionManager.get_instance()
        deleted = manager.delete_session(session_id)
        if not deleted:
            return ToolResult(
                success=False,
                error=f"Session {session_id} not found",
                metadata={"version": _get_version()},
            )
        return ToolResult(
            success=True,
            data={"deleted": session_id},
            metadata={"version": _get_version()},
        )
    except Exception as e:
        return ToolResult(success=False, error=str(e), metadata={"version": _get_version()})


def search_messages(
    *,
    query: str,
    session_id: Optional[str] = None,
    model_provider: Optional[str] = None,
    limit: int = 100,
) -> ToolResult:
    try:
        db = Database()
        results = db.search_messages(
            query=query,
            session_id=session_id,
            model_provider=model_provider,
            limit=limit,
        )
        return ToolResult(
            success=True,
            data=results,
            metadata={"version": _get_version(), "count": len(results)},
        )
    except Exception as e:
        return ToolResult(success=False, error=str(e), metadata={"version": _get_version()})


def get_stats() -> ToolResult:
    try:
        db = Database()
        stats = db.get_stats()
        return ToolResult(
            success=True,
            data=stats,
            metadata={"version": _get_version()},
        )
    except Exception as e:
        return ToolResult(success=False, error=str(e), metadata={"version": _get_version()})


def get_system_info() -> ToolResult:
    try:
        info = SystemInfo.collect().to_dict()
        return ToolResult(
            success=True,
            data=info,
            metadata={"version": _get_version()},
        )
    except Exception as e:
        return ToolResult(success=False, error=str(e), metadata={"version": _get_version()})


def merge_sessions(
    *,
    session_ids: list[str],
    merge_mode: str = "timeline",
    model_provider: Optional[str] = None,
    model_name: Optional[str] = None,
) -> ToolResult:
    try:
        manager = SessionManager.get_instance()
        merged = manager.merge_sessions(
            session_ids=session_ids,
            merge_mode=merge_mode,
            new_model_provider=model_provider,
            new_model_name=model_name,
        )
        if merged is None:
            return ToolResult(
                success=False,
                error="No valid sessions to merge",
                metadata={"version": _get_version()},
            )
        return ToolResult(
            success=True,
            data=merged.to_dict(),
            metadata={"version": _get_version(), "merged_count": len(session_ids)},
        )
    except Exception as e:
        return ToolResult(success=False, error=str(e), metadata={"version": _get_version()})
