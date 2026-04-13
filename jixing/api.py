from dataclasses import dataclass, field
from typing import Any, Optional

from .core import ModelRunner, SessionManager, SystemInfo
from .db import Database

__version__ = "1.0.0"


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
            metadata={"version": __version__},
        )

    except Exception as e:
        return ToolResult(success=False, error=str(e), metadata={"version": __version__})


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
            metadata={"version": __version__},
        )

    except Exception as e:
        return ToolResult(success=False, error=str(e), metadata={"version": __version__})


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
            metadata={"version": __version__, "count": len(sessions)},
        )
    except Exception as e:
        return ToolResult(success=False, error=str(e), metadata={"version": __version__})


def get_session(*, session_id: str) -> ToolResult:
    try:
        manager = SessionManager.get_instance()
        session = manager.get_session(session_id)
        if session is None:
            return ToolResult(
                success=False,
                error=f"Session {session_id} not found",
                metadata={"version": __version__},
            )
        return ToolResult(
            success=True,
            data=session.to_dict(),
            metadata={"version": __version__},
        )
    except Exception as e:
        return ToolResult(success=False, error=str(e), metadata={"version": __version__})


def delete_session(*, session_id: str) -> ToolResult:
    try:
        manager = SessionManager.get_instance()
        deleted = manager.delete_session(session_id)
        if not deleted:
            return ToolResult(
                success=False,
                error=f"Session {session_id} not found",
                metadata={"version": __version__},
            )
        return ToolResult(
            success=True,
            data={"deleted": session_id},
            metadata={"version": __version__},
        )
    except Exception as e:
        return ToolResult(success=False, error=str(e), metadata={"version": __version__})


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
            metadata={"version": __version__, "count": len(results)},
        )
    except Exception as e:
        return ToolResult(success=False, error=str(e), metadata={"version": __version__})


def get_stats() -> ToolResult:
    try:
        db = Database()
        stats = db.get_stats()
        return ToolResult(
            success=True,
            data=stats,
            metadata={"version": __version__},
        )
    except Exception as e:
        return ToolResult(success=False, error=str(e), metadata={"version": __version__})


def get_system_info() -> ToolResult:
    try:
        info = SystemInfo.collect().to_dict()
        return ToolResult(
            success=True,
            data=info,
            metadata={"version": __version__},
        )
    except Exception as e:
        return ToolResult(success=False, error=str(e), metadata={"version": __version__})
