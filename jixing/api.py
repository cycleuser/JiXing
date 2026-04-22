from dataclasses import dataclass, field
from importlib import import_module
from typing import Any, Callable, Optional

from .core import ModelRunner, SessionManager, SystemInfo
from .db import Database
from .context_manager import ContextWindowManager, CompressionConfig
from .memory import ConversationArchiver


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


def get_context_usage(
    *,
    session_id: str,
    context_limit: int = 128000,
) -> ToolResult:
    try:
        manager = SessionManager.get_instance()
        session = manager.get_session(session_id)
        if session is None:
            return ToolResult(
                success=False,
                error=f"Session {session_id} not found",
                metadata={"version": _get_version()},
            )

        ctx_manager = ContextWindowManager(context_limit=context_limit)
        usage = ctx_manager.get_context_usage(session.messages)

        return ToolResult(
            success=True,
            data=usage,
            metadata={"version": _get_version()},
        )
    except Exception as e:
        return ToolResult(success=False, error=str(e), metadata={"version": _get_version()})


def compress_context(
    *,
    session_id: str,
    target_ratio: float = 0.5,
    strategy: str = "smart",
    goal: str = "",
) -> ToolResult:
    try:
        manager = SessionManager.get_instance()
        session = manager.get_session(session_id)
        if session is None:
            return ToolResult(
                success=False,
                error=f"Session {session_id} not found",
                metadata={"version": _get_version()},
            )

        ctx_manager = ContextWindowManager(context_limit=session.context_limit)
        result = ctx_manager.compress_context(
            messages=session.messages,
            target_ratio=target_ratio,
            goal=goal or session.goal,
        )

        session.messages = result.compressed_messages
        manager._save_session(session)

        ctx_manager.archive_messages_to_jsonl(
            session_id,
            result.archived_messages,
            {"compression_ratio": result.compression_ratio, "strategy": result.strategy_used},
        )

        return ToolResult(
            success=True,
            data={
                "session_id": session_id,
                "original_tokens": result.original_token_count,
                "compressed_tokens": result.compressed_token_count,
                "compression_ratio": result.compression_ratio,
                "target_ratio": result.target_ratio,
                "strategy": result.strategy_used,
                "quality_score": result.quality_score,
                "message_count": len(result.compressed_messages),
                "archived_count": len(result.archived_messages),
            },
            metadata={"version": _get_version()},
        )
    except Exception as e:
        return ToolResult(success=False, error=str(e), metadata={"version": _get_version()})


def migrate_session(
    *,
    session_id: str,
    new_context_limit: Optional[int] = None,
    goal: str = "",
) -> ToolResult:
    try:
        manager = SessionManager.get_instance()
        session = manager.get_session(session_id)
        if session is None:
            return ToolResult(
                success=False,
                error=f"Session {session_id} not found",
                metadata={"version": _get_version()},
            )

        ctx_manager = ContextWindowManager(context_limit=session.context_limit)
        snapshot = ctx_manager.create_snapshot(
            session_id=session_id,
            messages=session.messages,
            goal=goal or session.goal,
        )

        def create_new_session():
            return manager.create_session(
                model_provider=session.model_provider,
                model_name=session.model_name,
                context_limit=new_context_limit or session.context_limit * 2,
                parent_session_id=session_id,
                migration_metadata=snapshot.to_dict(),
                goal=goal or session.goal,
            )

        migration = ctx_manager.migrate_session(
            snapshot=snapshot,
            new_session_creator=create_new_session,
            context_limit=new_context_limit,
        )

        ctx_manager.archive_messages_to_jsonl(
            session_id,
            session.messages,
            {"migration": True, "snapshot_id": snapshot.snapshot_id},
        )

        return ToolResult(
            success=True,
            data={
                "source_session_id": session_id,
                "new_session_id": migration["new_session"].id,
                "migration_messages": len(migration["migration_messages"]),
                "snapshot_id": snapshot.snapshot_id,
                "original_messages": snapshot.full_message_count,
                "context_limit_before": session.context_limit,
                "context_limit_after": new_context_limit or session.context_limit * 2,
            },
            metadata={"version": _get_version()},
        )
    except Exception as e:
        return ToolResult(success=False, error=str(e), metadata={"version": _get_version()})


def persist_runtime_state(
    *,
    session_id: str,
    goal: str,
    current_task: str,
    last_round_input: str,
    last_round_output: str = "",
    intermediate_requirements: Optional[list[str]] = None,
    decisions_made: Optional[list[str]] = None,
    pending_actions: Optional[list[str]] = None,
) -> ToolResult:
    try:
        ctx_manager = ContextWindowManager()
        state_path = ctx_manager.persist_runtime_state(
            session_id=session_id,
            goal=goal,
            intermediate_requirements=intermediate_requirements or [],
            decisions_made=decisions_made or [],
            current_task=current_task,
            last_round_input=last_round_input,
            last_round_output=last_round_output,
            pending_actions=pending_actions or [],
        )

        return ToolResult(
            success=True,
            data={
                "state_file": str(state_path),
                "session_id": session_id,
            },
            metadata={"version": _get_version()},
        )
    except Exception as e:
        return ToolResult(success=False, error=str(e), metadata={"version": _get_version()})


def archive_session(
    *,
    session_id: str,
) -> ToolResult:
    try:
        manager = SessionManager.get_instance()
        session = manager.get_session(session_id)
        if session is None:
            return ToolResult(
                success=False,
                error=f"Session {session_id} not found",
                metadata={"version": _get_version()},
            )

        archiver = ConversationArchiver()
        archive_path = archiver.archive_session(
            session_id=session_id,
            messages=session.messages,
            metadata={
                "model_provider": session.model_provider,
                "model_name": session.model_name,
                "total_tokens": session.total_tokens,
            },
        )

        return ToolResult(
            success=True,
            data={
                "session_id": session_id,
                "archive_file": str(archive_path),
                "message_count": len(session.messages),
            },
            metadata={"version": _get_version()},
        )
    except Exception as e:
        return ToolResult(success=False, error=str(e), metadata={"version": _get_version()})


def load_archived_session(
    *,
    session_id: str,
) -> ToolResult:
    try:
        archiver = ConversationArchiver()
        session_data = archiver.load_session(session_id)
        if session_data is None:
            return ToolResult(
                success=False,
                error=f"No archived session found for {session_id}",
                metadata={"version": _get_version()},
            )

        return ToolResult(
            success=True,
            data=session_data,
            metadata={"version": _get_version()},
        )
    except Exception as e:
        return ToolResult(success=False, error=str(e), metadata={"version": _get_version()})


def list_archived_sessions() -> ToolResult:
    try:
        archiver = ConversationArchiver()
        sessions = archiver.list_archived_sessions()

        return ToolResult(
            success=True,
            data=sessions,
            metadata={"version": _get_version(), "count": len(sessions)},
        )
    except Exception as e:
        return ToolResult(success=False, error=str(e), metadata={"version": _get_version()})


def auto_handle_overflow(
    *,
    session_id: str,
    goal: str = "",
    compressor_fn: Optional[Callable] = None,
    longer_context_limit: Optional[int] = None,
) -> ToolResult:
    try:
        manager = SessionManager.get_instance()
        session = manager.get_session(session_id)
        if session is None:
            return ToolResult(
                success=False,
                error=f"Session {session_id} not found",
                metadata={"version": _get_version()},
            )

        ctx_manager = ContextWindowManager(context_limit=session.context_limit)

        def create_new_session():
            return manager.create_session(
                model_provider=session.model_provider,
                model_name=session.model_name,
                context_limit=longer_context_limit or session.context_limit * 2,
                parent_session_id=session_id,
                goal=goal or session.goal,
            )

        result = ctx_manager.auto_handle_overflow(
            session_id=session_id,
            messages=session.messages,
            goal=goal or session.goal,
            compressor_fn=compressor_fn,
            session_creator=create_new_session,
            longer_context_limit=longer_context_limit,
        )

        if result["action"] == "migrated":
            session.messages = result["messages"]
            manager._save_session(result["new_session"])

        return ToolResult(
            success=True,
            data={
                "action": result["action"],
                "session_id": session_id,
                "new_session_id": result.get("new_session", {}).id if result.get("new_session") else None,
                "usage": result.get("usage"),
                "compression_result": result.get("compression_result"),
                "state_file": result.get("state_file"),
            },
            metadata={"version": _get_version()},
        )
    except Exception as e:
        return ToolResult(success=False, error=str(e), metadata={"version": _get_version()})
