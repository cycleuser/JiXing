import json
import logging
import os
import platform
import socket
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import requests

logger = logging.getLogger(__name__)


@dataclass
class SystemInfo:
    timestamp: str
    hostname: str
    os_system: str
    os_release: str
    os_version: str
    python_version: str
    platform_machine: str
    platform_processor: str
    network_hostname: str
    network_ip: str
    cwd: str
    environment: dict[str, str] = field(default_factory=dict)

    @classmethod
    def collect(cls) -> "SystemInfo":
        hostname = socket.gethostname()
        try:
            ip = socket.gethostbyname(hostname)
        except Exception:
            ip = "unknown"

        return cls(
            timestamp=datetime.now(timezone.utc).isoformat(),
            hostname=hostname,
            os_system=platform.system(),
            os_release=platform.release(),
            os_version=platform.version(),
            python_version=sys.version,
            platform_machine=platform.machine(),
            platform_processor=platform.processor(),
            network_hostname=hostname,
            network_ip=ip,
            cwd=os.getcwd(),
            environment={k: v for k, v in os.environ.items() if not _is_sensitive(k)},
        )

    def to_dict(self) -> dict:
        return asdict(self)


def _is_sensitive(key: str) -> bool:
    sensitive_patterns = ["KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "AUTH"]
    upper_key = key.upper()
    return any(pattern in upper_key for pattern in sensitive_patterns)


@dataclass
class Message:
    role: str
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metrics: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Session:
    id: str
    model_provider: str
    model_name: str
    created_at: str
    updated_at: str
    system_info: dict
    messages: list[dict] = field(default_factory=list)
    total_tokens: int = 0
    total_duration_ms: int = 0
    context_limit: int = 128000
    parent_session_id: Optional[str] = None
    migration_metadata: dict = field(default_factory=dict)
    goal: str = ""

    def add_message(
        self, role: str, content: str, metrics: Optional[dict[str, Any]] = None
    ) -> Message:
        msg = Message(role=role, content=content, metrics=metrics)
        self.messages.append(msg.to_dict())
        self.updated_at = datetime.now(timezone.utc).isoformat()
        if metrics:
            if "eval_count" in metrics:
                self.total_tokens += metrics.get("eval_count", 0)
            if "duration" in metrics:
                self.total_duration_ms += metrics.get("duration", 0)
        return msg

    def to_dict(self) -> dict:
        return asdict(self)


class SessionManager:
    _instance: Optional["SessionManager"] = None
    _lock = threading.Lock()

    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path) if db_path else Path.home() / ".jixing" / "sessions.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._sessions: dict[str, Session] = {}
        self._current_session: Optional[Session] = None
        self._load_sessions()

    @classmethod
    def get_instance(cls, db_path: str | Path | None = None) -> "SessionManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(db_path)
        return cls._instance

    def _init_db(self):
        import sqlite3

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                data TEXT NOT NULL
            )
        """)
        conn.commit()
        conn.close()

    def _load_sessions(self):
        if not self.db_path.exists():
            self._init_db()
            return
        import sqlite3

        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='sessions'")
        if cursor.fetchone() is None:
            conn.close()
            self._init_db()
            return
        cursor.execute("SELECT id, data FROM sessions")
        for row in cursor.fetchall():
            session_id, data = row
            session_data = json.loads(data)
            session = Session(**session_data)
            self._sessions[session_id] = session
        conn.close()

    def _save_session(self, session: Session):
        import sqlite3

        if not self.db_path.exists():
            self._init_db()
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO sessions (id, data) VALUES (?, ?)",
            (session.id, json.dumps(session.to_dict())),
        )
        conn.commit()
        conn.close()

    def create_session(
        self,
        model_provider: str,
        model_name: str,
        system_info: Optional[dict] = None,
        context_limit: int = 128000,
        parent_session_id: Optional[str] = None,
        migration_metadata: Optional[dict] = None,
        goal: str = "",
    ) -> Session:
        now = datetime.now(timezone.utc).isoformat()
        session = Session(
            id=str(uuid.uuid4()),
            model_provider=model_provider,
            model_name=model_name,
            created_at=now,
            updated_at=now,
            system_info=system_info or SystemInfo.collect().to_dict(),
            context_limit=context_limit,
            parent_session_id=parent_session_id,
            migration_metadata=migration_metadata or {},
            goal=goal,
        )
        self._sessions[session.id] = session
        self._current_session = session
        self._save_session(session)
        logger.info(f"Created session {session.id} for {model_provider}/{model_name}")
        return session

    def get_session(self, session_id: str) -> Optional[Session]:
        return self._sessions.get(session_id)

    def get_current_session(self) -> Optional[Session]:
        return self._current_session

    def list_sessions(
        self,
        model_provider: Optional[str] = None,
        model_name: Optional[str] = None,
        limit: int = 100,
    ) -> list[Session]:
        results = list(self._sessions.values())
        if model_provider:
            results = [s for s in results if s.model_provider == model_provider]
        if model_name:
            results = [s for s in results if model_name in s.model_name]
        results.sort(key=lambda s: s.created_at, reverse=True)
        return results[:limit]

    def delete_session(self, session_id: str) -> bool:
        if session_id in self._sessions:
            del self._sessions[session_id]
            import sqlite3

            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            cursor.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            conn.commit()
            conn.close()
            if self._current_session and self._current_session.id == session_id:
                self._current_session = None
            return True
        return False

    def query_messages(
        self,
        session_id: Optional[str] = None,
        query: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        results = []
        sessions_to_search = (
            [self._sessions.get(session_id)] if session_id else self._sessions.values()
        )
        for session in sessions_to_search:
            if session is None:
                continue
            for msg in session.messages:
                if query is None or query.lower() in msg.get("content", "").lower():
                    results.append(
                        {
                            "session_id": session.id,
                            "model_provider": session.model_provider,
                            "model_name": session.model_name,
                            **msg,
                        }
                    )
        results.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return results[:limit]

    def merge_sessions(
        self,
        session_ids: list[str],
        merge_mode: str = "timeline",
        new_model_provider: Optional[str] = None,
        new_model_name: Optional[str] = None,
    ) -> Optional[Session]:
        if not session_ids:
            return None

        sessions = [self._sessions.get(sid) for sid in session_ids]
        sessions = [s for s in sessions if s is not None]
        if not sessions:
            return None

        all_messages = []
        for session in sessions:
            for msg in session.messages:
                msg_with_source = {**msg, "source_session_id": session.id}
                all_messages.append(msg_with_source)

        if merge_mode == "timeline":
            all_messages.sort(key=lambda m: m.get("timestamp", ""))
        elif merge_mode == "reverse_timeline":
            all_messages.sort(key=lambda m: m.get("timestamp", ""), reverse=True)
        else:
            ordered_messages = []
            for sid in session_ids:
                session = self._sessions.get(sid)
                if session:
                    for msg in session.messages:
                        msg_with_source = {**msg, "source_session_id": sid}
                        ordered_messages.append(msg_with_source)
            all_messages = ordered_messages

        now = datetime.now(timezone.utc).isoformat()
        merged_session = Session(
            id=str(uuid.uuid4()),
            model_provider=new_model_provider or sessions[0].model_provider,
            model_name=new_model_name or sessions[0].model_name,
            created_at=now,
            updated_at=now,
            system_info=sessions[0].system_info,
            context_limit=max(s.context_limit for s in sessions),
            parent_session_id=session_ids[0] if session_ids else None,
            migration_metadata={
                "merged_from": session_ids,
                "merge_mode": merge_mode,
                "merge_timestamp": now,
            },
        )

        total_tokens = 0
        total_duration = 0
        clean_messages = []
        for msg in all_messages:
            clean_msg = {k: v for k, v in msg.items() if k != "source_session_id"}
            clean_messages.append(clean_msg)
            if msg.get("metrics"):
                if "eval_count" in msg["metrics"]:
                    total_tokens += msg["metrics"].get("eval_count", 0)
                if "duration" in msg["metrics"]:
                    total_duration += msg["metrics"].get("duration", 0)

        merged_session.messages = clean_messages
        merged_session.total_tokens = total_tokens
        merged_session.total_duration_ms = total_duration

        self._sessions[merged_session.id] = merged_session
        self._current_session = merged_session
        self._save_session(merged_session)

        logger.info(f"Merged {len(sessions)} sessions into {merged_session.id}")
        return merged_session


class ModelAdapter:
    def __init__(self, session: Session):
        self.session = session

    def prepare_messages(self, prompt: str, images: Optional[list[str]] = None) -> list[dict]:
        if images:
            content = [{"type": "text", "text": prompt}]
            for img in images:
                content.append({"type": "image_url", "image_url": {"url": f"file://{img}"}})
        else:
            content = prompt
        return [{"role": "user", "content": content}]

    def parse_response(self, response: Any) -> tuple[str, dict[str, Any]]:
        raise NotImplementedError


class OllamaAdapter(ModelAdapter):
    def __init__(self, session: Session, base_url: str = "http://localhost:11434"):
        super().__init__(session)
        self.base_url = base_url

    def prepare_messages(self, prompt: str, images: Optional[list[str]] = None) -> list[dict]:
        if images:
            content = prompt
            for img in images:
                with open(img, "rb") as f:
                    import base64

                    img_data = base64.b64encode(f.read()).decode("utf-8")
                content = [
                    {"role": "user", "content": prompt},
                    {"role": "user", "content": f"data:image/jpeg;base64,{img_data}"},
                ]
        else:
            content = [{"role": "user", "content": prompt}]
        return content

    def parse_response(self, response: requests.Response) -> tuple[str, dict[str, Any]]:
        data = response.json()
        response_text = data.get("response", "")
        metrics = {
            "eval_count": data.get("eval_count", 0),
            "eval_duration": data.get("eval_duration", 0),
            "load_duration": data.get("load_duration", 0),
            "prompt_eval_count": data.get("prompt_eval_count", 0),
            "prompt_eval_duration": data.get("prompt_eval_duration", 0),
            "total_duration": data.get("total_duration", 0),
        }
        return response_text, metrics

    def run(
        self, prompt: str, images: Optional[list[str]] = None, **kwargs
    ) -> tuple[str, dict[str, Any], dict[str, Any]]:
        start_time = time.time()
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.session.model_name,
            "prompt": prompt,
            "stream": False,
        }
        if images:
            payload["images"] = images
        payload.update(kwargs)

        response = requests.post(url, json=payload, timeout=kwargs.get("timeout", 300))
        response.raise_for_status()

        end_time = time.time()
        response_text, metrics = self.parse_response(response)
        metrics["wall_time_ms"] = int((end_time - start_time) * 1000)

        return response_text, metrics, {}

    def run_stream(
        self, prompt: str, history: Optional[list[dict]] = None, images: Optional[list[str]] = None, **kwargs
    ):
        """Stream response from Ollama API. Yields chunks of text."""
        url = f"{self.base_url}/api/chat"
        messages = history or [{"role": "user", "content": prompt}]
        if not history and prompt:
            messages = [{"role": "user", "content": prompt}]

        payload = {
            "model": self.session.model_name,
            "messages": messages,
            "stream": True,
        }
        if images:
            payload["images"] = images
        payload.update(kwargs)

        response = requests.post(url, json=payload, stream=True, timeout=kwargs.get("timeout", 300))
        response.raise_for_status()

        full_response = ""
        metrics = {}
        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line)
                    chunk = data.get("message", {}).get("content", "")
                    if chunk:
                        full_response += chunk
                        yield chunk, False, {}
                    if data.get("done", False):
                        metrics = {
                            "eval_count": data.get("eval_count", 0),
                            "eval_duration": data.get("eval_duration", 0),
                            "load_duration": data.get("load_duration", 0),
                            "prompt_eval_count": data.get("prompt_eval_count", 0),
                            "prompt_eval_duration": data.get("prompt_eval_duration", 0),
                            "total_duration": data.get("total_duration", 0),
                        }
                        yield "", True, metrics
                except json.JSONDecodeError:
                    continue

    def run_with_idle_timeout(
        self,
        prompt: str,
        timeout: int = 600,
        idle_timeout: int = 120,
        **kwargs,
    ) -> tuple[str, dict[str, Any], dict[str, Any]]:
        """Run with streaming and idle timeout detection.

        Tries /api/generate first, falls back to /api/chat if it fails.
        Detects when the model appears stuck (no output for idle_timeout seconds).

        Args:
            prompt: The prompt to send
            timeout: Total timeout for the request
            idle_timeout: Max seconds without any output before considering stuck

        Returns:
            Tuple of (response_text, metrics, usage)
        """
        try:
            return self._stream_generate(prompt, timeout, idle_timeout, **kwargs)
        except Exception as e:
            logger.warning(f"/api/generate failed: {e}, falling back to /api/chat")
            return self._stream_chat(prompt, timeout, idle_timeout, **kwargs)

    def _stream_generate(
        self,
        prompt: str,
        timeout: int = 600,
        idle_timeout: int = 120,
        **kwargs,
    ) -> tuple[str, dict[str, Any], dict[str, Any]]:
        """Stream via /api/generate endpoint."""
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.session.model_name,
            "prompt": prompt,
            "stream": True,
        }
        payload.update(kwargs)
        return self._do_stream(url, payload, "response", timeout, idle_timeout)

    def _stream_chat(
        self,
        prompt: str,
        timeout: int = 600,
        idle_timeout: int = 120,
        **kwargs,
    ) -> tuple[str, dict[str, Any], dict[str, Any]]:
        """Stream via /api/chat endpoint."""
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.session.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
        }
        payload.update(kwargs)
        return self._do_stream(url, payload, "message.content", timeout, idle_timeout)

    def _do_stream(
        self,
        url: str,
        payload: dict,
        content_key: str,
        timeout: int,
        idle_timeout: int,
    ) -> tuple[str, dict[str, Any], dict[str, Any]]:
        """Common streaming logic for both generate and chat APIs."""
        start_time = time.time()
        last_activity = start_time
        full_response = ""
        metrics = {}

        try:
            response = requests.post(url, json=payload, stream=True, timeout=timeout)
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        if content_key == "response":
                            chunk = data.get("response", "")
                        else:
                            chunk = data.get("message", {}).get("content", "")
                        if chunk:
                            full_response += chunk
                            last_activity = time.time()
                        if data.get("done", False):
                            metrics = {
                                "eval_count": data.get("eval_count", 0),
                                "eval_duration": data.get("eval_duration", 0),
                                "load_duration": data.get("load_duration", 0),
                                "prompt_eval_count": data.get("prompt_eval_count", 0),
                                "prompt_eval_duration": data.get("prompt_eval_duration", 0),
                                "total_duration": data.get("total_duration", 0),
                            }
                            break
                    except json.JSONDecodeError:
                        continue

                if time.time() - last_activity > idle_timeout and full_response:
                    raise TimeoutError(
                        f"Model appears stuck (no output for {idle_timeout}s). "
                        f"Generated {len(full_response)} chars before stalling."
                    )

                if time.time() - start_time > timeout:
                    raise TimeoutError(f"Total timeout of {timeout}s exceeded")

        except requests.exceptions.Timeout:
            raise TimeoutError(f"Request timeout after {timeout}s")

        if not full_response:
            raise ValueError("Model returned empty response")

        metrics["wall_time_ms"] = int((time.time() - start_time) * 1000)
        return full_response, metrics, {}

    def list_models(self) -> list[dict]:
        """List available models from Ollama."""
        url = f"{self.base_url}/api/tags"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data.get("models", [])

    def show_model(self, model_name: str) -> dict:
        """Show model information."""
        url = f"{self.base_url}/api/show"
        response = requests.post(url, json={"name": model_name}, timeout=30)
        response.raise_for_status()
        return response.json()

    def pull_model(self, model_name: str, stream: bool = False):
        """Pull a model from Ollama registry."""
        url = f"{self.base_url}/api/pull"
        payload = {"name": model_name, "stream": stream}
        if stream:
            response = requests.post(url, json=payload, stream=True, timeout=600)
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    yield json.loads(line)
        else:
            response = requests.post(url, json=payload, timeout=600)
            response.raise_for_status()
            return response.json()

    def delete_model(self, model_name: str) -> bool:
        """Delete a model from Ollama."""
        url = f"{self.base_url}/api/delete"
        response = requests.delete(url, json={"name": model_name}, timeout=30)
        response.raise_for_status()
        return response.status_code == 200


class MoxingAdapter(ModelAdapter):
    def __init__(self, session: Session, base_url: str = "http://localhost:8080"):
        super().__init__(session)
        self.base_url = base_url

    def parse_response(self, response: requests.Response) -> tuple[str, dict[str, Any]]:
        data = response.json()
        response_text = data.get("result", {}).get("text", "")
        metrics = {
            "tokens": data.get("usage", {}).get("completion_tokens", 0),
            "latency_ms": data.get("latency_ms", 0),
        }
        return response_text, metrics

    def run(self, prompt: str, **kwargs) -> tuple[str, dict[str, Any], dict[str, Any]]:
        start_time = time.time()
        url = f"{self.base_url}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": self.session.model_name,
            "messages": [{"role": "user", "content": prompt}],
        }
        payload.update(kwargs)

        response = requests.post(
            url, json=payload, headers=headers, timeout=kwargs.get("timeout", 300)
        )
        response.raise_for_status()

        response_text, metrics = self.parse_response(response)
        metrics["wall_time_ms"] = int((time.time() - start_time) * 1000)

        return response_text, metrics, {}


class ModelRunner:
    _adapters: dict[str, type[ModelAdapter]] = {
        "ollama": OllamaAdapter,
        "moxing": MoxingAdapter,
    }

    @classmethod
    def register_adapter(cls, name: str, adapter_class: type[ModelAdapter]):
        cls._adapters[name] = adapter_class

    @classmethod
    def run(
        cls,
        provider: str,
        model_name: str,
        prompt: str,
        session: Optional[Session] = None,
        system_info: Optional[dict] = None,
        context_manager=None,
        goal: str = "",
        compressor_fn=None,
        **kwargs,
    ) -> tuple[Session, str, dict[str, Any]]:
        if session is None:
            manager = SessionManager.get_instance()
            session = manager.create_session(provider, model_name, system_info)

        if context_manager and goal:
            session.goal = goal

        if context_manager and session.messages:
            usage = context_manager.get_context_usage(session.messages)
            if usage["needs_compression"]:
                result = context_manager.auto_handle_overflow(
                    session_id=session.id,
                    messages=session.messages,
                    goal=session.goal or goal,
                    compressor_fn=compressor_fn,
                    session_creator=lambda: SessionManager.get_instance().create_session(
                        model_provider=provider,
                        model_name=model_name,
                        context_limit=session.context_limit * 2,
                        parent_session_id=session.id,
                        goal=session.goal or goal,
                    ),
                    longer_context_limit=session.context_limit * 2,
                )
                if result["action"] == "migrated":
                    session = result["new_session"]
                    for msg in result["messages"]:
                        session.messages.append(msg)
                elif result["action"].startswith("compressed"):
                    session.messages = result["messages"]

        adapter_class = cls._adapters.get(provider)
        if not adapter_class:
            raise ValueError(
                f"Unknown provider: {provider}. Available: {list(cls._adapters.keys())}"
            )

        adapter = adapter_class(session)

        user_msg = session.add_message("user", prompt)
        logger.debug(f"User message added to session {session.id}")

        response_text, metrics, extra = adapter.run(prompt, **kwargs)

        session.add_message(
            "assistant",
            response_text,
            metrics={
                **metrics,
                "prompt_tokens": user_msg.metrics.get("eval_count", 0) if user_msg.metrics else 0,
            },
        )

        manager = SessionManager.get_instance()
        manager._save_session(session)

        from .db import Database

        try:
            db = Database()
            db.save_session(session)
        except Exception as e:
            logger.warning(f"Failed to save to database: {e}")

        return session, response_text, metrics
