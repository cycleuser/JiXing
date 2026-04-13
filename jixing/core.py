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
    ) -> Session:
        now = datetime.now(timezone.utc).isoformat()
        session = Session(
            id=str(uuid.uuid4()),
            model_provider=model_provider,
            model_name=model_name,
            created_at=now,
            updated_at=now,
            system_info=system_info or SystemInfo.collect().to_dict(),
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
        **kwargs,
    ) -> tuple[Session, str, dict[str, Any]]:
        if session is None:
            manager = SessionManager.get_instance()
            session = manager.create_session(provider, model_name, system_info)

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

        return session, response_text, metrics
