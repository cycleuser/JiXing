import hashlib
import json
import logging
import os
import platform
import socket
import sqlite3
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TimestampedMemory:
    memory_id: str
    session_id: str
    timestamp: str
    content: str
    content_hash: str
    semantic_vector: Optional[np.ndarray] = None
    spatial_context: dict = field(default_factory=dict)
    temporal_links: list[str] = field(default_factory=list)
    spatial_links: list[str] = field(default_factory=list)
    semantic_similarity: list[tuple[str, float]] = field(default_factory=list)
    compression_level: int = 1
    importance_score: float = 1.0
    access_count: int = 0
    last_accessed: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        data = asdict(self)
        if self.semantic_vector is not None:
            data["semantic_vector"] = self.semantic_vector.tolist()
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "TimestampedMemory":
        if data.get("semantic_vector") is not None:
            data["semantic_vector"] = np.array(data["semantic_vector"])
        return cls(**data)


@dataclass
class SpatialContext:
    device_id: str
    location: Optional[str] = None
    ip_address: Optional[str] = None
    hostname: Optional[str] = None
    os_info: Optional[str] = None
    cwd: Optional[str] = None
    environment_tags: list[str] = field(default_factory=list)

    @classmethod
    def collect(cls) -> "SpatialContext":
        hostname = socket.gethostname()
        try:
            ip = socket.gethostbyname(hostname)
        except Exception:
            ip = "unknown"
        return cls(
            device_id=hashlib.md5(f"{hostname}{platform.node()}".encode()).hexdigest()[:16],
            location=os.environ.get("JIXING_LOCATION", None),
            ip_address=ip,
            hostname=hostname,
            os_info=f"{platform.system()} {platform.release()}",
            cwd=os.getcwd(),
            environment_tags=_extract_env_tags(),
        )

    def to_dict(self) -> dict:
        return asdict(self)


def _extract_env_tags() -> list[str]:
    tags = []
    env_markers = ["PROJECT", "WORK", "HOME", "DEV", "PROD"]
    for key, value in os.environ.items():
        if any(m in key.upper() for m in env_markers):
            tags.append(f"{key}={value[:20]}")
    return tags


def _compute_content_hash(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()[:16]


class TemporalMemoryLinker:
    def __init__(self, max_temporal_distance_seconds: int = 3600):
        self.max_temporal_distance = max_temporal_distance_seconds
        self._link_cache: dict[str, list[str]] = {}

    def compute_temporal_distance(self, ts1: str, ts2: str) -> float:
        dt1 = datetime.fromisoformat(ts1.replace("Z", "+00:00"))
        dt2 = datetime.fromisoformat(ts2.replace("Z", "+00:00"))
        return abs((dt1 - dt2).total_seconds())

    def are_temporally_linked(self, ts1: str, ts2: str) -> bool:
        return self.compute_temporal_distance(ts1, ts2) <= self.max_temporal_distance

    def find_temporal_neighbors(
        self, target_ts: str, all_memories: list[TimestampedMemory]
    ) -> list[tuple[TimestampedMemory, float]]:
        distances = []
        for mem in all_memories:
            dist = self.compute_temporal_distance(target_ts, mem.timestamp)
            if dist <= self.max_temporal_distance * 24:
                distances.append((mem, dist))
        distances.sort(key=lambda x: x[1])
        return distances


class SpatialMemoryLinker:
    def __init__(self):
        self._spatial_index: dict[str, set[str]] = {}

    def index_by_spatial_context(
        self, memory: TimestampedMemory, all_memories: list[TimestampedMemory]
    ):
        if not memory.spatial_context:
            return

        device_id = memory.spatial_context.get("device_id")
        if not device_id:
            return

        if device_id not in self._spatial_index:
            self._spatial_index[device_id] = set()

        same_device = [
            m
            for m in all_memories
            if m.spatial_context.get("device_id") == device_id and m.memory_id != memory.memory_id
        ]
        for m in same_device:
            self._spatial_index[device_id].add(m.memory_id)

    def find_spatially_related(
        self, memory: TimestampedMemory, all_memories: list[TimestampedMemory]
    ) -> list[TimestampedMemory]:
        device_id = memory.spatial_context.get("device_id")
        if not device_id or device_id not in self._spatial_index:
            return []
        memory_id_set = self._spatial_index[device_id]
        return [m for m in all_memories if m.memory_id in memory_id_set]


class SemanticCompressor:
    def __init__(self, compression_threshold: int = 500):
        self.compression_threshold = compression_threshold
        self._summary_cache: dict[str, str] = {}

    def needs_compression(self, content: str) -> bool:
        return len(content) > self.compression_threshold

    def compress(self, memory: TimestampedMemory) -> TimestampedMemory:
        if not self.needs_compression(memory.content):
            return memory

        words = memory.content.split()
        if len(words) <= 50:
            return memory

        summary_length = min(100, len(words) // 2)
        summary_words = words[:summary_length]
        summary = " ".join(summary_words) + "..."

        compressed = TimestampedMemory(
            memory_id=memory.memory_id,
            session_id=memory.session_id,
            timestamp=memory.timestamp,
            content=summary,
            content_hash=memory.content_hash,
            semantic_vector=memory.semantic_vector,
            spatial_context=memory.spatial_context,
            temporal_links=memory.temporal_links,
            spatial_links=memory.spatial_links,
            semantic_similarity=memory.semantic_similarity,
            compression_level=memory.compression_level + 1,
            importance_score=memory.importance_score * 0.9,
            access_count=memory.access_count,
            last_accessed=memory.last_accessed,
            metadata={
                **memory.metadata,
                "compressed": True,
                "original_length": len(memory.content),
                "compression_ratio": len(summary) / len(memory.content),
            },
        )
        return compressed

    def compute_importance(
        self, memory: TimestampedMemory, context: Optional[dict] = None
    ) -> float:
        base_score = 1.0
        length_factor = min(1.0, len(memory.content) / 500)
        recency_factor = self._compute_recency(memory.timestamp)
        access_factor = min(1.0, memory.access_count / 10)

        if context:
            if context.get("has_code"):
                base_score *= 1.2
            if context.get("has_math"):
                base_score *= 1.1

        importance = base_score * (
            0.4 + 0.3 * length_factor + 0.2 * recency_factor + 0.1 * access_factor
        )
        return min(2.0, importance)

    def _compute_recency(self, timestamp: str) -> float:
        try:
            dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            age_hours = (datetime.now(timezone.utc) - dt).total_seconds() / 3600
            return max(0.1, 1.0 - (age_hours / (24 * 365)))
        except Exception:
            return 0.5


class SpatiotemporalMemoryManager:
    _instance: Optional["SpatiotemporalMemoryManager"] = None
    _lock = threading.Lock()

    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path) if db_path else Path.home() / ".jixing" / "spatiotemporal.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.temporal_linker = TemporalMemoryLinker()
        self.spatial_linker = SpatialMemoryLinker()
        self.compressor = SemanticCompressor()
        self._memories: dict[str, TimestampedMemory] = {}
        self._session_memories: dict[str, list[str]] = {}
        self._init_db()
        self._load_memories()

    @classmethod
    def get_instance(cls, db_path: str | Path | None = None) -> "SpatiotemporalMemoryManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(db_path)
        return cls._instance

    def _init_db(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    memory_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    content TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    semantic_vector BLOB,
                    spatial_context TEXT,
                    temporal_links TEXT,
                    spatial_links TEXT,
                    semantic_similarity TEXT,
                    compression_level INTEGER DEFAULT 1,
                    importance_score REAL DEFAULT 1.0,
                    access_count INTEGER DEFAULT 0,
                    last_accessed TEXT,
                    metadata TEXT
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_session ON memories(session_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_timestamp ON memories(timestamp)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_importance
                ON memories(importance_score DESC)
            """)
            conn.commit()

    def _get_connection(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _load_memories(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM memories")
            for row in cursor.fetchall():
                data = dict(row)
                data["semantic_vector"] = None
                if data["spatial_context"]:
                    data["spatial_context"] = json.loads(data["spatial_context"])
                if data["temporal_links"]:
                    data["temporal_links"] = json.loads(data["temporal_links"])
                if data["spatial_links"]:
                    data["spatial_links"] = json.loads(data["spatial_links"])
                if data["semantic_similarity"]:
                    data["semantic_similarity"] = json.loads(data["semantic_similarity"])
                if data["metadata"]:
                    data["metadata"] = json.loads(data["metadata"])
                mem = TimestampedMemory.from_dict(data)
                self._memories[mem.memory_id] = mem
                if mem.session_id not in self._session_memories:
                    self._session_memories[mem.session_id] = []
                self._session_memories[mem.session_id].append(mem.memory_id)

    def store_memory(
        self,
        session_id: str,
        content: str,
        spatial_context: Optional[SpatialContext] = None,
        semantic_vector: Optional[np.ndarray] = None,
    ) -> TimestampedMemory:
        timestamp = datetime.now(timezone.utc).isoformat()
        memory_id = str(uuid.uuid4())
        content_hash = _compute_content_hash(content)

        spatial = spatial_context.to_dict() if spatial_context else {}

        memory = TimestampedMemory(
            memory_id=memory_id,
            session_id=session_id,
            timestamp=timestamp,
            content=content,
            content_hash=content_hash,
            semantic_vector=semantic_vector,
            spatial_context=spatial,
            temporal_links=[],
            spatial_links=[],
            semantic_similarity=[],
            compression_level=1,
            importance_score=1.0,
            access_count=1,
            last_accessed=timestamp,
            metadata={},
        )

        self._establish_links(memory)
        memory = self.compressor.compress(memory)
        memory.importance_score = self.compressor.compute_importance(memory)

        self._save_memory(memory)
        self._memories[memory_id] = memory
        if session_id not in self._session_memories:
            self._session_memories[session_id] = []
        self._session_memories[session_id].append(memory_id)

        logger.debug(f"Stored memory {memory_id} with importance {memory.importance_score:.2f}")
        return memory

    def _establish_links(self, memory: TimestampedMemory):
        all_memories = list(self._memories.values())

        temporal_neighbors = self.temporal_linker.find_temporal_neighbors(
            memory.timestamp, all_memories
        )
        for neighbor, dist in temporal_neighbors[:5]:
            memory.temporal_links.append(neighbor.memory_id)
            if memory.memory_id not in neighbor.temporal_links:
                neighbor.temporal_links.append(memory.memory_id)

        self.spatial_linker.index_by_spatial_context(memory, all_memories)
        spatial_related = self.spatial_linker.find_spatially_related(memory, all_memories)
        for related in spatial_related[:5]:
            if related.memory_id not in memory.spatial_links:
                memory.spatial_links.append(related.memory_id)

        self._compute_semantic_similarity(memory, all_memories)

    def _compute_semantic_similarity(
        self, memory: TimestampedMemory, all_memories: list[TimestampedMemory]
    ):
        if memory.semantic_vector is None:
            return

        similarities = []
        for other in all_memories:
            if other.memory_id == memory.memory_id:
                continue
            if other.semantic_vector is None:
                continue
            try:
                sim = np.dot(memory.semantic_vector, other.semantic_vector) / (
                    np.linalg.norm(memory.semantic_vector) * np.linalg.norm(other.semantic_vector)
                )
                if sim > 0.5:
                    similarities.append((other.memory_id, float(sim)))
            except Exception:
                continue

        similarities.sort(key=lambda x: x[1], reverse=True)
        memory.semantic_similarity = similarities[:10]

    def _save_memory(self, memory: TimestampedMemory):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO memories
                (memory_id, session_id, timestamp, content, content_hash, semantic_vector,
                 spatial_context, temporal_links, spatial_links, semantic_similarity,
                 compression_level, importance_score, access_count, last_accessed, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    memory.memory_id,
                    memory.session_id,
                    memory.timestamp,
                    memory.content,
                    memory.content_hash,
                    memory.semantic_vector.tobytes()
                    if memory.semantic_vector is not None
                    else None,
                    json.dumps(memory.spatial_context),
                    json.dumps(memory.temporal_links),
                    json.dumps(memory.spatial_links),
                    json.dumps(memory.semantic_similarity),
                    memory.compression_level,
                    memory.importance_score,
                    memory.access_count,
                    memory.last_accessed,
                    json.dumps(memory.metadata),
                ),
            )
            conn.commit()

    def retrieve_memories(
        self,
        query: str,
        session_id: Optional[str] = None,
        time_range: Optional[tuple[str, str]] = None,
        spatial_context: Optional[SpatialContext] = None,
        limit: int = 20,
    ) -> list[TimestampedMemory]:
        results: list[tuple[TimestampedMemory, float]] = []
        query_hash = _compute_content_hash(query)

        for memory in self._memories.values():
            if session_id and memory.session_id != session_id:
                continue

            if time_range:
                start, end = time_range
                if not (start <= memory.timestamp <= end):
                    continue

            score = 0.0

            if memory.content_hash == query_hash:
                score += 10.0
            elif query.lower() in memory.content.lower():
                score += 3.0

            score += memory.importance_score

            if memory.spatial_context and spatial_context:
                if memory.spatial_context.get("device_id") == spatial_context.device_id:
                    score += 2.0

            if memory.temporal_links:
                score += 0.5 * len(memory.temporal_links)

            if memory.spatial_links:
                score += 0.5 * len(memory.spatial_links)

            if memory.semantic_similarity:
                score += 0.3 * len(memory.semantic_similarity)

            if score > 0:
                results.append((memory, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return [m for m, _ in results[:limit]]

    def get_session_memories(self, session_id: str) -> list[TimestampedMemory]:
        memory_ids = self._session_memories.get(session_id, [])
        return [self._memories[mid] for mid in memory_ids if mid in self._memories]

    def get_linked_memories(self, memory_id: str) -> dict[str, list[TimestampedMemory]]:
        memory = self._memories.get(memory_id)
        if not memory:
            return {"temporal": [], "spatial": [], "semantic": []}

        temporal = [self._memories[mid] for mid in memory.temporal_links if mid in self._memories]
        spatial = [self._memories[mid] for mid in memory.spatial_links if mid in self._memories]
        semantic_ids = [mid for mid, _ in memory.semantic_similarity[:5]]
        semantic = [self._memories[mid] for mid in semantic_ids if mid in self._memories]

        return {"temporal": temporal, "spatial": spatial, "semantic": semantic}

    def get_statistics(self) -> dict[str, Any]:
        total = len(self._memories)
        compressed = sum(1 for m in self._memories.values() if m.compression_level > 1)
        total_links = sum(
            len(m.temporal_links) + len(m.spatial_links) + len(m.semantic_similarity)
            for m in self._memories.values()
        )
        avg_importance = (
            sum(m.importance_score for m in self._memories.values()) / total if total > 0 else 0
        )

        return {
            "total_memories": total,
            "compressed_memories": compressed,
            "total_links": total_links,
            "average_importance": avg_importance,
            "sessions": len(self._session_memories),
        }

    def prune_low_importance(self, threshold: float = 0.3, keep_count: int = 1000):
        memories = sorted(self._memories.values(), key=lambda m: m.importance_score, reverse=True)
        to_prune = memories[keep_count:]
        for memory in to_prune:
            if memory.importance_score < threshold:
                self.delete_memory(memory.memory_id)

    def delete_memory(self, memory_id: str):
        if memory_id in self._memories:
            memory = self._memories[memory_id]
            self._session_memories[memory.session_id].remove(memory_id)
            del self._memories[memory_id]
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM memories WHERE memory_id = ?", (memory_id,))
                conn.commit()
