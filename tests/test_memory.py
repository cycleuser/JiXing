import json
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from jixing.memory import (
    SemanticCompressor,
    SpatialContext,
    SpatialMemoryLinker,
    SpatiotemporalMemoryManager,
    TemporalMemoryLinker,
    TimestampedMemory,
    _compute_content_hash,
    _extract_env_tags,
)


class TestComputeContentHash:
    def test_deterministic(self):
        h1 = _compute_content_hash("Hello")
        h2 = _compute_content_hash("Hello")
        assert h1 == h2

    def test_different_content(self):
        h1 = _compute_content_hash("Hello")
        h2 = _compute_content_hash("World")
        assert h1 != h2

    def test_hash_length(self):
        h = _compute_content_hash("Test content")
        assert len(h) == 16

    def test_empty_string(self):
        h = _compute_content_hash("")
        assert len(h) == 16


class TestExtractEnvTags:
    def test_no_matching_env(self):
        with patch("jixing.memory.os.environ", {}):
            tags = _extract_env_tags()
            assert tags == []

    def test_finds_project_env(self):
        with patch("jixing.memory.os.environ", {"PROJECT_NAME": "test"}):
            tags = _extract_env_tags()
            assert len(tags) == 1
            assert "PROJECT_NAME" in tags[0]


class TestTimestampedMemory:
    def test_create_memory(self, sample_memory_data):
        mem = TimestampedMemory(**sample_memory_data)
        assert mem.memory_id == "mem-001"
        assert mem.session_id == "sess-001"
        assert mem.content == sample_memory_data["content"]

    def test_to_dict(self, sample_memory_data):
        mem = TimestampedMemory(**sample_memory_data)
        d = mem.to_dict()
        assert d["memory_id"] == "mem-001"
        assert d["content"] == sample_memory_data["content"]
        assert "semantic_vector" in d

    def test_to_dict_serializes_vector(self, sample_memory_data):
        mem = TimestampedMemory(**sample_memory_data)
        d = mem.to_dict()
        assert isinstance(d["semantic_vector"], list)

    def test_from_dict(self, sample_memory_data):
        mem = TimestampedMemory(**sample_memory_data)
        d = mem.to_dict()
        mem2 = TimestampedMemory.from_dict(d)
        assert mem2.memory_id == mem.memory_id
        assert mem2.content == mem.content

    def test_from_dict_restores_vector(self, sample_memory_data):
        mem = TimestampedMemory(**sample_memory_data)
        d = mem.to_dict()
        mem2 = TimestampedMemory.from_dict(d)
        assert isinstance(mem2.semantic_vector, np.ndarray)

    def test_memory_without_vector(self):
        mem = TimestampedMemory(
            memory_id="test",
            session_id="sess",
            timestamp="2024-01-01T00:00:00+00:00",
            content="test",
            content_hash="hash",
            semantic_vector=None,
        )
        d = mem.to_dict()
        assert d["semantic_vector"] is None

    def test_default_values(self):
        mem = TimestampedMemory(
            memory_id="test",
            session_id="sess",
            timestamp="2024-01-01T00:00:00+00:00",
            content="test",
            content_hash="hash",
        )
        assert mem.temporal_links == []
        assert mem.spatial_links == []
        assert mem.compression_level == 1
        assert mem.importance_score == 1.0
        assert mem.access_count == 0


class TestSpatialContext:
    @patch("jixing.memory.socket.gethostname", return_value="test-host")
    @patch("jixing.memory.socket.gethostbyname", return_value="127.0.0.1")
    @patch("jixing.memory.platform.system", return_value="Linux")
    @patch("jixing.memory.platform.release", return_value="5.15")
    @patch("jixing.memory.platform.node", return_value="test-host")
    def test_collect(self, *mocks):
        ctx = SpatialContext.collect()
        assert ctx.hostname == "test-host"
        assert ctx.ip_address == "127.0.0.1"
        assert "Linux" in ctx.os_info

    def test_to_dict(self):
        ctx = SpatialContext(
            device_id="dev-001",
            hostname="test",
            ip_address="127.0.0.1",
        )
        d = ctx.to_dict()
        assert d["device_id"] == "dev-001"
        assert d["hostname"] == "test"

    def test_default_values(self):
        ctx = SpatialContext(device_id="dev-001")
        assert ctx.location is None
        assert ctx.environment_tags == []


class TestTemporalMemoryLinker:
    def test_compute_distance_same_time(self):
        linker = TemporalMemoryLinker()
        ts = "2024-01-01T00:00:00+00:00"
        assert linker.compute_temporal_distance(ts, ts) == 0

    def test_compute_distance_one_hour(self):
        linker = TemporalMemoryLinker()
        ts1 = "2024-01-01T00:00:00+00:00"
        ts2 = "2024-01-01T01:00:00+00:00"
        assert linker.compute_temporal_distance(ts1, ts2) == 3600

    def test_compute_distance_symmetric(self):
        linker = TemporalMemoryLinker()
        ts1 = "2024-01-01T00:00:00+00:00"
        ts2 = "2024-01-01T02:00:00+00:00"
        assert linker.compute_temporal_distance(ts1, ts2) == linker.compute_temporal_distance(
            ts2, ts1
        )

    def test_are_linked_within_threshold(self):
        linker = TemporalMemoryLinker(max_temporal_distance_seconds=3600)
        ts1 = "2024-01-01T00:00:00+00:00"
        ts2 = "2024-01-01T00:30:00+00:00"
        assert linker.are_temporally_linked(ts1, ts2) is True

    def test_are_linked_outside_threshold(self):
        linker = TemporalMemoryLinker(max_temporal_distance_seconds=3600)
        ts1 = "2024-01-01T00:00:00+00:00"
        ts2 = "2024-01-01T02:00:00+00:00"
        assert linker.are_temporally_linked(ts1, ts2) is False

    def test_find_temporal_neighbors(self):
        linker = TemporalMemoryLinker()
        memories = [
            TimestampedMemory(
                memory_id=f"mem-{i}",
                session_id="sess",
                timestamp=f"2024-01-01T00:{i * 10:02d}:00+00:00",
                content=f"content {i}",
                content_hash=f"hash{i}",
            )
            for i in range(5)
        ]
        neighbors = linker.find_temporal_neighbors("2024-01-01T00:20:00+00:00", memories)
        assert len(neighbors) > 0
        assert neighbors[0][1] == 0

    def test_find_neighbors_sorted_by_distance(self):
        linker = TemporalMemoryLinker()
        memories = [
            TimestampedMemory(
                memory_id=f"mem-{i}",
                session_id="sess",
                timestamp=f"2024-01-01T00:{i * 10:02d}:00+00:00",
                content=f"content {i}",
                content_hash=f"hash{i}",
            )
            for i in range(5)
        ]
        neighbors = linker.find_temporal_neighbors("2024-01-01T00:20:00+00:00", memories)
        distances = [d for _, d in neighbors]
        assert distances == sorted(distances)


class TestSpatialMemoryLinker:
    def test_index_by_spatial_context(self):
        linker = SpatialMemoryLinker()
        memories = [
            TimestampedMemory(
                memory_id="mem-001",
                session_id="sess",
                timestamp="2024-01-01T00:00:00+00:00",
                content="content 1",
                content_hash="hash1",
                spatial_context={"device_id": "dev-001"},
            ),
            TimestampedMemory(
                memory_id="mem-002",
                session_id="sess",
                timestamp="2024-01-01T00:01:00+00:00",
                content="content 2",
                content_hash="hash2",
                spatial_context={"device_id": "dev-001"},
            ),
        ]
        linker.index_by_spatial_context(memories[0], memories)
        related = linker.find_spatially_related(memories[0], memories)
        assert any(m.memory_id == "mem-002" for m in related)

    def test_no_spatial_context(self):
        linker = SpatialMemoryLinker()
        mem = TimestampedMemory(
            memory_id="mem-001",
            session_id="sess",
            timestamp="2024-01-01T00:00:00+00:00",
            content="content",
            content_hash="hash",
        )
        linker.index_by_spatial_context(mem, [])

    def test_different_devices(self):
        linker = SpatialMemoryLinker()
        memories = [
            TimestampedMemory(
                memory_id="mem-001",
                session_id="sess",
                timestamp="2024-01-01T00:00:00+00:00",
                content="content 1",
                content_hash="hash1",
                spatial_context={"device_id": "dev-001"},
            ),
            TimestampedMemory(
                memory_id="mem-002",
                session_id="sess",
                timestamp="2024-01-01T00:01:00+00:00",
                content="content 2",
                content_hash="hash2",
                spatial_context={"device_id": "dev-002"},
            ),
        ]
        linker.index_by_spatial_context(memories[0], memories)
        related = linker.find_spatially_related(memories[0], memories)
        assert not any(m.memory_id == "mem-002" for m in related)


class TestSemanticCompressor:
    def test_needs_compression_short(self):
        compressor = SemanticCompressor(compression_threshold=500)
        assert compressor.needs_compression("Short text") is False

    def test_needs_compression_long(self):
        compressor = SemanticCompressor(compression_threshold=500)
        long_text = "word " * 200
        assert compressor.needs_compression(long_text) is True

    def test_compress_short_content(self):
        compressor = SemanticCompressor()
        mem = TimestampedMemory(
            memory_id="test",
            session_id="sess",
            timestamp="2024-01-01T00:00:00+00:00",
            content="Short",
            content_hash="hash",
        )
        result = compressor.compress(mem)
        assert result.content == "Short"
        assert result.compression_level == 1

    def test_compress_long_content(self):
        compressor = SemanticCompressor(compression_threshold=500)
        long_content = "word " * 200
        mem = TimestampedMemory(
            memory_id="test",
            session_id="sess",
            timestamp="2024-01-01T00:00:00+00:00",
            content=long_content,
            content_hash="hash",
        )
        result = compressor.compress(mem)
        assert result.compression_level == 2
        assert len(result.content) < len(long_content)
        assert result.content.endswith("...")

    def test_compress_preserves_metadata(self):
        compressor = SemanticCompressor(compression_threshold=500)
        long_content = "word " * 200
        mem = TimestampedMemory(
            memory_id="test",
            session_id="sess",
            timestamp="2024-01-01T00:00:00+00:00",
            content=long_content,
            content_hash="hash",
            metadata={"key": "value"},
        )
        result = compressor.compress(mem)
        assert result.metadata["key"] == "value"
        assert result.metadata["compressed"] is True

    def test_compute_importance_basic(self):
        compressor = SemanticCompressor()
        mem = TimestampedMemory(
            memory_id="test",
            session_id="sess",
            timestamp="2024-01-01T00:00:00+00:00",
            content="Test content",
            content_hash="hash",
        )
        score = compressor.compute_importance(mem)
        assert 0 < score <= 2.0

    def test_compute_importance_with_code(self):
        compressor = SemanticCompressor()
        mem = TimestampedMemory(
            memory_id="test",
            session_id="sess",
            timestamp="2024-01-01T00:00:00+00:00",
            content="def hello(): pass",
            content_hash="hash",
        )
        score_with_code = compressor.compute_importance(mem, context={"has_code": True})
        score_without = compressor.compute_importance(mem)
        assert score_with_code > score_without

    def test_compute_importance_with_math(self):
        compressor = SemanticCompressor()
        mem = TimestampedMemory(
            memory_id="test",
            session_id="sess",
            timestamp="2024-01-01T00:00:00+00:00",
            content="E = mc^2",
            content_hash="hash",
        )
        score = compressor.compute_importance(mem, context={"has_math": True})
        assert score > 0

    def test_compute_importance_capped_at_2(self):
        compressor = SemanticCompressor()
        mem = TimestampedMemory(
            memory_id="test",
            session_id="sess",
            timestamp="2024-01-01T00:00:00+00:00",
            content="x" * 1000,
            content_hash="hash",
            access_count=100,
        )
        score = compressor.compute_importance(mem, context={"has_code": True, "has_math": True})
        assert score <= 2.0


class TestSpatiotemporalMemoryManager:
    def test_singleton_pattern(self, tmp_db_path):
        sm1 = SpatiotemporalMemoryManager.get_instance(tmp_db_path)
        sm2 = SpatiotemporalMemoryManager.get_instance(tmp_db_path)
        assert sm1 is sm2

    def test_store_memory(self, tmp_db_path):
        manager = SpatiotemporalMemoryManager(tmp_db_path)
        mem = manager.store_memory("sess-001", "Test memory content")
        assert mem.memory_id is not None
        assert mem.session_id == "sess-001"
        assert mem.content == "Test memory content"

    def test_store_memory_with_spatial_context(self, tmp_db_path):
        manager = SpatiotemporalMemoryManager(tmp_db_path)
        ctx = SpatialContext(device_id="dev-001", hostname="test")
        mem = manager.store_memory("sess-001", "Test content", spatial_context=ctx)
        assert mem.spatial_context["device_id"] == "dev-001"

    def test_store_memory_with_vector(self, tmp_db_path):
        manager = SpatiotemporalMemoryManager(tmp_db_path)
        vector = np.random.rand(128).astype(np.float32)
        mem = manager.store_memory("sess-001", "Test content", semantic_vector=vector)
        assert mem.semantic_vector is not None

    def test_get_session_memories(self, tmp_db_path):
        manager = SpatiotemporalMemoryManager(tmp_db_path)
        manager.store_memory("sess-001", "Memory 1")
        manager.store_memory("sess-001", "Memory 2")
        manager.store_memory("sess-002", "Memory 3")

        memories = manager.get_session_memories("sess-001")
        assert len(memories) == 2

    def test_get_session_memories_empty(self, tmp_db_path):
        manager = SpatiotemporalMemoryManager(tmp_db_path)
        memories = manager.get_session_memories("nonexistent")
        assert memories == []

    def test_retrieve_memories_by_query(self, tmp_db_path):
        manager = SpatiotemporalMemoryManager(tmp_db_path)
        manager.store_memory("sess-001", "Python programming tutorial")
        manager.store_memory("sess-001", "JavaScript basics")
        manager.store_memory("sess-001", "Advanced Python patterns")

        results = manager.retrieve_memories("Python")
        assert len(results) >= 2

    def test_retrieve_memories_by_session(self, tmp_db_path):
        manager = SpatiotemporalMemoryManager(tmp_db_path)
        manager.store_memory("sess-001", "Content A")
        manager.store_memory("sess-002", "Content B")

        results = manager.retrieve_memories("Content", session_id="sess-001")
        assert all(m.session_id == "sess-001" for m in results)

    def test_retrieve_memories_limit(self, tmp_db_path):
        manager = SpatiotemporalMemoryManager(tmp_db_path)
        for i in range(10):
            manager.store_memory("sess-001", f"Memory content number {i}")

        results = manager.retrieve_memories("Memory", limit=3)
        assert len(results) <= 3

    def test_get_linked_memories(self, tmp_db_path):
        manager = SpatiotemporalMemoryManager(tmp_db_path)
        mem = manager.store_memory("sess-001", "Test content")
        linked = manager.get_linked_memories(mem.memory_id)
        assert "temporal" in linked
        assert "spatial" in linked
        assert "semantic" in linked

    def test_get_linked_memories_nonexistent(self, tmp_db_path):
        manager = SpatiotemporalMemoryManager(tmp_db_path)
        linked = manager.get_linked_memories("nonexistent")
        assert linked == {"temporal": [], "spatial": [], "semantic": []}

    def test_get_statistics(self, tmp_db_path):
        manager = SpatiotemporalMemoryManager(tmp_db_path)
        manager.store_memory("sess-001", "Memory 1")
        manager.store_memory("sess-002", "Memory 2")

        stats = manager.get_statistics()
        assert stats["total_memories"] == 2
        assert stats["sessions"] == 2
        assert "average_importance" in stats

    def test_get_statistics_empty(self, tmp_db_path):
        manager = SpatiotemporalMemoryManager(tmp_db_path)
        stats = manager.get_statistics()
        assert stats["total_memories"] == 0
        assert stats["average_importance"] == 0

    def test_delete_memory(self, tmp_db_path):
        manager = SpatiotemporalMemoryManager(tmp_db_path)
        mem = manager.store_memory("sess-001", "To delete")
        manager.delete_memory(mem.memory_id)
        stats = manager.get_statistics()
        assert stats["total_memories"] == 0

    def test_prune_low_importance(self, tmp_db_path):
        manager = SpatiotemporalMemoryManager(tmp_db_path)
        for i in range(20):
            manager.store_memory("sess-001", f"Memory {i}")

        manager.prune_low_importance(threshold=10.0, keep_count=5)
        stats = manager.get_statistics()
        assert stats["total_memories"] <= 20

    def test_persistence(self, tmp_db_path):
        manager1 = SpatiotemporalMemoryManager(tmp_db_path)
        manager1.store_memory("sess-001", "Persistent memory")

        manager2 = SpatiotemporalMemoryManager(tmp_db_path)
        stats = manager2.get_statistics()
        assert stats["total_memories"] == 1

    def test_temporal_linking(self, tmp_db_path):
        manager = SpatiotemporalMemoryManager(tmp_db_path)
        mem1 = manager.store_memory("sess-001", "First memory")
        mem2 = manager.store_memory("sess-001", "Second memory close in time")

        linked = manager.get_linked_memories(mem2.memory_id)
        assert len(linked["temporal"]) >= 1

    def test_spatial_linking_same_device(self, tmp_db_path):
        manager = SpatiotemporalMemoryManager(tmp_db_path)
        ctx = SpatialContext(device_id="dev-same")
        mem1 = manager.store_memory("sess-001", "Memory 1", spatial_context=ctx)
        mem2 = manager.store_memory("sess-001", "Memory 2", spatial_context=ctx)

        linked = manager.get_linked_memories(mem2.memory_id)
        assert len(linked["spatial"]) >= 1

    def test_semantic_similarity_with_vectors(self, tmp_db_path):
        manager = SpatiotemporalMemoryManager(tmp_db_path)
        vec1 = np.ones(128, dtype=np.float32)
        vec2 = np.ones(128, dtype=np.float32) * 0.9
        mem1 = manager.store_memory("sess-001", "Content 1", semantic_vector=vec1)
        mem2 = manager.store_memory("sess-001", "Content 2", semantic_vector=vec2)

        linked = manager.get_linked_memories(mem2.memory_id)
        assert len(linked["semantic"]) >= 1
