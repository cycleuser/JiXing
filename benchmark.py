#!/usr/bin/env python3
"""Benchmark script for JiXing memory system.
Runs actual tests and collects real performance data.
"""

import json
import os
import platform
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from jixing.memory import (
    SemanticCompressor,
    SpatialContext,
    SpatiotemporalMemoryManager,
    TemporalMemoryLinker,
    TimestampedMemory,
    _compute_content_hash,
)


def get_system_info():
    return {
        "platform": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "python_impl": platform.python_implementation(),
        "cpu_count": os.cpu_count(),
        "hostname": platform.node(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def benchmark_store(n=1000):
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        manager = SpatiotemporalMemoryManager(db_path)
        contents = [
            f"Test memory content number {i} about topic {'Python' if i % 2 == 0 else 'AI'}"
            for i in range(n)
        ]
        start = time.perf_counter()
        for i, content in enumerate(contents):
            manager.store_memory(f"sess-{i % 5:03d}", content)
        elapsed = time.perf_counter() - start
        return {
            "count": n,
            "total_seconds": round(elapsed, 4),
            "ops_per_second": round(n / elapsed, 1),
            "avg_ms_per_op": round(elapsed / n * 1000, 3),
        }


def benchmark_retrieve(n=1000, queries=20):
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        manager = SpatiotemporalMemoryManager(db_path)
        for i in range(n):
            manager.store_memory(
                f"sess-{i % 5:03d}", f"Memory content {i} about Python programming"
            )
        test_queries = ["Python", "content", "Memory", "programming", "AI"]
        start = time.perf_counter()
        for _ in range(queries):
            for q in test_queries:
                manager.retrieve_memories(q, limit=10)
        elapsed = time.perf_counter() - start
        total_queries = queries * len(test_queries)
        return {
            "total_queries": total_queries,
            "total_seconds": round(elapsed, 4),
            "queries_per_second": round(total_queries / elapsed, 1),
            "avg_ms_per_query": round(elapsed / total_queries * 1000, 3),
        }


def benchmark_linking(n=500):
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        manager = SpatiotemporalMemoryManager(db_path)
        for i in range(n):
            manager.store_memory(f"sess-{i % 3:03d}", f"Memory {i} about Python and AI")
        stats = manager.get_statistics()
        total_links = stats["total_links"]
        avg_links = total_links / n if n > 0 else 0
        return {
            "total_memories": n,
            "total_links": total_links,
            "avg_links_per_memory": round(avg_links, 2),
            "compressed": stats["compressed_memories"],
        }


def benchmark_compression():
    compressor = SemanticCompressor(compression_threshold=500)
    test_cases = [
        ("short", "Short content"),
        ("medium", " ".join(["word"] * 200)),
        ("long", " ".join(["word"] * 500)),
        ("very_long", " ".join(["word"] * 1000)),
    ]
    results = []
    for name, content in test_cases:
        mem = TimestampedMemory(
            memory_id=f"test-{name}",
            session_id="sess",
            timestamp="2024-01-01T00:00:00+00:00",
            content=content,
            content_hash=_compute_content_hash(content),
        )
        start = time.perf_counter()
        compressed = compressor.compress(mem)
        elapsed = time.perf_counter() - start
        importance = compressor.compute_importance(compressed)
        results.append(
            {
                "name": name,
                "original_length": len(content),
                "compressed_length": len(compressed.content),
                "ratio": round(len(content) / max(1, len(compressed.content)), 2),
                "compression_level": compressed.compression_level,
                "importance_score": round(importance, 3),
                "time_ms": round(elapsed * 1000, 3),
            }
        )
    return results


def benchmark_temporal_linking():
    linker = TemporalMemoryLinker(max_temporal_distance_seconds=3600)
    memories = []
    for i in range(100):
        ts = f"2024-01-01T{i // 60:02d}:{i % 60:02d}:00+00:00"
        memories.append(
            TimestampedMemory(
                memory_id=f"mem-{i:03d}",
                session_id="sess",
                timestamp=ts,
                content=f"Content {i}",
                content_hash=_compute_content_hash(f"Content {i}"),
            )
        )
    linked_count = 0
    total_pairs = 0
    for i, m1 in enumerate(memories):
        for j, m2 in enumerate(memories):
            if i >= j:
                continue
            total_pairs += 1
            if linker.are_temporally_linked(m1.timestamp, m2.timestamp):
                linked_count += 1
    return {
        "total_memories": len(memories),
        "total_pairs": total_pairs,
        "linked_pairs": linked_count,
        "link_rate": round(linked_count / total_pairs * 100, 2),
    }


def benchmark_retrieval_quality(n=200):
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        manager = SpatiotemporalMemoryManager(db_path)
        topics = ["Python", "JavaScript", "Machine Learning", "Deep Learning", "Data Science"]
        for i in range(n):
            topic = topics[i % len(topics)]
            manager.store_memory(f"sess-{i % 4:03d}", f"Memory about {topic} topic number {i}")
        recall_at_5 = 0
        recall_at_10 = 0
        total_queries = 0
        for topic in topics:
            results = manager.retrieve_memories(topic, limit=20)
            relevant = [m for m in results if topic.lower() in m.content.lower()]
            total_found = len(
                [m for m in manager._memories.values() if topic.lower() in m.content.lower()]
            )
            if total_found > 0:
                recall_at_5 += len(relevant[:5]) / min(5, total_found)
                recall_at_10 += len(relevant[:10]) / min(10, total_found)
                total_queries += 1
        return {
            "total_memories": n,
            "topics": len(topics),
            "recall_at_5": round(recall_at_5 / total_queries * 100, 1) if total_queries > 0 else 0,
            "recall_at_10": round(recall_at_10 / total_queries * 100, 1)
            if total_queries > 0
            else 0,
        }


def main():
    print("=" * 60)
    print("JiXing Memory System Benchmark")
    print("=" * 60)

    sys_info = get_system_info()
    print(f"\nSystem: {sys_info['platform']} {sys_info['release']}")
    print(f"Machine: {sys_info['machine']}")
    print(f"Processor: {sys_info['processor']}")
    print(f"CPU cores: {sys_info['cpu_count']}")
    print(f"Python: {sys_info['python_version']} ({sys_info['python_impl']})")
    print(f"Timestamp: {sys_info['timestamp']}")

    print("\n--- Store Benchmark (1000 memories) ---")
    store_result = benchmark_store(1000)
    print(f"  Total time: {store_result['total_seconds']}s")
    print(f"  Ops/sec: {store_result['ops_per_second']}")
    print(f"  Avg ms/op: {store_result['avg_ms_per_op']}")

    print("\n--- Retrieve Benchmark (1000 memories, 100 queries) ---")
    retrieve_result = benchmark_retrieve(1000, 20)
    print(f"  Total queries: {retrieve_result['total_queries']}")
    print(f"  Queries/sec: {retrieve_result['queries_per_second']}")
    print(f"  Avg ms/query: {retrieve_result['avg_ms_per_query']}")

    print("\n--- Linking Benchmark (500 memories) ---")
    linking_result = benchmark_linking(500)
    print(f"  Total links: {linking_result['total_links']}")
    print(f"  Avg links/memory: {linking_result['avg_links_per_memory']}")
    print(f"  Compressed: {linking_result['compressed']}")

    print("\n--- Temporal Linking Analysis (100 memories) ---")
    temporal_result = benchmark_temporal_linking()
    print(f"  Linked pairs: {temporal_result['linked_pairs']}/{temporal_result['total_pairs']}")
    print(f"  Link rate: {temporal_result['link_rate']}%")

    print("\n--- Compression Benchmark ---")
    compression_results = benchmark_compression()
    for r in compression_results:
        print(
            f"  {r['name']}: {r['original_length']} -> {r['compressed_length']} chars (ratio: {r['ratio']}x, level: {r['compression_level']})"
        )

    print("\n--- Retrieval Quality Benchmark (200 memories, 5 topics) ---")
    quality_result = benchmark_retrieval_quality(200)
    print(f"  Recall@5: {quality_result['recall_at_5']}%")
    print(f"  Recall@10: {quality_result['recall_at_10']}%")

    results = {
        "system_info": sys_info,
        "store": store_result,
        "retrieve": retrieve_result,
        "linking": linking_result,
        "temporal_linking": temporal_result,
        "compression": compression_results,
        "retrieval_quality": quality_result,
    }

    output_path = Path(__file__).parent / "tests" / "data" / "benchmark_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
