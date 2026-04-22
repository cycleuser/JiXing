"""Context window management for JiXing agent.

Handles context overflow detection, smart compression with controllable ratios,
session migration, and runtime state persistence. Implements strategies inspired
by VM live migration with graceful degradation.

Key features:
- Multi-level compression with precise ratio control
- Structured conversation archiving (JSONL)
- Session migration with context preservation
- Runtime state persistence (MD files)
- Quality-aware compression that preserves critical information
"""

import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class ContextSnapshot:
    """A snapshot of the current context state for migration."""

    snapshot_id: str
    source_session_id: str
    timestamp: str
    original_goal: str
    intermediate_requirements: list[str]
    last_user_message: str
    last_assistant_message: str
    full_message_count: int
    compressed_summary: str
    compression_level: int
    migration_metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ContextSnapshot":
        return cls(**data)


@dataclass
class RuntimeState:
    """Runtime state persisted to MD files when at max context."""

    state_id: str
    session_id: str
    timestamp: str
    original_goal: str
    intermediate_requirements: list[str]
    decisions_made: list[str]
    current_task: str
    last_round_input: str
    last_round_output: str
    pending_actions: list[str]
    context_usage: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_markdown(self) -> str:
        lines = [
            f"# Runtime State - {self.timestamp}",
            "",
            f"**Session ID**: `{self.session_id}`",
            f"**State ID**: `{self.state_id}`",
            "",
            "## Original Goal",
            "",
            self.original_goal,
            "",
            "## Intermediate Requirements",
            "",
        ]
        for i, req in enumerate(self.intermediate_requirements, 1):
            lines.append(f"{i}. {req}")
        lines.extend([
            "",
            "## Decisions Made",
            "",
        ])
        for i, decision in enumerate(self.decisions_made, 1):
            lines.append(f"{i}. {decision}")
        lines.extend([
            "",
            "## Current Task",
            "",
            self.current_task,
            "",
            "## Last Round Input",
            "",
            "```",
            self.last_round_input,
            "```",
            "",
            "## Last Round Output",
            "",
        ])
        if self.last_round_output:
            lines.extend([
                "```",
                self.last_round_output,
                "```",
            ])
        else:
            lines.append("*No output yet*")
        if self.pending_actions:
            lines.extend([
                "",
                "## Pending Actions",
                "",
            ])
            for i, action in enumerate(self.pending_actions, 1):
                lines.append(f"{i}. {action}")
        if self.context_usage:
            lines.extend([
                "",
                "## Context Usage",
                "",
                f"- Tokens used: {self.context_usage.get('tokens_used', 'N/A')}",
                f"- Context limit: {self.context_usage.get('context_limit', 'N/A')}",
                f"- Usage percentage: {self.context_usage.get('usage_percentage', 'N/A')}%",
            ])
        lines.append("")
        return "\n".join(lines)

    @classmethod
    def from_markdown(cls, content: str) -> "RuntimeState":
        state = cls(
            state_id="",
            session_id="",
            timestamp="",
            original_goal="",
            intermediate_requirements=[],
            decisions_made=[],
            current_task="",
            last_round_input="",
            last_round_output="",
            pending_actions=[],
        )
        lines = content.split("\n")
        current_section = None
        list_buffer = []
        code_block = None

        for line in lines:
            if line.startswith("# Runtime State"):
                ts = line.split(" - ", 1)[1] if " - " in line else ""
                state.timestamp = ts
            elif line.startswith("**Session ID**"):
                state.session_id = line.split("`")[1] if "`" in line else ""
            elif line.startswith("**State ID**"):
                state.state_id = line.split("`")[1] if "`" in line else ""
            elif line.startswith("## "):
                if current_section and list_buffer:
                    cls._flush_list(state, current_section, list_buffer)
                    list_buffer = []
                current_section = line[3:].strip()
            elif line.startswith("```"):
                if code_block is None:
                    code_block = []
                else:
                    code_content = "\n".join(code_block)
                    if current_section == "Last Round Input":
                        state.last_round_input = code_content
                    elif current_section == "Last Round Output":
                        state.last_round_output = code_content
                    code_block = None
            elif code_block is not None:
                code_block.append(line)
            elif line.startswith("- Tokens used:"):
                state.context_usage["tokens_used"] = line.split(": ", 1)[1] if ": " in line else ""
            elif line.startswith("- Context limit:"):
                state.context_usage["context_limit"] = line.split(": ", 1)[1] if ": " in line else ""
            elif line.startswith("- Usage percentage:"):
                pct = line.split(": ", 1)[1].rstrip("%") if ": " in line else ""
                state.context_usage["usage_percentage"] = pct
            elif line.startswith("*") and line.endswith("*"):
                continue
            elif line.strip() and current_section:
                if line[0].isdigit() and ". " in line:
                    list_buffer.append(line.split(". ", 1)[1])
                elif not line[0].isdigit() and not line.startswith("-"):
                    if current_section in ("Original Goal", "Current Task"):
                        attr = cls._section_to_attr(current_section)
                        if attr:
                            existing = getattr(state, attr, "")
                            if existing:
                                setattr(state, attr, existing + "\n" + line)
                            else:
                                setattr(state, attr, line)

        if list_buffer:
            cls._flush_list(state, current_section, list_buffer)

        return state

    @staticmethod
    def _flush_list(state: "RuntimeState", section: str, items: list[str]):
        attr_map = {
            "Intermediate Requirements": "intermediate_requirements",
            "Decisions Made": "decisions_made",
            "Pending Actions": "pending_actions",
        }
        attr = attr_map.get(section)
        if attr:
            setattr(state, attr, items)

    @staticmethod
    def _section_to_attr(section: str) -> str:
        mapping = {
            "Original Goal": "original_goal",
            "Current Task": "current_task",
            "Last Round Input": "last_round_input",
            "Last Round Output": "last_round_output",
        }
        return mapping.get(section, "")


@dataclass
class CompressionResult:
    """Result of context compression operation."""

    success: bool
    original_token_count: int
    compressed_token_count: int
    compression_ratio: float
    target_ratio: float
    strategy_used: str
    compressed_messages: list[dict]
    summary: str
    archived_messages: list[dict] = field(default_factory=list)
    quality_score: float = 1.0
    metadata: dict = field(default_factory=dict)


@dataclass
class CompressionConfig:
    """Configuration for compression behavior."""

    target_ratio: float = 0.5
    min_ratio: float = 0.1
    max_ratio: float = 0.9
    preserve_first_user: bool = True
    preserve_last_n: int = 4
    preserve_code_blocks: bool = True
    preserve_user_messages: bool = True
    min_message_length: int = 50
    compression_levels: int = 3
    quality_threshold: float = 0.6

    def validate(self) -> bool:
        return (
            self.min_ratio <= self.target_ratio <= self.max_ratio
            and self.preserve_last_n > 0
            and self.compression_levels > 0
        )


class MessageAnalyzer:
    """Analyzes messages to determine importance and compression strategy."""

    @staticmethod
    def estimate_tokens(content: str) -> int:
        """Estimate token count for content."""
        if not content:
            return 0
        chinese_chars = sum(1 for c in content if '\u4e00' <= c <= '\u9fff')
        other_chars = len(content) - chinese_chars
        return int(other_chars / 4 + chinese_chars / 1.5)

    @staticmethod
    def has_code_block(content: str) -> bool:
        return "```" in content

    @staticmethod
    def has_structured_data(content: str) -> bool:
        indicators = ["{", "}", "[", "]", ":", ",", "def ", "class ", "import "]
        return any(ind in content for ind in indicators)

    @staticmethod
    def is_goal_statement(content: str) -> bool:
        goal_indicators = [
            "goal", "objective", "task", "implement", "create", "build",
            "write", "design", "develop", "fix", "add", "实现", "创建", "完成",
        ]
        lower = content.lower()
        return any(ind in lower for ind in goal_indicators) and len(content) < 500

    @staticmethod
    def is_requirement(content: str) -> bool:
        req_indicators = [
            "require", "must", "should", "need", "constraint", "specification",
            "要求", "必须", "应该", "需要", "约束",
        ]
        lower = content.lower()
        return any(ind in lower for ind in req_indicators)

    @staticmethod
    def compute_importance(
        msg: dict,
        index: int,
        total: int,
        config: CompressionConfig,
    ) -> float:
        """Compute importance score for a message (0-10)."""
        score = 0.0
        content = msg.get("content", "")
        role = msg.get("role", "")
        tokens = MessageAnalyzer.estimate_tokens(content)

        if role == "user":
            score += 3.0
        if role == "system":
            score += 4.0

        if index == 0:
            score += 3.0
        if index >= total - config.preserve_last_n:
            score += 2.5

        if MessageAnalyzer.has_code_block(content):
            score += 1.5
        if MessageAnalyzer.has_structured_data(content):
            score += 1.0
        if MessageAnalyzer.is_goal_statement(content):
            score += 2.0
        if MessageAnalyzer.is_requirement(content):
            score += 1.5

        if tokens > 500:
            score += 0.5
        if tokens > 2000:
            score += 0.5

        if config.preserve_user_messages and role == "user":
            score += 1.0

        return min(10.0, score)


class ContextWindowManager:
    """Manages context window limits, compression, and session migration.

    Supports:
    - Multi-level compression with precise ratio control
    - Quality-aware message preservation
    - JSONL-based conversation archiving
    - Session migration (VM live migration style)
    - Runtime state persistence to MD files
    """

    def __init__(
        self,
        context_limit: int = 128000,
        compression_threshold: float = 0.8,
        config: Optional[CompressionConfig] = None,
        jsonl_dir: Optional[str | Path] = None,
        state_dir: Optional[str | Path] = None,
    ):
        self.context_limit = context_limit
        self.compression_threshold = compression_threshold
        self.config = config or CompressionConfig()
        self.jsonl_dir = Path(jsonl_dir) if jsonl_dir else Path.home() / ".jixing" / "context_archive"
        self.state_dir = Path(state_dir) if state_dir else Path.home() / ".jixing" / "runtime_states"
        self.jsonl_dir.mkdir(parents=True, exist_ok=True)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._compression_hooks: list[Callable] = []
        self._migration_history: list[dict] = []

    def set_compression_ratio(self, ratio: float) -> bool:
        """Set target compression ratio. Returns False if ratio is invalid."""
        if self.config.min_ratio <= ratio <= self.config.max_ratio:
            self.config.target_ratio = ratio
            return True
        return False

    def estimate_token_count(self, messages: list[dict]) -> int:
        """Estimate token count for a list of messages."""
        total = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        total += MessageAnalyzer.estimate_tokens(item.get("text", ""))
            else:
                total += MessageAnalyzer.estimate_tokens(str(content))
        return total

    def needs_compression(self, messages: list[dict]) -> bool:
        """Check if current context needs compression."""
        token_count = self.estimate_token_count(messages)
        return token_count > self.context_limit * self.compression_threshold

    def get_context_usage(self, messages: list[dict]) -> dict[str, Any]:
        """Get current context usage statistics."""
        token_count = self.estimate_token_count(messages)
        return {
            "tokens_used": token_count,
            "context_limit": self.context_limit,
            "usage_percentage": round(token_count / self.context_limit * 100, 2) if self.context_limit > 0 else 0,
            "remaining_tokens": max(0, self.context_limit - token_count),
            "message_count": len(messages),
            "needs_compression": self.needs_compression(messages),
            "compression_threshold": self.compression_threshold,
        }

    def compress_context(
        self,
        messages: list[dict],
        target_ratio: Optional[float] = None,
        goal: str = "",
        compressor_fn: Optional[Callable] = None,
        config: Optional[CompressionConfig] = None,
    ) -> CompressionResult:
        """Compress context with controllable ratio.

        Args:
            messages: Original messages to compress
            target_ratio: Target compression ratio (0.1-0.9). Lower = more compression.
            goal: Original goal to preserve
            compressor_fn: Optional AI-powered compression function
            config: Override compression config

        Returns:
            CompressionResult with compressed messages and metadata
        """
        if not messages:
            return CompressionResult(
                success=False,
                original_token_count=0,
                compressed_token_count=0,
                compression_ratio=1.0,
                target_ratio=target_ratio or self.config.target_ratio,
                strategy_used="none",
                compressed_messages=[],
                summary="No messages to compress.",
            )

        cfg = config or self.config
        ratio = target_ratio if target_ratio is not None else cfg.target_ratio
        ratio = max(cfg.min_ratio, min(cfg.max_ratio, ratio))

        current_tokens = self.estimate_token_count(messages)
        target_tokens = int(current_tokens * ratio)

        if current_tokens <= target_tokens:
            return CompressionResult(
                success=True,
                original_token_count=current_tokens,
                compressed_token_count=current_tokens,
                compression_ratio=1.0,
                target_ratio=ratio,
                strategy_used="none",
                compressed_messages=messages,
                summary="No compression needed.",
                quality_score=1.0,
            )

        scored_messages = []
        for i, msg in enumerate(messages):
            importance = MessageAnalyzer.compute_importance(msg, i, len(messages), cfg)
            scored_messages.append((importance, msg))

        strategies = [
            ("preserve_critical", self._preserve_critical),
            ("smart", self._smart_compress),
            ("progressive", self._progressive_compress),
            ("summary", self._summary_compress),
        ]

        best_result = None
        for strategy_name, strategy_fn in strategies:
            try:
                result = strategy_fn(scored_messages, messages, goal, target_tokens, ratio, cfg, compressor_fn)
                if result and result.compressed_token_count <= target_tokens * 1.2:
                    if best_result is None or result.quality_score > best_result.quality_score:
                        best_result = result
                        best_result.strategy_used = strategy_name
                        if result.compressed_token_count <= target_tokens:
                            break
            except Exception as e:
                logger.warning(f"Strategy {strategy_name} failed: {e}")

        if best_result is None:
            best_result = self._fallback_compress(messages, goal, target_tokens, ratio)

        self.trigger_compression_hooks(best_result)
        return best_result

    def _preserve_critical(
        self,
        scored_messages: list,
        messages: list[dict],
        goal: str,
        target_tokens: int,
        target_ratio: float,
        config: CompressionConfig,
        compressor_fn: Optional[Callable],
    ) -> Optional[CompressionResult]:
        """Strategy: Only keep critical messages, minimal compression."""
        critical = []
        current_tokens = 0

        sorted_msgs = sorted(scored_messages, key=lambda x: x[0], reverse=True)

        for score, msg in sorted_msgs:
            if score >= 7.0:
                msg_tokens = MessageAnalyzer.estimate_tokens(str(msg.get("content", "")))
                if current_tokens + msg_tokens <= target_tokens:
                    critical.append(msg)
                    current_tokens += msg_tokens

        if not critical:
            return None

        critical.sort(key=lambda m: messages.index(m) if m in messages else 0)

        summary = self._generate_summary_from_messages(
            [m for _, m in scored_messages if m not in critical],
            compressor_fn,
        )

        if goal and critical:
            first_user = next((m for m in critical if m.get("role") == "user"), None)
            if first_user:
                idx = critical.index(first_user)
                critical[idx] = {
                    **first_user,
                    "content": f"[Original Goal]\n{goal}\n\n[Compressed Context]\n{summary}",
                }

        quality = len(critical) / len(messages) if messages else 0
        quality = min(1.0, quality * 2)

        return CompressionResult(
            success=True,
            original_token_count=self.estimate_token_count(messages),
            compressed_token_count=current_tokens,
            compression_ratio=current_tokens / self.estimate_token_count(messages) if self.estimate_token_count(messages) > 0 else 1,
            target_ratio=target_ratio,
            strategy_used="preserve_critical",
            compressed_messages=critical,
            summary=summary,
            archived_messages=[m for _, m in scored_messages if m not in critical],
            quality_score=quality,
        )

    def _smart_compress(
        self,
        scored_messages: list,
        messages: list[dict],
        goal: str,
        target_tokens: int,
        target_ratio: float,
        config: CompressionConfig,
        compressor_fn: Optional[Callable],
    ) -> Optional[CompressionResult]:
        """Strategy: Keep first user, last N, compress middle intelligently."""
        total = len(messages)
        preserved_indices = set()

        if config.preserve_first_user:
            for i, msg in enumerate(messages):
                if msg.get("role") == "user":
                    preserved_indices.add(i)
                    break

        for i in range(max(0, total - config.preserve_last_n), total):
            preserved_indices.add(i)

        preserved_msgs = [messages[i] for i in sorted(preserved_indices)]
        preserved_tokens = self.estimate_token_count(preserved_msgs)

        middle_msgs = [m for i, m in enumerate(messages) if i not in preserved_indices]
        middle_tokens = self.estimate_token_count(middle_msgs)

        remaining_budget = max(0, target_tokens - preserved_tokens)

        if middle_tokens > 0 and remaining_budget > 0:
            middle_summary = self._generate_summary_from_messages(middle_msgs, compressor_fn)
            summary_tokens = MessageAnalyzer.estimate_tokens(middle_summary)

            if summary_tokens <= remaining_budget:
                compressed = preserved_msgs.copy()
                if goal:
                    first_user = next((m for m in preserved_msgs if m.get("role") == "user"), None)
                    if first_user:
                        idx = preserved_msgs.index(first_user)
                        compressed[idx] = {
                            **first_user,
                            "content": f"[Original Goal]\n{goal}\n\n[Context Summary]\n{middle_summary}",
                        }
                else:
                    compressed.insert(0, {
                        "role": "system",
                        "content": f"[Context Summary]\n{middle_summary}",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    })

                compressed_tokens = self.estimate_token_count(compressed)
                quality = 0.8 if compressed_tokens <= target_tokens else 0.6

                return CompressionResult(
                    success=True,
                    original_token_count=self.estimate_token_count(messages),
                    compressed_token_count=compressed_tokens,
                    compression_ratio=compressed_tokens / self.estimate_token_count(messages) if self.estimate_token_count(messages) > 0 else 1,
                    target_ratio=target_ratio,
                    strategy_used="smart",
                    compressed_messages=compressed,
                    summary=middle_summary,
                    archived_messages=middle_msgs,
                    quality_score=quality,
                )

        return None

    def _progressive_compress(
        self,
        scored_messages: list,
        messages: list[dict],
        goal: str,
        target_tokens: int,
        target_ratio: float,
        config: CompressionConfig,
        compressor_fn: Optional[Callable],
    ) -> Optional[CompressionResult]:
        """Strategy: Progressive compression by importance score."""
        sorted_msgs = sorted(scored_messages, key=lambda x: x[0], reverse=True)

        compressed = []
        current_tokens = 0
        archived = []

        for score, msg in sorted_msgs:
            msg_tokens = MessageAnalyzer.estimate_tokens(str(msg.get("content", "")))

            if current_tokens + msg_tokens <= target_tokens:
                compressed.append(msg)
                current_tokens += msg_tokens
            else:
                archived.append(msg)

        compressed.sort(key=lambda m: messages.index(m) if m in messages else 0)

        if archived:
            summary = self._generate_summary_from_messages(archived, compressor_fn)
            if goal:
                first_user = next((m for m in compressed if m.get("role") == "user"), None)
                if first_user:
                    idx = compressed.index(first_user)
                    compressed[idx] = {
                        **first_user,
                        "content": f"[Original Goal]\n{goal}\n\n[Compressed History]\n{summary}",
                    }

        compressed_tokens = self.estimate_token_count(compressed)
        quality = min(1.0, len(compressed) / len(messages) * 1.5) if messages else 0

        return CompressionResult(
            success=True,
            original_token_count=self.estimate_token_count(messages),
            compressed_token_count=compressed_tokens,
            compression_ratio=compressed_tokens / self.estimate_token_count(messages) if self.estimate_token_count(messages) > 0 else 1,
            target_ratio=target_ratio,
            strategy_used="progressive",
            compressed_messages=compressed,
            summary=self._generate_summary_from_messages(archived, compressor_fn) if archived else "",
            archived_messages=archived,
            quality_score=quality,
        )

    def _summary_compress(
        self,
        scored_messages: list,
        messages: list[dict],
        goal: str,
        target_tokens: int,
        target_ratio: float,
        config: CompressionConfig,
        compressor_fn: Optional[Callable],
    ) -> Optional[CompressionResult]:
        """Strategy: Full summary compression."""
        summary = self._generate_summary_from_messages(messages, compressor_fn)

        content_parts = []
        if goal:
            content_parts.append(f"[Original Goal]\n{goal}")
        content_parts.append(f"[Full Conversation Summary]\n{summary}")

        compressed = [{
            "role": "user",
            "content": "\n\n".join(content_parts),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": {"type": "full_summary"},
        }]

        compressed_tokens = self.estimate_token_count(compressed)

        return CompressionResult(
            success=True,
            original_token_count=self.estimate_token_count(messages),
            compressed_token_count=compressed_tokens,
            compression_ratio=compressed_tokens / self.estimate_token_count(messages) if self.estimate_token_count(messages) > 0 else 1,
            target_ratio=target_ratio,
            strategy_used="summary",
            compressed_messages=compressed,
            summary=summary,
            archived_messages=messages,
            quality_score=0.5,
        )

    def _fallback_compress(
        self,
        messages: list[dict],
        goal: str,
        target_tokens: int,
        target_ratio: float,
    ) -> CompressionResult:
        """Fallback: Truncate to fit, preserving first and last messages."""
        if not messages:
            return CompressionResult(
                success=False,
                original_token_count=0,
                compressed_token_count=0,
                compression_ratio=1.0,
                target_ratio=target_ratio,
                strategy_used="fallback",
                compressed_messages=[],
                summary="No messages.",
            )

        first = messages[0]
        last = messages[-1]
        middle = messages[1:-1] if len(messages) > 2 else []

        first_tokens = MessageAnalyzer.estimate_tokens(str(first.get("content", "")))
        last_tokens = MessageAnalyzer.estimate_tokens(str(last.get("content", "")))

        remaining = max(0, target_tokens - first_tokens - last_tokens)

        kept_middle = []
        current = 0
        for msg in reversed(middle):
            msg_tokens = MessageAnalyzer.estimate_tokens(str(msg.get("content", "")))
            if current + msg_tokens <= remaining:
                kept_middle.append(msg)
                current += msg_tokens
            else:
                break
        kept_middle.reverse()

        compressed = [first] + kept_middle + [last]

        if goal:
            compressed[0] = {
                **first,
                "content": f"[Goal]\n{goal}\n\n{first.get('content', '')}",
            }

        archived = [m for m in middle if m not in kept_middle]
        compressed_tokens = self.estimate_token_count(compressed)

        return CompressionResult(
            success=True,
            original_token_count=self.estimate_token_count(messages),
            compressed_token_count=compressed_tokens,
            compression_ratio=compressed_tokens / self.estimate_token_count(messages) if self.estimate_token_count(messages) > 0 else 1,
            target_ratio=target_ratio,
            strategy_used="fallback",
            compressed_messages=compressed,
            summary=f"Truncated {len(archived)} middle messages to fit token budget.",
            archived_messages=archived,
            quality_score=0.4,
        )

    def _generate_summary_from_messages(
        self,
        messages: list[dict],
        compressor_fn: Optional[Callable],
    ) -> str:
        """Generate summary from messages, using AI or fallback."""
        if not messages:
            return "No conversation history."

        if compressor_fn:
            try:
                content = "\n".join(
                    f"[{m.get('role', 'unknown')}] {m.get('content', '')}"
                    for m in messages
                )
                return compressor_fn(content)
            except Exception as e:
                logger.warning(f"AI compression failed: {e}")

        return self._fallback_summary(messages)

    def _fallback_summary(self, messages: list[dict]) -> str:
        """Generate fallback summary when AI compression unavailable."""
        if not messages:
            return "No conversation history."

        user_msgs = [m for m in messages if m.get("role") == "user"]
        assistant_msgs = [m for m in messages if m.get("role") == "assistant"]
        code_msgs = [m for m in messages if MessageAnalyzer.has_code_block(m.get("content", ""))]

        parts = [
            f"Conversation: {len(messages)} messages total ({len(user_msgs)} user, {len(assistant_msgs)} assistant).",
        ]

        if user_msgs:
            first_content = user_msgs[0].get("content", "")[:300]
            parts.append(f"First request: {first_content}...")
            last_content = user_msgs[-1].get("content", "")[:300]
            parts.append(f"Last request: {last_content}...")

        if code_msgs:
            parts.append(f"Code blocks discussed: {len(code_msgs)} messages.")

        key_topics = self._extract_key_topics(messages)
        if key_topics:
            parts.append(f"Key topics: {', '.join(key_topics)}")

        return "\n".join(parts)

    def _extract_key_topics(self, messages: list[dict], max_topics: int = 5) -> list[str]:
        """Extract key topics from messages using keyword frequency."""
        keywords = {}
        for msg in messages:
            content = msg.get("content", "").lower()
            words = content.split()
            for word in words:
                if len(word) > 4 and word.isalpha():
                    keywords[word] = keywords.get(word, 0) + 1

        sorted_keywords = sorted(keywords.items(), key=lambda x: x[1], reverse=True)
        return [k for k, _ in sorted_keywords[:max_topics]]

    def archive_messages_to_jsonl(
        self,
        session_id: str,
        messages: list[dict],
        metadata: Optional[dict] = None,
    ) -> Path:
        """Archive messages to a JSONL file for later retrieval."""
        jsonl_path = self.jsonl_dir / f"{session_id}.jsonl"

        with open(jsonl_path, "a", encoding="utf-8") as f:
            for msg in messages:
                record = {
                    "session_id": session_id,
                    "archived_at": datetime.now(timezone.utc).isoformat(),
                    "metadata": metadata or {},
                    **msg,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.info(f"Archived {len(messages)} messages to {jsonl_path}")
        return jsonl_path

    def load_archived_messages(
        self,
        session_id: str,
        limit: Optional[int] = None,
    ) -> list[dict]:
        """Load archived messages from JSONL file."""
        jsonl_path = self.jsonl_dir / f"{session_id}.jsonl"
        if not jsonl_path.exists():
            return []

        messages = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    messages.append({
                        "role": record.get("role", ""),
                        "content": record.get("content", ""),
                        "timestamp": record.get("timestamp", ""),
                        "metrics": record.get("metrics"),
                    })
                if limit and len(messages) >= limit:
                    break

        return messages

    def get_archived_sessions(self) -> list[str]:
        """List all session IDs that have archived messages."""
        return [p.stem for p in self.jsonl_dir.glob("*.jsonl")]

    def create_snapshot(
        self,
        session_id: str,
        messages: list[dict],
        goal: str = "",
        intermediate_requirements: Optional[list[str]] = None,
        compression_level: int = 1,
    ) -> ContextSnapshot:
        """Create a context snapshot for session migration."""
        user_msgs = [m for m in messages if m.get("role") == "user"]
        assistant_msgs = [m for m in messages if m.get("role") == "assistant"]

        last_user = user_msgs[-1].get("content", "") if user_msgs else ""
        last_assistant = assistant_msgs[-1].get("content", "") if assistant_msgs else ""

        summary_result = self.compress_context(
            messages,
            target_ratio=0.2,
            goal=goal,
        )

        return ContextSnapshot(
            snapshot_id=str(uuid.uuid4()),
            source_session_id=session_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            original_goal=goal,
            intermediate_requirements=intermediate_requirements or [],
            last_user_message=last_user,
            last_assistant_message=last_assistant,
            full_message_count=len(messages),
            compressed_summary=summary_result.summary,
            compression_level=compression_level,
            migration_metadata={
                "original_token_count": summary_result.original_token_count,
                "compressed_token_count": summary_result.compressed_token_count,
                "compression_ratio": summary_result.compression_ratio,
            },
        )

    def migrate_session(
        self,
        snapshot: ContextSnapshot,
        new_session_creator: Callable,
        context_limit: Optional[int] = None,
    ) -> dict[str, Any]:
        """Migrate context to a new session (VM live migration style)."""
        old_limit = self.context_limit
        if context_limit:
            self.context_limit = context_limit

        new_session = new_session_creator()

        migration_messages = []

        if snapshot.original_goal:
            migration_messages.append({
                "role": "system",
                "content": (
                    f"[MIGRATED SESSION]\n"
                    f"This conversation was migrated from session `{snapshot.source_session_id}`.\n"
                    f"Original goal: {snapshot.original_goal}\n"
                    f"Migration timestamp: {snapshot.timestamp}\n"
                    f"Original messages: {snapshot.full_message_count}\n"
                ),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "metadata": {"type": "migration_header"},
            })

        if snapshot.compressed_summary:
            migration_messages.append({
                "role": "user",
                "content": f"[Conversation History Summary]\n{snapshot.compressed_summary}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "metadata": {"type": "migrated_summary"},
            })

        if snapshot.intermediate_requirements:
            reqs = "\n".join(f"- {r}" for r in snapshot.intermediate_requirements)
            migration_messages.append({
                "role": "user",
                "content": f"[Intermediate Requirements from Previous Session]\n{reqs}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "metadata": {"type": "migrated_requirements"},
            })

        if snapshot.last_user_message:
            migration_messages.append({
                "role": "user",
                "content": f"[Last User Message Before Migration]\n{snapshot.last_user_message}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "metadata": {"type": "migrated_last_user"},
            })

        if snapshot.last_assistant_message:
            migration_messages.append({
                "role": "assistant",
                "content": f"[Last Assistant Response Before Migration]\n{snapshot.last_assistant_message}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "metadata": {"type": "migrated_last_assistant"},
            })

        migration_record = {
            "snapshot_id": snapshot.snapshot_id,
            "source_session_id": snapshot.source_session_id,
            "target_session_id": getattr(new_session, "id", "unknown"),
            "timestamp": snapshot.timestamp,
            "original_messages": snapshot.full_message_count,
            "migrated_messages": len(migration_messages),
            "context_limit_before": old_limit,
            "context_limit_after": self.context_limit,
            "compression_level": snapshot.compression_level,
        }
        self._migration_history.append(migration_record)

        self.context_limit = old_limit

        return {
            "success": True,
            "new_session": new_session,
            "migration_messages": migration_messages,
            "migration_record": migration_record,
        }

    def persist_runtime_state(
        self,
        session_id: str,
        goal: str,
        intermediate_requirements: list[str],
        decisions_made: list[str],
        current_task: str,
        last_round_input: str,
        last_round_output: str = "",
        pending_actions: Optional[list[str]] = None,
        context_usage: Optional[dict] = None,
    ) -> Path:
        """Persist runtime state to an MD file when at max context."""
        state = RuntimeState(
            state_id=str(uuid.uuid4()),
            session_id=session_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            original_goal=goal,
            intermediate_requirements=intermediate_requirements,
            decisions_made=decisions_made,
            current_task=current_task,
            last_round_input=last_round_input,
            last_round_output=last_round_output,
            pending_actions=pending_actions or [],
            context_usage=context_usage or {},
        )

        md_path = self.state_dir / f"{session_id}_{state.state_id[:8]}.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(state.to_markdown())

        json_path = self.state_dir / f"{session_id}_{state.state_id[:8]}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(state.to_dict(), f, ensure_ascii=False, indent=2)

        logger.info(f"Persisted runtime state to {md_path}")
        return md_path

    def load_runtime_state(self, state_path: str | Path) -> RuntimeState:
        """Load runtime state from an MD file."""
        path = Path(state_path)
        if not path.exists():
            raise FileNotFoundError(f"Runtime state file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        return RuntimeState.from_markdown(content)

    def list_runtime_states(self, session_id: Optional[str] = None) -> list[Path]:
        """List all persisted runtime states."""
        pattern = f"{session_id}_*.md" if session_id else "*.md"
        return sorted(self.state_dir.glob(pattern))

    def build_migration_prompt(
        self,
        snapshot: ContextSnapshot,
        new_goal: str = "",
    ) -> str:
        """Build a prompt for continuing in a new session."""
        parts = [
            "# Session Migration Context",
            "",
            f"This conversation continues from session `{snapshot.source_session_id}`.",
            f"Original conversation had {snapshot.full_message_count} messages.",
            "",
        ]

        if snapshot.original_goal:
            parts.extend([
                "## Original Goal",
                "",
                snapshot.original_goal,
                "",
            ])

        if snapshot.intermediate_requirements:
            parts.extend([
                "## Requirements Established",
                "",
            ])
            for i, req in enumerate(snapshot.intermediate_requirements, 1):
                parts.append(f"{i}. {req}")
            parts.append("")

        if snapshot.compressed_summary:
            parts.extend([
                "## Conversation Summary",
                "",
                snapshot.compressed_summary,
                "",
            ])

        if snapshot.last_user_message:
            parts.extend([
                "## Last User Input",
                "",
                snapshot.last_user_message,
                "",
            ])

        if snapshot.last_assistant_message:
            parts.extend([
                "## Last Assistant Response",
                "",
                snapshot.last_assistant_message,
                "",
            ])

        if new_goal:
            parts.extend([
                "## New Instructions",
                "",
                new_goal,
                "",
            ])

        return "\n".join(parts)

    def register_compression_hook(self, hook: Callable):
        """Register a hook to be called when compression is triggered."""
        self._compression_hooks.append(hook)

    def trigger_compression_hooks(self, result: CompressionResult):
        """Trigger all registered compression hooks."""
        for hook in self._compression_hooks:
            try:
                hook(result)
            except Exception as e:
                logger.warning(f"Compression hook failed: {e}")

    def get_migration_history(self) -> list[dict]:
        """Get history of all session migrations."""
        return self._migration_history

    def auto_handle_overflow(
        self,
        session_id: str,
        messages: list[dict],
        goal: str = "",
        compressor_fn: Optional[Callable] = None,
        session_creator: Optional[Callable] = None,
        longer_context_limit: Optional[int] = None,
    ) -> dict[str, Any]:
        """Automatically handle context overflow with graceful degradation.

        Flow:
        1. Try smart compression with configurable ratio
        2. If still over limit, try stronger compression
        3. If at max model context, persist runtime state to MD
        4. If session_creator provided, migrate to new session
        5. Archive full history to JSONL

        Returns:
            Action taken and result
        """
        usage = self.get_context_usage(messages)

        if not usage["needs_compression"]:
            return {
                "action": "none",
                "success": True,
                "messages": messages,
                "usage": usage,
            }

        self.archive_messages_to_jsonl(session_id, messages, {"reason": "auto_overflow"})

        compression_levels = [
            (0.6, "light"),
            (0.4, "medium"),
            (0.2, "heavy"),
            (0.1, "extreme"),
        ]

        for ratio, level_name in compression_levels:
            result = self.compress_context(
                messages,
                target_ratio=ratio,
                goal=goal,
                compressor_fn=compressor_fn,
            )

            if result.success and result.compressed_token_count <= self.context_limit * 0.9:
                return {
                    "action": f"compressed_{level_name}",
                    "success": True,
                    "messages": result.compressed_messages,
                    "usage": self.get_context_usage(result.compressed_messages),
                    "compression_result": result,
                }

        if longer_context_limit and session_creator:
            snapshot = self.create_snapshot(
                session_id, messages, goal,
                compression_level=len(compression_levels),
            )
            migration = self.migrate_session(
                snapshot, session_creator, longer_context_limit
            )
            return {
                "action": "migrated",
                "success": migration["success"],
                "new_session": migration.get("new_session"),
                "messages": migration.get("migration_messages", []),
                "migration_record": migration.get("migration_record"),
            }

        state_path = self.persist_runtime_state(
            session_id=session_id,
            goal=goal,
            intermediate_requirements=[],
            decisions_made=[],
            current_task=goal,
            last_round_input=messages[-1].get("content", "") if messages else "",
            last_round_output="",
            context_usage=usage,
        )

        result = self.compress_context(
            messages,
            target_ratio=0.1,
            goal=goal,
        )

        return {
            "action": "persisted_and_extreme_compressed",
            "success": result.success,
            "messages": result.compressed_messages,
            "state_file": str(state_path),
            "usage": self.get_context_usage(result.compressed_messages),
            "compression_result": result,
        }
