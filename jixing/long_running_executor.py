"""Long-running task executor for JiXing.

Continuously runs Ollama models to complete tasks with:
- Flexible time limits (seconds, minutes, hours, days, weeks, months)
- Quality limits (quality_threshold via self-evaluation)
- Automatic context compression when exceeding limits
- Session migration when context is full
- Persistent state for resume capability
- Progress tracking and reporting
"""

import json
import logging
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Union

from .compressor import SemanticCompressor
from .context_manager import ContextWindowManager, CompressionConfig
from .core import OllamaAdapter, Session, SessionManager

logger = logging.getLogger(__name__)


def parse_duration(duration: Union[str, int, float]) -> float:
    """Parse a duration string into seconds.

    Supported formats:
    - Integer/float: treated as seconds (e.g., 300, 3600.5)
    - String with unit suffix:
        s/sec/second/seconds -> seconds
        m/min/minute/minutes -> minutes
        h/hr/hour/hours -> hours
        d/day/days -> days
        w/week/weeks -> weeks
        M/month/months -> months (30 days)
        y/year/years -> years (365 days)

    Examples:
        parse_duration(300) -> 300.0
        parse_duration("5m") -> 300.0
        parse_duration("2h") -> 7200.0
        parse_duration("1d") -> 86400.0
        parse_duration("1w") -> 604800.0
        parse_duration("1M") -> 2592000.0
        parse_duration("2h 30m") -> 9000.0
        parse_duration("1d 12h") -> 129600.0
        parse_duration("unlimited") -> None
        parse_duration("forever") -> None
    """
    if duration is None:
        return None

    if isinstance(duration, (int, float)):
        return float(duration)

    if isinstance(duration, str):
        duration = duration.strip()
        if duration.lower() in ("unlimited", "forever", "infinite", "none", "0"):
            return None

        total_seconds = 0.0
        pattern = r"(\d+(?:\.\d+)?)\s*(seconds|second|sec|minutes|minute|min|hours|hour|hr|days|day|weeks|week|months|month|years|year|s|m|h|d|w|M|y)"
        matches = re.findall(pattern, duration)

        if not matches:
            try:
                return float(duration)
            except ValueError:
                raise ValueError(
                    f"Invalid duration format: '{duration}'. "
                    f"Use formats like '5m', '2h', '1d', '1w', '1M', or a number in seconds."
                )

        for value, unit in matches:
            value = float(value)
            if unit in ("s", "sec", "second", "seconds"):
                total_seconds += value
            elif unit in ("m", "min", "minute", "minutes"):
                total_seconds += value * 60
            elif unit in ("h", "hr", "hour", "hours"):
                total_seconds += value * 3600
            elif unit in ("d", "day", "days"):
                total_seconds += value * 86400
            elif unit in ("w", "week", "weeks"):
                total_seconds += value * 604800
            elif unit in ("M", "month", "months"):
                total_seconds += value * 2592000
            elif unit in ("y", "year", "years"):
                total_seconds += value * 31536000

        return total_seconds if total_seconds > 0 else None

    raise ValueError(f"Invalid duration type: {type(duration)}")


def format_duration(seconds: float) -> str:
    """Format seconds into a human-readable duration string."""
    if seconds is None:
        return "unlimited"

    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    elif seconds < 86400:
        return f"{seconds / 3600:.1f}h"
    elif seconds < 2592000:
        return f"{seconds / 86400:.1f}d"
    elif seconds < 31536000:
        return f"{seconds / 2592000:.1f}M"
    else:
        return f"{seconds / 31536000:.1f}y"


@dataclass
class TaskProgress:
    """Tracks progress of a long-running task."""

    task_id: str
    goal: str
    model_name: str
    started_at: str
    elapsed_seconds: float = 0
    rounds_completed: int = 0
    total_tokens_used: int = 0
    compressions_performed: int = 0
    migrations_performed: int = 0
    current_session_id: str = ""
    last_output: str = ""
    status: str = "running"
    quality_score: float = 0
    error: str = ""
    checkpoints: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TaskResult:
    """Final result of a long-running task."""

    success: bool
    task_id: str
    goal: str
    final_output: str
    rounds_completed: int
    total_tokens_used: int
    elapsed_seconds: float
    compressions_performed: int
    migrations_performed: int
    session_ids: list[str]
    quality_score: float
    stop_reason: str
    checkpoints: list[dict] = field(default_factory=list)
    files_written: list[str] = field(default_factory=list)
    error: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class LongRunningTaskExecutor:
    """Executes long-running tasks with automatic context management.

    Features:
    - Continuous loop until time limit or quality threshold reached
    - Automatic context compression using existing compressor
    - Session migration when context window is full
    - Self-evaluation of output quality
    - Progress checkpoints for resume capability
    - Error recovery and retry logic
    """

    SYSTEM_PROMPT_TEMPLATE = """You are working on a long-term task. Here is the context:

GOAL: {goal}

PROGRESS SO FAR:
{progress_summary}

CURRENT ROUND: {round_number}
PREVIOUS OUTPUT: {previous_output}

IMPORTANT: When writing code, ALWAYS use markdown code blocks with the file path as the language tag.
Format: ```path/to/file.ext
<code here>
```

Example:
```python
import pygame
# player tank code
```

For shell commands, use:
```bash
mkdir -p tank_game
pip install pygame
```

Continue working towards the goal. If the goal is complete, output exactly:
[COMPLETE]
followed by a brief summary of what was accomplished.

Otherwise, continue with the next step of the work."""

    QUALITY_EVALUATION_PROMPT = """Evaluate the following output against the original goal.

GOAL: {goal}
OUTPUT: {output}

Rate the quality from 0.0 to 1.0 where:
- 0.0: Completely irrelevant or wrong
- 0.3: Partially relevant but major issues
- 0.5: Mostly correct but incomplete
- 0.7: Good quality with minor issues
- 1.0: Perfect, goal fully achieved

Respond with ONLY a number between 0.0 and 1.0, nothing else."""

    def __init__(
        self,
        model_name: str,
        goal: str,
        base_url: str = "http://localhost:11434",
        max_duration: Optional[Union[str, int, float]] = None,
        max_rounds: Optional[int] = None,
        quality_threshold: float = 0.8,
        context_limit: int = 128000,
        compression_threshold: float = 0.8,
        checkpoint_dir: Optional[str | Path] = None,
        progress_callback: Optional[Callable[[TaskProgress], None]] = None,
        max_retries: int = 3,
        work_dir: Optional[str | Path] = None,
        **model_kwargs,
    ):
        self.model_name = model_name
        self.goal = goal
        self.base_url = base_url
        self.max_duration = parse_duration(max_duration)
        self.max_rounds = max_rounds
        self.quality_threshold = quality_threshold
        self.max_retries = max_retries
        self.work_dir = Path(work_dir).resolve() if work_dir else Path.cwd()
        self.model_kwargs = model_kwargs

        self.task_id = str(uuid.uuid4())[:8]
        self.start_time = time.time()
        self.rounds_completed = 0
        self.total_tokens_used = 0
        self.compressions_performed = 0
        self.migrations_performed = 0
        self.session_ids: list[str] = []
        self.checkpoints: list[dict] = []
        self._files_written: list[str] = []

        self.manager = SessionManager.get_instance()
        self.session = self.manager.create_session(
            model_provider="ollama",
            model_name=model_name,
            goal=goal,
            context_limit=context_limit,
        )
        self.session_ids.append(self.session.id)

        self.ctx_manager = ContextWindowManager(
            context_limit=context_limit,
            compression_threshold=compression_threshold,
            config=CompressionConfig(
                target_ratio=0.5,
                preserve_first_user=True,
                preserve_last_n=4,
            ),
        )

        self.adapter = OllamaAdapter(self.session, base_url=base_url)
        self.compressor = SemanticCompressor()

        self.checkpoint_dir = (
            Path(checkpoint_dir) if checkpoint_dir
            else Path.home() / ".jixing" / "task_checkpoints"
        )
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.progress_callback = progress_callback
        self._running = False
        self._last_output = ""

        self._check_model_capability()

    def _check_model_capability(self):
        """Check model size and warn if too small for complex tasks."""
        try:
            model_info = self.adapter.show_model(self.model_name)
            details = model_info.get("model_info", {})
            param_count = details.get("general.parameter_count", "")

            if param_count:
                size_b = self._parse_param_count(param_count)
                if size_b < 2:
                    logger.warning(
                        f"Model {self.model_name} has only {param_count} parameters. "
                        f"Models <2B may struggle with complex tasks like code generation. "
                        f"Consider using a larger model for better results."
                    )
                elif size_b < 7:
                    logger.info(
                        f"Model {self.model_name} ({param_count}) is suitable for simple tasks. "
                        f"Complex code generation may require multiple rounds."
                    )
        except Exception as e:
            logger.debug(f"Could not check model capability: {e}")

    def _parse_param_count(self, param_str: str) -> float:
        """Parse parameter count string like '820M', '1.5B', '7B' to billions."""
        param_str = param_str.upper().strip()
        if param_str.endswith("B"):
            return float(param_str[:-1])
        elif param_str.endswith("M"):
            return float(param_str[:-1]) / 1000
        elif param_str.endswith("T"):
            return float(param_str[:-1]) * 1000
        return float(param_str)

    def _extract_and_write_files(self, text: str) -> list[str]:
        """Extract code blocks from text and write them to files.

        Supports formats:
        - ```path/to/file.ext\\n<code>\\n```
        - ```ext\\n<code>\\n``` (creates file with extension)
        - ```bash\\n<command>\\n``` (logs but doesn't write)

        Returns list of written file paths.
        """
        written_files = []
        pattern = re.compile(r"```(\S*?)\s*\n(.*?)```", re.DOTALL)

        for match in pattern.finditer(text):
            lang_or_path = match.group(1).strip().lower()
            code = match.group(2).strip()

            if not code:
                continue

            if lang_or_path in ("bash", "sh", "shell", "zsh"):
                logger.info(f"Shell command detected (not executing): {code[:100]}...")
                continue

            file_path = self._resolve_file_path(lang_or_path, code)
            if file_path:
                self._write_file(file_path, code)
                written_files.append(str(file_path))
                self._files_written.append(str(file_path))

        if written_files:
            logger.info(f"Wrote {len(written_files)} file(s): {', '.join(written_files)}")

        return written_files

    def _resolve_file_path(self, lang_or_path: str, code: str) -> Optional[Path]:
        """Resolve a file path from language tag or code content."""
        if "/" in lang_or_path or "\\" in lang_or_path or "." in lang_or_path:
            return self.work_dir / lang_or_path

        ext_map = {
            "python": ".py", "py": ".py",
            "javascript": ".js", "js": ".js", "typescript": ".ts", "ts": ".ts",
            "html": ".html", "css": ".css",
            "json": ".json", "yaml": ".yaml", "yml": ".yml",
            "markdown": ".md", "md": ".md",
            "rust": ".rs", "go": ".go", "java": ".java",
            "cpp": ".cpp", "c": ".c", "h": ".h", "hpp": ".hpp",
            "ruby": ".rb", "php": ".php", "swift": ".swift",
            "kotlin": ".kt", "scala": ".scala",
            "sql": ".sql", "xml": ".xml",
            "toml": ".toml", "ini": ".ini", "cfg": ".cfg",
            "txt": ".txt", "text": ".txt",
        }
        ext = ext_map.get(lang_or_path, "")
        if not ext:
            return None

        if ext == ".py" and self._is_main_script(code):
            return self.work_dir / "main.py"

        name = self._infer_name_from_code(code, ext)
        return self.work_dir / f"{name}{ext}"

    def _is_main_script(self, code: str) -> bool:
        """Check if code looks like a complete runnable main script."""
        indicators = [
            r"if\s+__name__\s*==\s*['\"]__main__['\"]",
            r"while\s+\w+:",
            r"pygame\.init\(\)",
            r"app\s*=\s*\w+\(",
            r"def\s+main\s*\(",
        ]
        matches = sum(1 for p in indicators if re.search(p, code))
        return matches >= 2

    def _infer_name_from_code(self, code: str, ext: str) -> str:
        """Infer a filename from code content."""
        patterns = [
            (r"class\s+(\w+)[\(:]", 1),
            (r"(?:def|function)\s+(\w+)", 1),
            (r"(?:const|let|var)\s+(\w+)\s*=", 1),
        ]
        for pattern, group in patterns:
            match = re.search(pattern, code)
            if match:
                name = match.group(group)
                if name and name[0].isupper() and ext == ".py":
                    name = self._camel_to_snake(name)
                return name
        return "main"

    def _camel_to_snake(self, name: str) -> str:
        s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
        return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()

    def _write_file(self, file_path: Path, content: str):
        """Write content to a file, creating directories as needed."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info(f"  -> {file_path} ({len(content)} bytes)")

    def _get_progress_summary(self) -> str:
        """Generate a summary of progress so far."""
        elapsed = time.time() - self.start_time
        summary = (
            f"Task ID: {self.task_id}\n"
            f"Elapsed: {elapsed:.0f}s\n"
            f"Rounds completed: {self.rounds_completed}\n"
            f"Total tokens used: {self.total_tokens_used}\n"
            f"Compressions: {self.compressions_performed}\n"
            f"Migrations: {self.migrations_performed}\n"
            f"Current session: {self.session.id[:8]}..."
        )

        if self.checkpoints:
            summary += f"\nCheckpoints: {len(self.checkpoints)}"
            last_cp = self.checkpoints[-1]
            summary += f"\nLast checkpoint (round {last_cp.get('round', 0)}): {last_cp.get('summary', '')[:200]}"

        return summary

    def _evaluate_quality(self, output: str) -> float:
        """Self-evaluate output quality using the same model."""
        try:
            prompt = self.QUALITY_EVALUATION_PROMPT.format(
                goal=self.goal,
                output=output[-2000:] if len(output) > 2000 else output,
            )

            eval_session = self.manager.create_session(
                model_provider="ollama",
                model_name=self.model_name,
                goal="quality evaluation",
                context_limit=8000,
            )

            eval_adapter = OllamaAdapter(eval_session, base_url=self.base_url)
            response, metrics, _ = eval_adapter.run(prompt, timeout=60)

            score_str = response.strip().split("\n")[0]
            score = float(score_str)
            return min(1.0, max(0.0, score))

        except Exception as e:
            logger.warning(f"Quality evaluation failed: {e}")
            return 0.0

    def _check_stop_conditions(self) -> Optional[str]:
        """Check if any stop condition is met. Returns reason or None."""
        elapsed = time.time() - self.start_time

        if self.max_duration and elapsed >= self.max_duration:
            return f"Time limit reached ({format_duration(self.max_duration)})"

        if self.max_rounds and self.rounds_completed >= self.max_rounds:
            return f"Round limit reached ({self.max_rounds})"

        return None

    def _compress_context_if_needed(self) -> bool:
        """Compress context if approaching limits. Returns True if compressed."""
        usage = self.ctx_manager.get_context_usage(self.session.messages)

        if not usage["needs_compression"]:
            return False

        logger.info(
            f"Context usage: {usage['usage_percentage']}%, compressing..."
        )

        result = self.ctx_manager.compress_context(
            messages=self.session.messages,
            target_ratio=0.4,
            goal=self.goal,
            compressor_fn=lambda content: self.compressor.compress(content).compressed_text,
        )

        if result.success:
            self.session.messages = result.compressed_messages
            self.compressions_performed += 1
            self.manager._save_session(self.session)

            self.ctx_manager.archive_messages_to_jsonl(
                self.session.id,
                result.archived_messages,
                {"compression_round": self.rounds_completed},
            )

            logger.info(
                f"Compressed: {result.original_token_count} -> {result.compressed_token_count} tokens "
                f"(ratio: {result.compression_ratio:.2f})"
            )
            return True

        return False

    def _migrate_session_if_needed(self) -> bool:
        """Migrate to new session if context is still too large."""
        usage = self.ctx_manager.get_context_usage(self.session.messages)

        if usage["usage_percentage"] < 90:
            return False

        logger.info(
            f"Context at {usage['usage_percentage']}%, migrating session..."
        )

        snapshot = self.ctx_manager.create_snapshot(
            session_id=self.session.id,
            messages=self.session.messages,
            goal=self.goal,
            compression_level=3,
        )

        new_session = self.manager.create_session(
            model_provider="ollama",
            model_name=self.model_name,
            goal=self.goal,
            context_limit=self.session.context_limit * 2,
            parent_session_id=self.session.id,
            migration_metadata=snapshot.to_dict(),
        )
        self.session_ids.append(new_session.id)

        migration = self.ctx_manager.migrate_session(
            snapshot=snapshot,
            new_session_creator=lambda: new_session,
            context_limit=new_session.context_limit,
        )

        self.session = migration["new_session"]
        self.adapter = OllamaAdapter(self.session, base_url=self.base_url)
        self.migrations_performed += 1

        self.ctx_manager.archive_messages_to_jsonl(
            snapshot.source_session_id,
            self.session.messages,
            {"migration_round": self.rounds_completed},
        )

        logger.info(f"Migrated to session {new_session.id[:8]}...")
        return True

    def _save_checkpoint(self, output: str, quality: float):
        """Save a checkpoint for resume capability."""
        checkpoint = {
            "task_id": self.task_id,
            "round": self.rounds_completed,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "output_summary": output[:500] if output else "",
            "quality_score": quality,
            "session_id": self.session.id,
            "total_tokens": self.total_tokens_used,
        }
        self.checkpoints.append(checkpoint)

        cp_path = self.checkpoint_dir / f"{self.task_id}_round_{self.rounds_completed}.json"
        with open(cp_path, "w", encoding="utf-8") as f:
            json.dump(checkpoint, f, ensure_ascii=False, indent=2)

    def _build_prompt(self, previous_output: str) -> str:
        """Build the prompt for the next round."""
        return self.SYSTEM_PROMPT_TEMPLATE.format(
            goal=self.goal,
            progress_summary=self._get_progress_summary(),
            round_number=self.rounds_completed + 1,
            previous_output=previous_output[-3000:] if previous_output else "No previous output yet.",
        )

    def _emit_progress(self):
        """Emit progress update."""
        progress = TaskProgress(
            task_id=self.task_id,
            goal=self.goal,
            model_name=self.model_name,
            started_at=datetime.fromtimestamp(self.start_time, tz=timezone.utc).isoformat(),
            elapsed_seconds=time.time() - self.start_time,
            rounds_completed=self.rounds_completed,
            total_tokens_used=self.total_tokens_used,
            compressions_performed=self.compressions_performed,
            migrations_performed=self.migrations_performed,
            current_session_id=self.session.id,
            last_output=self._last_output[-500:] if self._last_output else "",
            status="running",
            quality_score=self._last_quality if hasattr(self, "_last_quality") else 0,
        )

        if self.progress_callback:
            try:
                self.progress_callback(progress)
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")

    def execute(self) -> TaskResult:
        """Execute the long-running task.

        Returns:
            TaskResult with final output and metadata
        """
        self._running = True
        self._last_output = ""
        self._last_quality = 0.0
        full_output_parts = []

        logger.info(
            f"Starting long-running task {self.task_id}\n"
            f"Goal: {self.goal}\n"
            f"Model: {self.model_name}\n"
            f"Max duration: {format_duration(self.max_duration)}\n"
            f"Max rounds: {self.max_rounds or 'unlimited'}\n"
            f"Quality threshold: {self.quality_threshold}"
        )

        try:
            while self._running:
                stop_reason = self._check_stop_conditions()
                if stop_reason:
                    logger.info(f"Stopping: {stop_reason}")
                    break

                self._compress_context_if_needed()
                self._migrate_session_if_needed()

                prompt = self._build_prompt(self._last_output)

                timeout = self.model_kwargs.get("timeout", 600)
                idle_timeout = self.model_kwargs.get("idle_timeout", max(120, timeout // 5))
                response = None
                metrics = {}
                round_failed = True

                for attempt in range(self.max_retries + 1):
                    try:
                        response, metrics, _ = self.adapter.run_with_idle_timeout(
                            prompt,
                            timeout=timeout,
                            idle_timeout=idle_timeout,
                        )
                        round_failed = False
                        break
                    except Exception as e:
                        is_timeout = "timeout" in str(e).lower() or "timed out" in str(e).lower() or "stuck" in str(e).lower()
                        if attempt < self.max_retries:
                            wait_time = min(2 ** attempt * 5, 60)
                            logger.warning(
                                f"Round {self.rounds_completed + 1} attempt {attempt + 1} failed "
                                f"({'timeout' if is_timeout else 'error'}): {e}. "
                                f"Retrying in {wait_time}s (attempt {attempt + 2}/{self.max_retries + 1})..."
                            )
                            time.sleep(wait_time)
                            if is_timeout:
                                timeout = int(timeout * 1.5)
                                idle_timeout = int(idle_timeout * 1.5)
                                logger.info(f"Increased timeout to {timeout}s, idle_timeout to {idle_timeout}s for next attempt")
                        else:
                            logger.error(
                                f"Round {self.rounds_completed + 1} failed after {self.max_retries + 1} attempts: {e}"
                            )

                if round_failed:
                    self._last_output = f"Error in round {self.rounds_completed + 1}: all {self.max_retries + 1} attempts failed"
                    full_output_parts.append(self._last_output)
                    self.rounds_completed += 1
                    self._emit_progress()
                    time.sleep(5)
                    continue

                tokens_this_round = metrics.get("eval_count", 0)
                self.total_tokens_used += tokens_this_round
                self._last_output = response
                full_output_parts.append(response)
                self.rounds_completed += 1

                written = self._extract_and_write_files(response)

                self.session.add_message("user", prompt)
                self.session.add_message("assistant", response, metrics=metrics)
                self.manager._save_session(self.session)

                quality = self._evaluate_quality(response)
                self._last_quality = quality

                logger.info(
                    f"Round {self.rounds_completed}: "
                    f"tokens={tokens_this_round}, "
                    f"quality={quality:.2f}, "
                    f"total_tokens={self.total_tokens_used}"
                )

                self._save_checkpoint(response, quality)
                self._emit_progress()

                if "[COMPLETE]" in response or quality >= self.quality_threshold:
                    if quality >= self.quality_threshold:
                        stop_reason = f"Quality threshold reached ({quality:.2f} >= {self.quality_threshold})"
                    else:
                        stop_reason = "Task marked complete by model"
                    logger.info(f"Stopping: {stop_reason}")
                    break

        except KeyboardInterrupt:
            stop_reason = "Interrupted by user"
            logger.info(stop_reason)

        finally:
            self._running = False

        final_output = "\n\n".join(full_output_parts)

        result = TaskResult(
            success=stop_reason and "Error" not in str(stop_reason),
            task_id=self.task_id,
            goal=self.goal,
            final_output=final_output,
            rounds_completed=self.rounds_completed,
            total_tokens_used=self.total_tokens_used,
            elapsed_seconds=time.time() - self.start_time,
            compressions_performed=self.compressions_performed,
            migrations_performed=self.migrations_performed,
            session_ids=self.session_ids,
            quality_score=self._last_quality,
            stop_reason=stop_reason or "Unknown",
            checkpoints=self.checkpoints,
            files_written=self._files_written,
        )

        result_path = self.checkpoint_dir / f"{self.task_id}_result.json"
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)

        logger.info(
            f"Task {self.task_id} completed:\n"
            f"  Rounds: {result.rounds_completed}\n"
            f"  Tokens: {result.total_tokens_used}\n"
            f"  Time: {result.elapsed_seconds:.1f}s\n"
            f"  Quality: {result.quality_score:.2f}\n"
            f"  Reason: {result.stop_reason}"
        )

        return result

    def stop(self):
        """Stop the running task."""
        self._running = False
        logger.info(f"Task {self.task_id} stop requested")

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        **overrides,
    ) -> "LongRunningTaskExecutor":
        """Resume a task from a checkpoint file."""
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            checkpoint = json.load(f)

        task_id = checkpoint.get("task_id", "")
        session_id = checkpoint.get("session_id", "")

        manager = SessionManager.get_instance()
        session = manager.get_session(session_id)

        if not session:
            raise ValueError(f"Session {session_id} not found")

        executor = cls(
            model_name=session.model_name,
            goal=session.goal,
            **overrides,
        )

        executor.task_id = task_id
        executor.session = session
        executor.adapter = OllamaAdapter(session, base_url=executor.base_url)
        executor.rounds_completed = checkpoint.get("round", 0)
        executor.total_tokens_used = checkpoint.get("total_tokens", 0)

        return executor
