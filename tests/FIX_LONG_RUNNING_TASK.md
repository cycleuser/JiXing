# 修复 long_running_executor 任务完成但无文件生成的问题

## 问题描述

运行 `jixing task run gemma4:e2b "写个坦克大战" --timeout 900 --max-retries 5` 后，任务显示完成（4轮，质量0.70），但实际没有创建任何项目文件。

### 原始输出示例

```
Task completed successfully!
  Task ID: cd671e37
  Rounds: 4
  Tokens: 6104
  Time: 1111.7s
  Quality: 0.70
  Stop reason: Task marked complete by model
```

但 `ls` 显示没有新文件生成。

## 根因分析

### 1. 模型输出格式问题

模型输出的是 ` ```python` 格式（只有语言标签），而不是 ` ```path/to/file.py` 格式：

```python
# 模型实际输出格式
```python
import pygame
# 代码内容
```

# 期望的输出格式
```tank_game/main.py
import pygame
# 代码内容
```
```

### 2. 多轮增量输出问题

模型在每轮只输出部分代码（增量更新），而不是完整文件：
- Round 1: 基础框架
- Round 2: 添加坦克类
- Round 3: 添加敌人
- Round 4: `[COMPLETE]` 标记

最后一轮只是完成标记，没有完整代码。

### 3. 代码提取逻辑不够鲁棒

原始 `_extract_and_write_files` 方法：
- 正则表达式 ` ```(\S*?)\s*\n(.*?)``` ` 能匹配语言标签
- `_resolve_file_path` 能推断文件名
- 但没有处理多轮增量输出的情况
- 没有最终整合步骤确保完整项目生成

## 修复方案

### 1. 强化系统提示 (long_running_executor.py:180-225)

**修改前：**
```python
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
```

**修改后：**
```python
SYSTEM_PROMPT_TEMPLATE = """You are working on a long-term task. Here is the context:

GOAL: {goal}

PROGRESS SO FAR:
{progress_summary}

CURRENT ROUND: {round_number}
PREVIOUS OUTPUT: {previous_output}

CRITICAL RULES FOR CODE OUTPUT:
1. ALWAYS write the COMPLETE, FULLY RUNNABLE code in every response. Never output partial snippets.
2. ALWAYS use markdown code blocks with the EXACT file path as the fence tag.
   Format: ```path/to/file.py
   <COMPLETE code here>
   ```
3. If creating a project with multiple files, write EACH file in its own code block with its path.
4. Each round should contain the FULL consolidated code, not just the new changes.
5. For a single-file project, always output the entire working code.

Examples of CORRECT format:
```tank_game/main.py
import pygame
# ... complete runnable code here ...
if __name__ == "__main__":
    main()
```

```tank_game/README.md
# Tank Game
...
```

For shell commands, use:
```bash
mkdir -p tank_game
pip install pygame
```

Continue working towards the goal. When the goal is fully achieved with working code written to files, output exactly:
[COMPLETE]
followed by a brief summary.

Otherwise, continue with the next step, writing complete code files."""
```

**关键改进：**
- 明确要求每次输出**完整可运行代码**
- 强调使用文件路径格式
- 提供正确的多文件示例
- 明确禁止增量输出

### 2. 改进代码提取逻辑 (long_running_executor.py:351-383)

**修改前：**
```python
def _extract_and_write_files(self, text: str) -> list[str]:
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
```

**修改后：**
```python
def _extract_and_write_files(self, text: str) -> list[str]:
    written_files = []
    # More robust pattern: handle various fence formats
    pattern = re.compile(r"```(\S*)\s*\n(.*?)```", re.DOTALL)

    for match in pattern.finditer(text):
        lang_or_path = match.group(1).strip()
        code = match.group(2).strip()

        if not code:
            continue

        # Normalize for comparison
        lang_lower = lang_or_path.lower()

        if lang_lower in ("bash", "sh", "shell", "zsh", "cmd", "powershell"):
            logger.info(f"Shell command detected (not executing): {code[:100]}...")
            continue

        # Skip non-code blocks
        if lang_lower in ("text", "plaintext", "output", "log", "diff", "json"):
            continue

        file_path = self._resolve_file_path(lang_or_path, code)
        if file_path:
            self._write_file(file_path, code)
            written_files.append(str(file_path))
            self._files_written.append(str(file_path))

    if written_files:
        logger.info(f"Wrote {len(written_files)} file(s): {', '.join(written_files)}")

    return written_files
```

**关键改进：**
- 更鲁棒的正则表达式（非贪婪改为贪婪）
- 保留原始大小写用于路径解析
- 跳过非代码块（text, plaintext, output, log, diff, json）
- 支持更多 shell 类型（cmd, powershell）
- 未知语言时给出警告

### 3. 改进文件路径解析 (long_running_executor.py:385-413)

**修改前：**
```python
def _resolve_file_path(self, lang_or_path: str, code: str) -> Optional[Path]:
    if "/" in lang_or_path or "\\" in lang_or_path or "." in lang_or_path:
        return self.work_dir / lang_or_path

    ext_map = {
        "python": ".py", "py": ".py",
        # ... 其他映射
    }
    ext = ext_map.get(lang_or_path, "")
    if not ext:
        return None

    if ext == ".py" and self._is_main_script(code):
        return self.work_dir / "main.py"

    name = self._infer_name_from_code(code, ext)
    return self.work_dir / f"{name}{ext}"
```

**修改后：**
```python
def _resolve_file_path(self, lang_or_path: str, code: str) -> Optional[Path]:
    # Check if it looks like a file path
    if "/" in lang_or_path or "\\" in lang_or_path or "." in lang_or_path:
        return self.work_dir / lang_or_path

    ext_map = {
        "python": ".py", "py": ".py",
        # ... 其他映射
    }
    ext = ext_map.get(lang_or_path.lower(), "")
    if not ext:
        logger.warning(f"Unknown language tag '{lang_or_path}', skipping code block")
        return None

    # For Python, try to infer a good filename
    if ext == ".py":
        if self._is_main_script(code):
            return self.work_dir / "main.py"

    name = self._infer_name_from_code(code, ext)
    return self.work_dir / f"{name}{ext}"
```

**关键改进：**
- 使用 `.lower()` 进行语言映射查找
- 未知语言时记录警告而不是静默跳过

### 4. 新增最终整合步骤 (long_running_executor.py:650-697)

**新增方法：**
```python
def _final_consolidation(self) -> list[str]:
    """Ask the model to output all complete files for the project.

    This ensures that even if the model output incremental code during
    the task, we get the final consolidated versions of all files.
    """
    consolidation_prompt = """The task is now complete. Please output ALL the final files needed for this project.

For EACH file, use this exact format:
```path/to/file.py
<COMPLETE final code here>
```

Include every file that needs to exist for the project to work. Do not skip any files.
Do not include explanations, just the code blocks with file paths."""

    written_files = []
    try:
        timeout = self.model_kwargs.get("timeout", 600)
        idle_timeout = self.model_kwargs.get("idle_timeout", max(120, timeout // 5))

        logger.info("Running final consolidation to write all project files...")
        response, metrics, _ = self.adapter.run_with_idle_timeout(
            consolidation_prompt,
            timeout=timeout,
            idle_timeout=idle_timeout,
        )

        written_files = self._extract_and_write_files(response)

        tokens = metrics.get("eval_count", 0)
        self.total_tokens_used += tokens
        self._last_output = response

        self.session.add_message("user", consolidation_prompt)
        self.session.add_message("assistant", response, metrics=metrics)
        self.manager._save_session(self.session)

        if written_files:
            logger.info(f"Final consolidation wrote {len(written_files)} file(s): {', '.join(written_files)}")
        else:
            logger.warning("Final consolidation produced no files. Model may not have followed format.")

    except Exception as e:
        logger.warning(f"Final consolidation failed: {e}")

    return written_files
```

**在 execute 方法中调用：**
```python
finally:
    self._running = False

# Final consolidation: ensure all project files are written
if not self._files_written:
    logger.info("No files written during task, attempting final consolidation...")
    self._final_consolidation()
else:
    # Even if files were written, do a final consolidation to get complete versions
    logger.info("Running final consolidation to ensure complete project files...")
    self._final_consolidation()

final_output = "\n\n".join(full_output_parts)
```

**关键改进：**
- 任务完成后额外请求模型输出所有完整文件
- 确保即使中间轮次没有正确写文件，最后也能生成完整项目
- 记录详细的日志信息

### 5. 添加测试用例 (tests/test_long_running.py)

**新增测试类：**
```python
class TestFileExtraction:
    """Test code extraction and file writing logic."""

    @pytest.fixture
    def mock_session_manager(self):
        with patch("jixing.long_running_executor.SessionManager") as mock:
            manager = MagicMock()
            session = MagicMock()
            session.id = "test-session-id"
            session.model_name = "test-model"
            session.goal = "test goal"
            session.context_limit = 128000
            session.messages = []
            manager.create_session.return_value = session
            manager.get_instance.return_value = manager
            mock.get_instance.return_value = manager
            yield mock, manager, session

    @pytest.fixture
    def mock_ollama_adapter(self):
        with patch("jixing.long_running_executor.OllamaAdapter") as mock:
            adapter = MagicMock()
            adapter.run.return_value = ("test response", {"eval_count": 100}, {})
            adapter.show_model.return_value = {"model_info": {"general.parameter_count": "1B"}}
            mock.return_value = adapter
            yield mock, adapter

    def test_extract_python_code_block(self, mock_session_manager, mock_ollama_adapter, tmp_path):
        """Test extracting code from ```python block."""
        # ... 测试代码

    def test_extract_code_with_file_path(self, mock_session_manager, mock_ollama_adapter, tmp_path):
        """Test extracting code from ```path/to/file.py block."""
        # ... 测试代码

    def test_extract_multiple_files(self, mock_session_manager, mock_ollama_adapter, tmp_path):
        """Test extracting multiple code blocks."""
        # ... 测试代码

    def test_skip_bash_blocks(self, mock_session_manager, mock_ollama_adapter, tmp_path):
        """Test that bash blocks are not written as files."""
        # ... 测试代码

    def test_is_main_script(self, mock_session_manager, mock_ollama_adapter, tmp_path):
        """Test main script detection."""
        # ... 测试代码
```

**测试结果：**
```
tests/test_long_running.py::TestFileExtraction::test_extract_python_code_block PASSED
tests/test_long_running.py::TestFileExtraction::test_extract_code_with_file_path PASSED
tests/test_long_running.py::TestFileExtraction::test_extract_multiple_files PASSED
tests/test_long_running.py::TestFileExtraction::test_skip_bash_blocks PASSED
tests/test_long_running.py::TestFileExtraction::test_is_main_script PASSED

5 passed in 0.17s
```

## 使用建议

1. **使用更大的模型**：小模型（如 `qwen3.5:0.8B`）可能无法遵循复杂的格式要求，建议使用 `gemma4:e2b`

2. **设置合理的超时时间**：代码生成需要较长时间，建议 `--timeout 900` 或更长

3. **检查工作目录**：文件会写入当前工作目录或 `--work-dir` 指定的目录

4. **查看日志**：使用 `-v` 参数查看详细日志，确认文件是否正确写入

## 相关文件

- `jixing/long_running_executor.py` - 主要修复文件
- `tests/test_long_running.py` - 新增测试用例
- `tests/FIX_LONG_RUNNING_TASK.md` - 本文档
