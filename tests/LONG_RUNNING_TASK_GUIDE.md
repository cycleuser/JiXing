# JiXing 长时间运行任务完整训练过程

## 概述

JiXing 的 `task run` 命令支持长时间持续运行，自动多轮迭代直到任务完成。本文档记录完整的训练/使用过程。

## 核心概念

### 什么是长时间运行任务？

- 模型持续多轮对话，自动迭代改进
- 每轮自动评估输出质量
- 达到质量阈值或模型标记完成时停止
- 支持自动上下文压缩和会话迁移
- 支持断点续传

### 关键参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `model` | 使用的模型 | `gemma4:e2b` |
| `--timeout` | 每轮超时时间（秒） | `900` |
| `--max-retries` | 每轮最大重试次数 | `5` |
| `--quality` | 质量阈值（0-1） | `0.8` |
| `--max-duration` | 总时长限制 | `2h`, `30m` |
| `--max-rounds` | 最大轮次 | `20` |
| `--work-dir` | 工作目录 | `./tank_game` |

## 完整训练过程

### 第一步：准备环境

```bash
# 安装 jixing
pip install -e ./

# 确保 Ollama 运行中
ollama serve

# 拉取模型（建议使用 4B 以上模型）
ollama pull gemma4:e2b
```

### 第二步：启动长时间任务

```bash
# 基础用法
jixing task run gemma4:e2b "写个坦克大战游戏"

# 完整参数
jixing task run gemma4:e2b "写个坦克大战游戏" \
  --timeout 9000 \
  --max-retries 50 \
  --quality 0.8 \
  --work-dir ./tank_game
```

### 第三步：观察任务运行

任务启动后会显示：

```
Starting long-running task:
  Model: gemma4:e2b
  Goal: 写个坦克大战
  Quality threshold: 0.8
  Timeout per round: 900s
  Max retries: 5

INFO: Created session xxxxxxxx for ollama/gemma4:e2b
INFO: Starting long-running task xxxxxxxx
Goal: 写个坦克大战
Model: gemma4:e2b
Max duration: unlimited
Max rounds: unlimited
Quality threshold: 0.8
```

### 第四步：多轮迭代过程

任务会自动进行多轮迭代，每轮输出类似：

```
INFO: Round 1: tokens=857, quality=0.70, total_tokens=857
INFO: Round 2: tokens=1198, quality=0.50, total_tokens=2055
INFO: Round 3: tokens=1813, quality=0.50, total_tokens=3868
INFO: Round 4: tokens=2236, quality=0.70, total_tokens=6104
INFO: Round 5: tokens=3200, quality=0.85, total_tokens=9304
INFO: Stopping: Quality threshold reached (0.85 >= 0.8)
```

**每轮发生了什么：**

1. **构建提示词**：包含目标、进度、上一轮输出
2. **模型生成**：调用模型生成代码或文本
3. **文件提取**：自动从代码块中提取文件并写入
4. **质量评估**：用同一模型评估输出质量（0-1）
5. **检查停止条件**：质量达标、时间到、轮次到、或模型标记 `[COMPLETE]`
6. **保存检查点**：每轮保存进度到 `~/.jixing/task_checkpoints/`

### 第五步：上下文管理（自动）

任务运行中会自动处理：

```
# 上下文使用超过 80% 时自动压缩
INFO: Context usage: 82%, compressing...
INFO: Compressed: 45000 -> 22000 tokens (ratio: 0.49)

# 上下文超过 90% 时自动迁移会话
INFO: Context at 92%, migrating session...
INFO: Migrated to session yyyyyyyy...
```

### 第六步：最终整合

任务完成后，会自动执行最终整合步骤：

```
INFO: Running final consolidation to ensure complete project files...
INFO: Final consolidation wrote 3 file(s): tank_game/main.py, tank_game/README.md, tank_game/requirements.txt
```

这确保即使中间轮次只输出了部分代码，最后也能得到完整的项目文件。

### 第七步：任务完成

```
Task completed successfully!
  Task ID: cd671e37
  Rounds: 5
  Tokens: 9304
  Time: 1111.7s
  Quality: 0.85
  Stop reason: Quality threshold reached (0.85 >= 0.8)

Files written (3):
  - /path/to/tank_game/main.py
  - /path/to/tank_game/README.md
  - /path/to/tank_game/requirements.txt

Final output:
----------------------------------------
[模型最终输出内容]
```

## 任务停止条件

任务会在以下情况停止：

| 条件 | 说明 | 示例输出 |
|------|------|----------|
| 质量达标 | 质量评分 >= 阈值 | `Quality threshold reached (0.85 >= 0.8)` |
| 模型完成 | 模型输出 `[COMPLETE]` | `Task marked complete by model` |
| 时间到 | 达到 `--max-duration` | `Time limit reached (2.0h)` |
| 轮次到 | 达到 `--max-rounds` | `Round limit reached (20)` |
| 用户中断 | `Ctrl+C` | `Interrupted by user` |

## 监控任务状态

### 查看任务列表

```bash
jixing task list
```

输出：
```
Found 3 tasks:

  cd671e37 [OK] rounds=5 quality=0.85 time=1112s
  2445809c [FAIL] rounds=0 quality=0.00 time=16s
  a0d567b1 [FAIL] rounds=0 quality=0.00 time=0s
```

### 查看任务详情

```bash
jixing task status cd671e37
```

输出：
```
Task: cd671e37
Checkpoints: 5
Status: completed
Rounds: 5
Quality: 0.85
Stop reason: Quality threshold reached (0.85 >= 0.8)

Checkpoints:
  Round 1: quality=0.70
  Round 2: quality=0.50
  Round 3: quality=0.50
  Round 4: quality=0.70
  Round 5: quality=0.85
```

### 查看检查点文件

```bash
ls ~/.jixing/task_checkpoints/cd671e37_*
# cd671e37_round_1.json
# cd671e37_round_2.json
# ...
# cd671e37_result.json
```

## 断点续传

如果任务中断，可以从检查点恢复：

```bash
# 从最新检查点恢复
jixing task resume cd671e37

# 从特定轮次恢复
jixing task resume cd671e37 --from-round 3
```

## 模型选择建议

### 推荐模型

| 模型 | 大小 | 适用场景 | 说明 |
|------|------|----------|------|
| `gemma4:e2b` | E2B | 代码生成、长时间任务 | 默认推荐，支持工具调用 |

### 不推荐

| 模型 | 大小 | 问题 |
|------|------|------|
| `qwen3.5:0.8B` | 0.8B | 太小，无法遵循格式要求 |
| `phi3:3.8b` | 3.8B | 代码生成能力弱 |

**小模型问题：**
- 无法理解复杂的系统提示
- 输出格式混乱
- 代码不完整或有语法错误
- 质量评估不准确

## 常见问题

### Q: 任务一直不结束怎么办？

```bash
# 设置最大轮次
jixing task run gemma4:e2b "写个坦克大战" --max-rounds 10

# 设置最大时长
jixing task run gemma4:e2b "写个坦克大战" --max-duration 2h

# 手动中断
Ctrl+C
```

### Q: 质量评分一直很低？

1. 使用更大的模型
2. 降低质量阈值：`--quality 0.6`
3. 增加每轮超时：`--timeout 1200`

### Q: 文件没有生成？

1. 检查日志：`jixing task run ... -v`
2. 检查工作目录：`ls ./tank_game`
3. 查看检查点：`cat ~/.jixing/task_checkpoints/xxx_result.json`

### Q: 如何指定输出目录？

```bash
jixing task run gemma4:e2b "写个坦克大战" --work-dir ./my_project
```

## 完整示例：坦克大战项目

```bash
# 1. 创建工作目录
mkdir -p tank_game && cd tank_game

# 2. 启动任务
jixing task run gemma4:e2b \
  "创建一个完整的坦克大战游戏，使用 Python 和 Pygame。
   要求：
   - 玩家坦克可以用方向键移动
   - 空格键发射子弹
   - 5个敌人坦克自动移动
   - 碰撞检测
   - 分数显示
   - 游戏结束画面" \
  --timeout 900 \
  --max-retries 5 \
  --quality 0.8 \
  --work-dir .

# 3. 等待任务完成（可能需要 10-30 分钟）

# 4. 查看生成的文件
ls -la

# 5. 运行游戏
python main.py
```

## 高级用法

### 自定义质量评估

可以在代码中修改质量评估提示词：

```python
# 在 long_running_executor.py 中
QUALITY_EVALUATION_PROMPT = """
根据你的需求自定义质量评估标准...
"""
```

### 进度回调

```python
from jixing.long_running_executor import LongRunningTaskExecutor, TaskProgress

def my_callback(progress: TaskProgress):
    print(f"Round {progress.rounds_completed}: quality={progress.quality_score:.2f}")

executor = LongRunningTaskExecutor(
    model_name="gemma4:e2b",
    goal="写个坦克大战",
    progress_callback=my_callback,
)
result = executor.execute()
```

### 从检查点恢复

```python
from jixing.long_running_executor import LongRunningTaskExecutor

executor = LongRunningTaskExecutor.from_checkpoint(
    "~/.jixing/task_checkpoints/xxx_round_3.json",
    max_rounds=10,  # 可以继续设置参数
)
result = executor.execute()
```

## 日志级别

```bash
# 详细日志
jixing task run gemma4:e2b "写个坦克大战" -v

# 静默模式
jixing task run gemma4:e2b "写个坦克大战" -q

# JSON 输出
jixing task run gemma4:e2b "写个坦克大战" --json
```

## 相关文件

- `jixing/long_running_executor.py` - 长时间任务执行器
- `jixing/cli.py` - 命令行接口
- `jixing/core.py` - 核心会话管理
- `jixing/compressor.py` - 上下文压缩
- `jixing/context_manager.py` - 上下文管理
- `tests/test_long_running.py` - 测试用例
