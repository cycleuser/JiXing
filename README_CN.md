# JiXing

本地AI模型助手与长期记忆系统。

一个统一运行本地AI模型（Ollama、Moxing）的工具，完整记录所有交互以便后续检索。

## 功能特点

- **多模型支持**：统一界面运行 Ollama 和 Moxing 模型
- **完整日志**：记录所有提示词、响应、时间戳、系统信息和性能指标
- **长期记忆**：基于 SQLite 的存储，可搜索所有历史交互
- **Web界面**：可选的 Flask Web UI，用于浏览会话和搜索消息
- **CLI界面**：功能完整的命令行工具，支持子命令
- **Python API**：通过 ToolResult 模式进行编程访问
- **OpenAI函数调用**：通过函数工具定义与 OpenAI 智能体集成

## 系统要求

- Python 3.10+
- Ollama（可选，支持 Ollama 模型）
- Moxing（可选，支持 GGUF 模型）
- Flask（可选，支持 Web 界面）

## 安装

```bash
pip install -e .
```

## 快速开始

### CLI 使用

运行 Ollama 模型：
```bash
jixing ollama run gemma3:1b "你好，近来如何？"
```

交互模式：
```bash
jixing ollama run gemma3:1b -i
```

运行 Moxing（GGUF）模型：
```bash
jixing moxing serve gguf-model-name "你的提示词"
```

列出所有会话：
```bash
jixing sessions list
```

搜索消息历史：
```bash
jixing search "之前关于Python的对话"
```

显示统计信息：
```bash
jixing stats
```

启动 Web 界面：
```bash
jixing web --port 5000
```

### Python API

```python
from jixing import run_ollama, query_sessions, ToolResult

result = run_ollama(model="gemma3:1b", prompt="你好！")
print(result.success)    # True / False
print(result.data)       # 响应数据
print(result.metadata)   # 元数据，包含版本信息
```

## CLI 帮助

```
$ jixing --help
usage: jixing [-h] [-V] [-v] [-o OUTPUT] [--json] [-q] {ollama,moxing,sessions,search,stats,info,web} ...

位置参数：
  {ollama,moxing,sessions,search,stats,info,web}
    ollama              运行 Ollama 模型
    moxing              运行 Moxing 模型
    sessions            管理会话
    search              搜索消息历史
    stats               显示统计信息
    info                显示系统信息
    web                 启动 Web 界面

选项：
  -V, --version         显示版本
  -v, --verbose         详细输出
  -o, --output OUTPUT   输出路径
  --json                JSON 格式输出
  -q, --quiet           静默模式
```

## 项目结构

```
jixing/
├── jixing/
│   ├── __init__.py     # 包导出
│   ├── __version__.py  # 版本信息
│   ├── core.py         # 核心逻辑、会话、模型运行器
│   ├── api.py          # ToolResult API 函数
│   ├── cli.py          # CLI 接口
│   ├── db.py           # SQLite 数据库
│   ├── models/         # 模型适配器
│   ├── tools.py        # OpenAI 函数定义
│   └── web.py          # Flask Web 界面
├── tests/
├── pyproject.toml
└── README.md
```

## 智能体集成

将 JiXing 与 OpenAI 函数调用智能体集成：

```python
from openai import OpenAI
from jixing.tools import TOOLS, dispatch

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "运行 Ollama 模型 gemma3:1b，提示词为 '你好'"}],
    tools=TOOLS,
)

for tool_call in response.choices[0].message.tool_calls:
    result = dispatch(tool_call.function.name, tool_call.function.arguments)
    # 将结果反馈给 LLM...
```

## 许可证

GPL-3.0
