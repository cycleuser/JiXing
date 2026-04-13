# JiXing

Local AI Model Assistant with Long-term Memory System.

A unified tool to run local AI models (Ollama, Moxing) with complete logging of all interactions for later retrieval.

## Features

- **Multi-Provider Support**: Run models from Ollama and Moxing with unified interface
- **Complete Logging**: Records all prompts, responses, timestamps, system info, and performance metrics
- **Long-term Memory**: SQLite-based storage for searching through all past interactions
- **Web Interface**: Optional Flask-based web UI for browsing sessions and searching messages
- **CLI Interface**: Full-featured command-line tool with subcommands
- **Python API**: Programmatic access via ToolResult pattern
- **OpenAI Function Calling**: Integrate with OpenAI agents via function tool definitions

## Requirements

- Python 3.10+
- Ollama (optional, for Ollama model support)
- Moxing (optional, for GGUF model support)
- Flask (optional, for web interface)

## Installation

```bash
pip install -e .
```

## Quick Start

### CLI Usage

Run an Ollama model:
```bash
jixing ollama run gemma3:1b "Hello, how are you?"
```

Run in interactive mode:
```bash
jixing ollama run gemma3:1b -i
```

Run a Moxing (GGUF) model:
```bash
jixing moxing serve gguf-model-name "Your prompt here"
```

List all sessions:
```bash
jixing sessions list
```

Search through message history:
```bash
jixing search "previous conversation about Python"
```

Show statistics:
```bash
jixing stats
```

Start web interface:
```bash
jixing web --port 5000
```

### Python API

```python
from jixing import run_ollama, query_sessions, ToolResult

result = run_ollama(model="gemma3:1b", prompt="Hello!")
print(result.success)    # True / False
print(result.data)       # Response data
print(result.metadata)   # Metadata including version
```

## CLI Help

```
$ jixing --help
usage: jixing [-h] [-V] [-v] [-o OUTPUT] [--json] [-q] {ollama,moxing,sessions,search,stats,info,web} ...

positional arguments:
  {ollama,moxing,sessions,search,stats,info,web}
    ollama              Run Ollama model
    moxing              Run Moxing model
    sessions            Manage sessions
    search              Search message history
    stats               Show statistics
    info                Show system information
    web                 Start web interface

options:
  -V, --version         Show version
  -v, --verbose         Verbose output
  -o, --output OUTPUT   Output path
  --json                Output results as JSON
  -q, --quiet           Suppress non-essential output
```

## Architecture

```
jixing/
├── jixing/
│   ├── __init__.py     # Package exports
│   ├── __version__.py  # Version info
│   ├── core.py         # Core logic, Session, ModelRunner
│   ├── api.py          # ToolResult API functions
│   ├── cli.py          # CLI interface
│   ├── db.py           # SQLite database
│   ├── models/         # Model adapters
│   ├── tools.py        # OpenAI function definitions
│   └── web.py          # Flask web interface
├── tests/
├── pyproject.toml
└── README.md
```

## Agent Integration

Integrate JiXing with OpenAI function-calling agents:

```python
from openai import OpenAI
from jixing.tools import TOOLS, dispatch

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Run Ollama model gemma3:1b with prompt 'Hello'"}],
    tools=TOOLS,
)

for tool_call in response.choices[0].message.tool_calls:
    result = dispatch(tool_call.function.name, tool_call.function.arguments)
    # Feed result back to LLM...
```

## License

GPL-3.0
