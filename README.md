# JiXing

Local AI Model Assistant with Long-term Memory System & Context Management.

A unified tool to run local AI models (Ollama, Moxing) with complete logging of all interactions, advanced context window management, semantic compression, session migration, and spatiotemporal memory retrieval.

## Features

- **Multi-Provider Support**: Run models from Ollama and Moxing with unified interface
- **Complete Logging**: Records all prompts, responses, timestamps, system info, and performance metrics
- **Long-term Memory**: SQLite-based storage for searching through all past interactions
- **Context Window Management**: Intelligent context overflow detection and multi-level compression
- **Semantic Compression**: Rule-based text compression for Chinese and English (150+ patterns)
- **Session Migration**: VM live migration-style session transfer with context preservation
- **Spatiotemporal Memory**: Memory linking by time, device, and semantic similarity
- **Conversation Archiving**: JSONL-based full session archiving and retrieval
- **Runtime State Persistence**: MD file state snapshots for long-running tasks
- **Web Interface**: Optional Flask-based web UI for browsing sessions and searching messages
- **CLI Interface**: Full-featured command-line tool with subcommands including model management
- **Python API**: Programmatic access via ToolResult pattern
- **OpenAI Function Calling**: Integrate with OpenAI agents via function tool definitions
- **Session Merging**: Combine multiple sessions with timeline or custom ordering
- **Streaming Responses**: Real-time streaming output from Ollama models

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

Run in interactive mode with context compression:
```bash
jixing ollama run gemma3:1b -i --compress
```

Run a Moxing (GGUF) model:
```bash
jixing moxing serve gguf-model-name "Your prompt here"
```

**Model Management**:
```bash
jixing ollama list                    # List available models
jixing ollama show gemma3:1b          # Show model information
jixing ollama pull llama3.2           # Pull a model
jixing ollama delete old-model        # Delete a model
```

**Session Management**:
```bash
jixing sessions list                  # List all sessions
jixing sessions list --provider ollama # Filter by provider
jixing sessions get <session-id>      # Get session details
jixing sessions delete <session-id>   # Delete a session
jixing sessions merge <id1> <id2> --mode timeline  # Merge sessions
```

Search through message history:
```bash
jixing search "previous conversation about Python"
```

Show statistics:
```bash
jixing stats
```

Show system information:
```bash
jixing info
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

**Context Management**:
```python
from jixing.api import (
    get_context_usage,
    compress_context,
    migrate_session,
    persist_runtime_state,
    auto_handle_overflow,
    archive_session,
    load_archived_session,
    list_archived_sessions,
)

# Check context usage
usage = get_context_usage(session_id="xxx")

# Compress context with target ratio
compressed = compress_context(
    session_id="xxx",
    target_ratio=0.5,  # 50% of original size
    goal="Fix the bug in core.py",
)

# Migrate session to new context
migration = migrate_session(
    session_id="xxx",
    new_context_limit=256000,
    goal="Continue the task",
)

# Archive and restore sessions
archive_session(session_id="xxx")
list_archived_sessions()
```

**Spatiotemporal Memory**:
```python
from jixing.memory import SpatiotemporalMemoryManager, SpatialContext

manager = SpatiotemporalMemoryManager.get_instance()

# Store memory with spatial context
spatial = SpatialContext.collect()
memory = manager.store_memory(
    session_id="xxx",
    content="Important decision about architecture",
    spatial_context=spatial,
)

# Retrieve memories
memories = manager.retrieve_memories(
    query="architecture decision",
    session_id="xxx",
    limit=10,
)

# Get linked memories (temporal, spatial, semantic)
links = manager.get_linked_memories(memory.memory_id)

# Get statistics
stats = manager.get_statistics()
```

**Session Merging**:
```python
from jixing.api import merge_sessions

merged = merge_sessions(
    session_ids=["id1", "id2", "id3"],
    merge_mode="timeline",  # or "reverse_timeline", "custom"
)
```

## Context Window Management

JiXing implements sophisticated context management inspired by VM live migration:

### Compression Strategies

| Strategy | Description | Best For |
|----------|-------------|----------|
| `preserve_critical` | Keeps only high-importance messages (score >= 7.0) | Maximum compression |
| `smart` | Preserves first user + last N messages, summarizes middle | Balanced quality |
| `progressive` | Progressive compression by importance score | Gradual reduction |
| `summary` | Full conversation summary | Extreme compression |
| `fallback` | Truncates middle messages | Last resort |

### Compression Levels

| Level | Ratio | Description |
|-------|-------|-------------|
| Light | 0.6 | Minimal compression, preserves most context |
| Medium | 0.4 | Moderate compression, good balance |
| Heavy | 0.2 | Strong compression, keeps essentials |
| Extreme | 0.1 | Maximum compression, summary only |

### Auto Overflow Handling

When context exceeds threshold, JiXing automatically:
1. Archives full history to JSONL
2. Attempts light compression (0.6 ratio)
3. Falls back to medium (0.4), heavy (0.2), extreme (0.1)
4. If at max limit, persists runtime state to MD file
5. Optionally migrates to new session with larger context

## Architecture

```
jixing/
├── jixing/
│   ├── __init__.py         # Package exports with lazy loading
│   ├── __version__.py      # Version info (1.0.4)
│   ├── core.py             # Core logic: Session, SessionManager, ModelAdapter
│   ├── api.py              # ToolResult API functions (15+ functions)
│   ├── cli.py              # CLI interface with full subcommand support
│   ├── db.py               # SQLite database for sessions and messages
│   ├── compressor.py       # Semantic text compression (ZH + EN rules)
│   ├── context_manager.py  # Context window management & session migration
│   ├── memory.py           # Spatiotemporal memory system
│   ├── models/             # Model adapters
│   ├── tools.py            # OpenAI function definitions + dispatch
│   ├── web.py              # Flask web interface
│   ├── static/             # Web static assets
│   └── templates/          # Web templates
├── tests/
├── pyproject.toml
└── README.md
```

## Data Storage

All data is stored in `~/.jixing/`:

| Path | Purpose |
|------|---------|
| `~/.jixing/sessions.db` | In-memory session cache persistence |
| `~/.jixing/jixing.db` | Main database for sessions and messages |
| `~/.jixing/spatiotemporal.db` | Spatiotemporal memory storage |
| `~/.jixing/conversations/*.jsonl` | Full session archives |
| `~/.jixing/context_archive/*.jsonl` | Compressed context archives |
| `~/.jixing/runtime_states/*.md` | Runtime state snapshots |

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

## CLI Help

```
$ jixing --help
usage: jixing [-h] [-V] [-v] [-o OUTPUT] [--json] [-q] {ollama,moxing,sessions,search,stats,info,web} ...

positional arguments:
  {ollama,moxing,sessions,search,stats,info,web}
    ollama              Ollama model management (run, list, show, pull, delete)
    moxing              Run Moxing model
    sessions            Manage sessions (list, get, delete, merge)
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

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Lint
ruff check .

# Type check
mypy jixing
```

## License

GPL-3.0
