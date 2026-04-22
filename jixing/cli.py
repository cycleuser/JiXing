import argparse
import json
import logging
import sys
from pathlib import Path

from .api import (
    delete_session,
    get_session,
    get_stats,
    get_system_info,
    merge_sessions,
    query_sessions,
    run_moxing,
    run_ollama,
    search_messages,
)
from .core import OllamaAdapter, Session, SessionManager
from .compressor import SemanticCompressor


def _get_version():
    from importlib import import_module

    return import_module("jixing").__version__


logger = logging.getLogger(__name__)


def setup_logging(verbose: bool, quiet: bool):
    level = logging.INFO
    if quiet:
        level = logging.WARNING
    elif verbose:
        level = logging.DEBUG
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def parse_args():
    parser = argparse.ArgumentParser(
        prog="jixing",
        description="Local AI Model Assistant with Long-term Memory System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  jixing ollama run gemma3:1b "Say hello"
  jixing ollama run gemma3:1b -i
  jixing ollama list
  jixing ollama show gemma3:1b
  jixing ollama pull llama3.2
  jixing ollama delete old-model
  jixing moxing serve gguf_model
  jixing sessions list --provider ollama
  jixing search "previous conversation about"
  jixing stats
  jixing info

Commands:
  ollama       Ollama model management (run, list, show, pull, delete)
  moxing       Run Moxing model
  sessions     Manage sessions
  search       Search message history
  stats        Show statistics
  info         Show system information
  web          Start web interface
        """,
    )

    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"jixing {_get_version()}",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output path",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output results as JSON",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress non-essential output",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    ollama_parser = subparsers.add_parser("ollama", help="Ollama model management")
    ollama_sub = ollama_parser.add_subparsers(dest="subcommand", required=True)

    run_parser = ollama_sub.add_parser("run", help="Run a model")
    run_parser.add_argument("model", help="Model name (e.g., gemma3:1b)")
    run_parser.add_argument("prompt", nargs="*", help="Prompt text")
    run_parser.add_argument("--session", help="Session ID to continue")
    run_parser.add_argument("-i", "--interactive", action="store_true", help="Interactive mode")
    run_parser.add_argument("--base-url", default="http://localhost:11434", help="Ollama base URL")
    run_parser.add_argument("--compress", action="store_true", help="Enable context compression")

    list_parser = ollama_sub.add_parser("list", help="List available models")
    list_parser.add_argument("--base-url", default="http://localhost:11434", help="Ollama base URL")
    list_parser.add_argument("--json", action="store_true", dest="json_output", help="JSON output")

    show_parser = ollama_sub.add_parser("show", help="Show model information")
    show_parser.add_argument("model", help="Model name")
    show_parser.add_argument("--base-url", default="http://localhost:11434", help="Ollama base URL")
    show_parser.add_argument("--json", action="store_true", dest="json_output", help="JSON output")

    pull_parser = ollama_sub.add_parser("pull", help="Pull a model")
    pull_parser.add_argument("model", help="Model name to pull")
    pull_parser.add_argument("--base-url", default="http://localhost:11434", help="Ollama base URL")

    delete_parser = ollama_sub.add_parser("delete", help="Delete a model")
    delete_parser.add_argument("model", help="Model name to delete")
    delete_parser.add_argument("--base-url", default="http://localhost:11434", help="Ollama base URL")

    moxing_parser = subparsers.add_parser("moxing", help="Run Moxing model")
    moxing_sub = moxing_parser.add_subparsers(dest="subcommand", required=True)

    serve_parser = moxing_sub.add_parser("serve", help="Serve a GGUF model")
    serve_parser.add_argument("model", help="Model name or file path")
    serve_parser.add_argument("prompt", nargs="*", help="Prompt text")
    serve_parser.add_argument("--session", help="Session ID to continue")
    serve_parser.add_argument("-i", "--interactive", action="store_true", help="Interactive mode")

    ollama_serve_parser = moxing_sub.add_parser("ollama", help="Use Ollama backend via Moxing")
    ollama_serve_parser.add_argument("subcommand2", choices=["serve"], help="Serve command")
    ollama_serve_parser.add_argument("model", help="Model name")
    ollama_serve_parser.add_argument("prompt", nargs="*", help="Prompt text")
    ollama_serve_parser.add_argument("--session", help="Session ID to continue")
    ollama_serve_parser.add_argument(
        "-i", "--interactive", action="store_true", help="Interactive mode"
    )

    sessions_parser = subparsers.add_parser("sessions", help="Manage sessions")
    sessions_sub = sessions_parser.add_subparsers(dest="subcommand", required=True)

    list_parser = sessions_sub.add_parser("list", help="List sessions")
    list_parser.add_argument("--provider", help="Filter by provider (ollama, moxing)")
    list_parser.add_argument("--model", help="Filter by model name")
    list_parser.add_argument("--limit", type=int, default=100, help="Max results")

    get_parser = sessions_sub.add_parser("get", help="Get session details")
    get_parser.add_argument("session_id", help="Session ID")

    del_parser = sessions_sub.add_parser("delete", help="Delete a session")
    del_parser.add_argument("session_id", help="Session ID")

    merge_parser = sessions_sub.add_parser("merge", help="Merge multiple sessions")
    merge_parser.add_argument("session_ids", nargs="+", help="Session IDs to merge")
    merge_parser.add_argument(
        "--mode",
        choices=["timeline", "reverse_timeline", "custom"],
        default="timeline",
        help="Merge mode",
    )
    merge_parser.add_argument("--provider", help="Model provider for merged session")
    merge_parser.add_argument("--model", help="Model name for merged session")

    search_parser = subparsers.add_parser("search", help="Search messages")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--session", help="Session ID to search within")
    search_parser.add_argument("--provider", help="Filter by provider")
    search_parser.add_argument("--limit", type=int, default=100, help="Max results")

    subparsers.add_parser("stats", help="Show statistics")

    subparsers.add_parser("info", help="Show system information")

    web_parser = subparsers.add_parser("web", help="Start web interface")
    web_parser.add_argument("--host", default="127.0.0.1", help="Host to bind")
    web_parser.add_argument("--port", type=int, default=5000, help="Port to bind")

    return parser.parse_args()


def handle_ollama_run(args) -> int:
    prompt = " ".join(args.prompt) if args.prompt else None
    base_url = getattr(args, "base_url", "http://localhost:11434")
    use_compress = getattr(args, "compress", False)
    session_id = getattr(args, "session", None)

    manager = SessionManager.get_instance()
    session = None
    if session_id:
        session = manager.get_session(session_id)

    if not session:
        session = manager.create_session(
            model_provider="ollama",
            model_name=args.model,
        )
        session_id = session.id

    adapter = OllamaAdapter(session, base_url=base_url)
    compressor = SemanticCompressor()

    conversation_history = [m for m in session.messages]

    if args.interactive:
        print(f"Interactive mode with {args.model}. Type 'exit' to quit, 'clear' to clear history.")
        if use_compress:
            print("Context compression enabled.")
        print(f"Session: {session_id[:8]}...")
        print()

        while True:
            try:
                user_input = input("\nYou: ")
                if user_input.lower() in ("exit", "quit"):
                    break
                if user_input.lower() == "clear":
                    conversation_history = []
                    print("Conversation history cleared.")
                    continue
                if not user_input.strip():
                    continue

                messages = conversation_history + [{"role": "user", "content": user_input}]

                if use_compress and len(conversation_history) > 4:
                    compressed = compressor.compress_messages(conversation_history)
                    if len(compressed) < len(conversation_history):
                        logger.debug(f"Compressed {len(conversation_history)} -> {len(compressed)} messages")
                        conversation_history = compressed

                messages_to_send = conversation_history + [{"role": "user", "content": user_input}]

                print("\nAssistant: ", end="", flush=True)
                full_response = ""
                try:
                    for chunk, done, metrics in adapter.run_stream(
                        prompt=None, history=messages_to_send
                    ):
                        if chunk:
                            print(chunk, end="", flush=True)
                            full_response += chunk
                    print()
                except Exception as e:
                    print(f"\nError: {e}", file=sys.stderr)
                    continue

                session.add_message("user", user_input)
                session.add_message("assistant", full_response)
                manager._save_session(session)

                conversation_history.append({"role": "user", "content": user_input})
                conversation_history.append({"role": "assistant", "content": full_response})

            except KeyboardInterrupt:
                print("\nExiting...")
                break

    elif prompt:
        print("\nAssistant: ", end="", flush=True)
        full_response = ""
        try:
            messages = conversation_history + [{"role": "user", "content": prompt}]
            for chunk, done, metrics in adapter.run_stream(
                prompt=None, history=messages
            ):
                if chunk:
                    print(chunk, end="", flush=True)
                    full_response += chunk
            print()
        except Exception as e:
            print(f"\nError: {e}", file=sys.stderr)
            return 1

        session.add_message("user", prompt)
        session.add_message("assistant", full_response)
        manager._save_session(session)
    else:
        print(
            "Error: Prompt required. Use -i for interactive mode or provide prompt text.",
            file=sys.stderr,
        )
        return 1

    return 0


def handle_ollama_list(args) -> int:
    base_url = getattr(args, "base_url", "http://localhost:11434")
    adapter = OllamaAdapter(
        Session(id="temp", model_provider="ollama", model_name="", created_at="", updated_at="", system_info={}),
        base_url=base_url,
    )

    try:
        models = adapter.list_models()
    except Exception as e:
        print(f"Error connecting to Ollama: {e}", file=sys.stderr)
        return 1

    if getattr(args, "json_output", False):
        print(json.dumps({"models": models}))
    else:
        if not models:
            print("No models found. Pull a model with: jixing ollama pull <model>")
        else:
            print(f"Found {len(models)} models:\n")
            print(f"{'NAME':<45} {'ID':<15} {'SIZE':<10} {'MODIFIED'}")
            print("-" * 90)
            for m in models:
                name = m.get("name", "")
                digest = m.get("digest", "")[:12]
                size = m.get("size", 0)
                size_str = _format_size(size)
                modified = m.get("modified_at", "")[:19] if m.get("modified_at") else ""
                print(f"{name:<45} {digest:<15} {size_str:<10} {modified}")

    return 0


def handle_ollama_show(args) -> int:
    base_url = getattr(args, "base_url", "http://localhost:11434")
    adapter = OllamaAdapter(
        Session(id="temp", model_provider="ollama", model_name="", created_at="", updated_at="", system_info={}),
        base_url=base_url,
    )

    try:
        info = adapter.show_model(args.model)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if getattr(args, "json_output", False):
        print(json.dumps(info))
    else:
        print(f"Model: {args.model}")
        print("=" * 40)
        if "details" in info:
            details = info["details"]
            print(f"Format: {details.get('format', 'N/A')}")
            print(f"Family: {details.get('family', 'N/A')}")
            print(f"Parameter Size: {details.get('parameter_size', 'N/A')}")
            print(f"Quantization: {details.get('quantization_level', 'N/A')}")
        if "modelfile" in info:
            print(f"\nModelfile:\n{info['modelfile'][:500]}...")

    return 0


def handle_ollama_pull(args) -> int:
    base_url = getattr(args, "base_url", "http://localhost:11434")
    adapter = OllamaAdapter(
        Session(id="temp", model_provider="ollama", model_name="", created_at="", updated_at="", system_info={}),
        base_url=base_url,
    )

    print(f"Pulling {args.model}...")
    try:
        for status in adapter.pull_model(args.model, stream=True):
            if "status" in status:
                print(f"  {status['status']}")
            if "completed" in status and "total" in status:
                pct = status["completed"] / status["total"] * 100
                print(f"\r  Downloading: {pct:.1f}%", end="", flush=True)
        print()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    return 0


def handle_ollama_delete(args) -> int:
    base_url = getattr(args, "base_url", "http://localhost:11434")
    adapter = OllamaAdapter(
        Session(id="temp", model_provider="ollama", model_name="", created_at="", updated_at="", system_info={}),
        base_url=base_url,
    )

    try:
        success = adapter.delete_model(args.model)
        if success:
            print(f"Deleted model: {args.model}")
        else:
            print(f"Failed to delete model: {args.model}", file=sys.stderr)
            return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    return 0


def handle_moxing_serve(args, provider_override: str = "moxing") -> int:
    prompt = " ".join(args.prompt) if args.prompt else None

    if args.interactive:
        print("Interactive mode. Type 'exit' to quit.")
        while True:
            try:
                prompt = input("\nYou: ")
                if prompt.lower() in ("exit", "quit"):
                    break
                if not prompt.strip():
                    continue

                result = run_moxing(model=args.model, prompt=prompt, session_id=args.session)

                if args.json_output:
                    print(json.dumps(result.to_dict()))
                else:
                    if result.success:
                        print(f"\nAssistant: {result.data['response']}")
                        print(f"[Tokens: {result.data['metrics'].get('tokens', 'N/A')}]")
                    else:
                        print(f"Error: {result.error}")

                args.session = result.data["session_id"] if result.success else args.session
            except KeyboardInterrupt:
                print("\nExiting...")
                break
    elif prompt:
        result = run_moxing(model=args.model, prompt=prompt, session_id=args.session)
        if args.json_output:
            print(json.dumps(result.to_dict()))
        else:
            if result.success:
                print(result.data["response"])
            else:
                print(f"Error: {result.error}", file=sys.stderr)
                return 1
    else:
        print(
            "Error: Prompt required. Use -i for interactive mode or provide prompt text.",
            file=sys.stderr,
        )
        return 1

    return 0


def handle_sessions(args) -> int:
    if args.subcommand == "list":
        result = query_sessions(
            model_provider=args.provider,
            model_name=args.model,
            limit=args.limit,
        )
        if args.json_output:
            print(json.dumps(result.to_dict()))
        else:
            if result.success:
                sessions = result.data
                if not sessions:
                    print("No sessions found.")
                else:
                    print(f"Found {len(sessions)} sessions:\n")
                    for s in sessions:
                        sid = s["id"][:8]
                        prov = s["model_provider"]
                        model = s["model_name"]
                        created = s["created_at"][:19]
                        msgs = len(s.get("messages", []))
                        tokens = s.get("total_tokens", 0)
                        print(f"  {sid}... | {prov}/{model} | {created} | {msgs} msgs | {tokens} tokens")
            else:
                print(f"Error: {result.error}", file=sys.stderr)
                return 1

    elif args.subcommand == "get":
        result = get_session(session_id=args.session_id)
        if args.json_output:
            print(json.dumps(result.to_dict()))
        else:
            if result.success:
                s = result.data
                print(f"Session: {s['id']}")
                print(f"Provider: {s['model_provider']}")
                print(f"Model: {s['model_name']}")
                print(f"Created: {s['created_at']}")
                print(f"Messages: {len(s['messages'])}")
                print(f"Total tokens: {s['total_tokens']}")
                if s.get("parent_session_id"):
                    print(f"Parent session: {s['parent_session_id'][:8]}...")
                if s.get("goal"):
                    print(f"Goal: {s['goal']}")
            else:
                print(f"Error: {result.error}", file=sys.stderr)
                return 1

    elif args.subcommand == "delete":
        result = delete_session(session_id=args.session_id)
        if args.json_output:
            print(json.dumps(result.to_dict()))
        else:
            if result.success:
                print(f"Deleted session {args.session_id}")
            else:
                print(f"Error: {result.error}", file=sys.stderr)
                return 1

    elif args.subcommand == "merge":
        result = merge_sessions(
            session_ids=args.session_ids,
            merge_mode=args.mode,
            model_provider=args.provider,
            model_name=args.model,
        )
        if args.json_output:
            print(json.dumps(result.to_dict()))
        else:
            if result.success:
                s = result.data
                print(
                    f"Merged {result.metadata.get('merged_count', len(args.session_ids))} sessions"
                )
                print(f"New session ID: {s['id']}")
                print(f"Provider: {s['model_provider']}")
                print(f"Model: {s['model_name']}")
                print(f"Messages: {len(s['messages'])}")
                print(f"Total tokens: {s['total_tokens']}")
            else:
                print(f"Error: {result.error}", file=sys.stderr)
                return 1

    return 0


def handle_search(args) -> int:
    result = search_messages(
        query=args.query,
        session_id=args.session,
        model_provider=args.provider,
        limit=args.limit,
    )
    if args.json_output:
        print(json.dumps(result.to_dict()))
    else:
        if result.success:
            messages = result.data
            if not messages:
                print("No messages found.")
            else:
                print(f"Found {len(messages)} messages:\n")
                for m in messages:
                    role = m["role"].upper()
                    ts = m["timestamp"][:19]
                    content = (
                        m["content"][:100] + "..." if len(m["content"]) > 100 else m["content"]
                    )
                    print(f"  [{ts}] {role} ({m['model_provider']}/{m['model_name'][:20]}):")
                    print(f"    {content}\n")
        else:
            print(f"Error: {result.error}", file=sys.stderr)
            return 1

    return 0


def handle_stats(args) -> int:
    result = get_stats()
    if args.json_output:
        print(json.dumps(result.to_dict()))
    else:
        if result.success:
            s = result.data
            print("JiXing Statistics")
            print("=" * 40)
            print(f"Total Sessions: {s['total_sessions']}")
            print(f"Total Messages: {s['total_messages']}")
            print(f"Total Tokens: {s['total_tokens']}")
            print(f"Total Duration: {s['total_duration_ms']}ms")
            print("\nSessions by Provider:")
            for provider, count in s["sessions_by_provider"].items():
                print(f"  {provider}: {count}")
        else:
            print(f"Error: {result.error}", file=sys.stderr)
            return 1

    return 0


def handle_info(args) -> int:
    result = get_system_info()
    if args.json_output:
        print(json.dumps(result.to_dict()))
    else:
        if result.success:
            info = result.data
            print("System Information")
            print("=" * 40)
            print(f"Hostname: {info['hostname']}")
            print(f"OS: {info['os_system']} {info['os_release']}")
            print(f"Python: {info['python_version']}")
            print(f"Platform: {info['platform_machine']}")
            print(f"IP: {info['network_ip']}")
            print(f"CWD: {info['cwd']}")
        else:
            print(f"Error: {result.error}", file=sys.stderr)
            return 1

    return 0


def handle_web(args) -> int:
    from .web import create_app

    app = create_app()
    print(f"Starting web interface at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port)
    return 0


def _format_size(size_bytes: int) -> str:
    """Format bytes to human readable size."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.0f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.0f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"


def main():
    args = parse_args()
    setup_logging(args.verbose, args.quiet)

    try:
        if args.command == "ollama":
            if args.subcommand == "run":
                return handle_ollama_run(args)
            elif args.subcommand == "list":
                return handle_ollama_list(args)
            elif args.subcommand == "show":
                return handle_ollama_show(args)
            elif args.subcommand == "pull":
                return handle_ollama_pull(args)
            elif args.subcommand == "delete":
                return handle_ollama_delete(args)

        elif args.command == "moxing":
            if args.subcommand == "serve":
                return handle_moxing_serve(args, "moxing")
            elif (
                args.subcommand == "ollama"
                and hasattr(args, "subcommand2")
                and args.subcommand2 == "serve"
            ):
                return handle_moxing_serve(args, "ollama")

        elif args.command == "sessions":
            return handle_sessions(args)

        elif args.command == "search":
            return handle_search(args)

        elif args.command == "stats":
            return handle_stats(args)

        elif args.command == "info":
            return handle_info(args)

        elif args.command == "web":
            return handle_web(args)

        else:
            print(f"Unknown command: {args.command}", file=sys.stderr)
            return 1

    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except Exception as e:
        logger.exception("Unexpected error")
        print(f"Error: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
