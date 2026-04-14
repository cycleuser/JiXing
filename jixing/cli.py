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
  jixing ollama run gemma3:1b
  jixing moxing serve gguf_model
  jixing moxing ollama serve gemma3:1b
  jixing sessions list --provider ollama
  jixing search "previous conversation about"
  jixing stats

Commands:
  ollama       Run Ollama model
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

    ollama_parser = subparsers.add_parser("ollama", help="Run Ollama model")
    ollama_sub = ollama_parser.add_subparsers(dest="subcommand", required=True)

    run_parser = ollama_sub.add_parser("run", help="Run a model")
    run_parser.add_argument("model", help="Model name (e.g., gemma3:1b)")
    run_parser.add_argument("prompt", nargs="*", help="Prompt text")
    run_parser.add_argument("--session", help="Session ID to continue")
    run_parser.add_argument("-i", "--interactive", action="store_true", help="Interactive mode")

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

    if args.interactive:
        print("Interactive mode. Type 'exit' to quit.")
        while True:
            try:
                prompt = input("\nYou: ")
                if prompt.lower() in ("exit", "quit"):
                    break
                if not prompt.strip():
                    continue

                result = run_ollama(model=args.model, prompt=prompt, session_id=args.session)

                if args.json_output:
                    print(json.dumps(result.to_dict()))
                else:
                    if result.success:
                        print(f"\nAssistant: {result.data['response']}")
                        print(f"[Tokens: {result.data['metrics'].get('eval_count', 'N/A')}]")
                    else:
                        print(f"Error: {result.error}")

                args.session = result.data["session_id"] if result.success else args.session
            except KeyboardInterrupt:
                print("\nExiting...")
                break
    elif prompt:
        result = run_ollama(model=args.model, prompt=prompt, session_id=args.session)
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
                        print(f"  {sid}... | {prov}/{model} | {created}")
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


def main():
    args = parse_args()
    setup_logging(args.verbose, args.quiet)

    try:
        if args.command == "ollama" and args.subcommand == "run":
            return handle_ollama_run(args)

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
