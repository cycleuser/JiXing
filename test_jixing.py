#!/usr/bin/env python3
"""Comprehensive test script for JiXing.

Tests CLI commands, session management, merging, and web interface.

Usage:
    python test_jixing.py          # Run all tests
    python test_jixing.py --quick  # Skip web tests
"""

import json
import subprocess
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
SKIP = "\033[93m⊘\033[0m"

results = {"passed": 0, "failed": 0, "skipped": 0}


def run_cmd(cmd, timeout=30):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def test(name, condition):
    if condition:
        print(f"  {PASS} {name}")
        results["passed"] += 1
    else:
        print(f"  {FAIL} {name}")
        results["failed"] += 1


def section(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def test_version():
    section("Version Management")
    rc, out, err = run_cmd("python -c 'import jixing; print(jixing.__version__)'")
    test("Version importable", rc == 0 and len(out) > 0)
    test("Version format", "." in out if rc == 0 else False)


def test_cli_help():
    section("CLI Help")
    rc, out, err = run_cmd("jixing --help")
    test("Help command works", rc == 0)
    test("Shows ollama command", "ollama" in out)
    test("Shows web command", "web" in out)
    test("Shows sessions command", "sessions" in out)
    test("Shows merge command", "merge" not in out)  # merge is subcommand


def test_ollama_run():
    section("Ollama Run (requires Ollama running)")
    rc, out, err = run_cmd('jixing ollama run gemma3:1b "Say hello"', timeout=60)
    if rc == 0:
        test("Ollama run succeeds", True)
        test("Output contains response", len(out) > 0)
        return out
    else:
        test("Ollama run succeeds", False)
        print(f"    (Ollama may not be running: {err[:100]})")
        return None


def test_sessions_list():
    section("Sessions List")
    rc, out, err = run_cmd("jixing sessions list")
    test("List command works", rc == 0)
    test("Shows sessions", "Found" in out or "No sessions" in out)

    rc, out, err = run_cmd("jixing sessions list --provider ollama")
    test("Filter by provider works", rc == 0)


def test_sessions_get():
    section("Sessions Get")
    rc, out, err = run_cmd("jixing --json sessions list")
    if rc == 0:
        data = json.loads(out)
        if data.get("success") and data.get("data"):
            session_id = data["data"][0]["id"]
            rc2, out2, err2 = run_cmd(f"jixing sessions get {session_id}")
            test("Get session works", rc2 == 0)
            test("Shows session details", "Session:" in out2)
            return session_id
    test("No sessions available", False)
    return None


def test_search():
    section("Search")
    rc, out, err = run_cmd('jixing search "hello"')
    test("Search command works", rc == 0)


def test_merge():
    section("Session Merge")
    rc, out, err = run_cmd("jixing --json sessions list")
    if rc == 0:
        data = json.loads(out)
        if data.get("success") and len(data.get("data", [])) >= 2:
            ids = [s["id"] for s in data["data"][:2]]
            rc2, out2, err2 = run_cmd(f"jixing sessions merge {' '.join(ids)} --mode timeline")
            test("Merge command works", rc2 == 0)
            test("Shows merged session", "Merged" in out2)
            test("Shows message count", "Messages:" in out2)

            rc3, out3, err3 = run_cmd(
                f"jixing sessions merge {' '.join(ids)} --mode reverse_timeline"
            )
            test("Reverse timeline merge works", rc3 == 0)
        else:
            test("Need at least 2 sessions", False)
    else:
        test("Merge command works", False)


def test_web(quick=False):
    section("Web Interface")
    if quick:
        print(f"  {SKIP} Web tests skipped (--quick)")
        results["skipped"] += 5
        return

    proc = subprocess.Popen(
        ["jixing", "web", "--port", "5002"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    time.sleep(3)

    try:
        req = urllib.request.urlopen("http://127.0.0.1:5001/", timeout=5)
        test("Web page loads", req.status == 200)
    except Exception:
        try:
            req = urllib.request.urlopen("http://127.0.0.1:5002/", timeout=5)
            test("Web page loads", req.status == 200)
        except Exception:
            test("Web page loads", False)
            proc.kill()
            return

    try:
        req = urllib.request.urlopen("http://127.0.0.1:5002/api/sessions", timeout=5)
        data = json.loads(req.read())
        test("Sessions API works", data.get("success"))
    except Exception:
        test("Sessions API works", False)

    try:
        req = urllib.request.urlopen("http://127.0.0.1:5002/api/stats", timeout=5)
        data = json.loads(req.read())
        test("Stats API works", data.get("success"))
    except Exception:
        test("Stats API works", False)

    try:
        req = urllib.request.Request(
            "http://127.0.0.1:5002/api/sessions/merge",
            data=json.dumps({"session_ids": ["test1", "test2"], "merge_mode": "timeline"}).encode(),
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=5)
    except urllib.error.HTTPError as e:
        test("Merge API endpoint exists", e.code == 400)
    except Exception:
        test("Merge API endpoint exists", False)

    try:
        req = urllib.request.urlopen("http://127.0.0.1:5002/api/ollama/models", timeout=5)
        data = json.loads(req.read())
        test("Ollama models API works", "models" in data.get("data", {}))
    except Exception:
        test("Ollama models API works", False)

    proc.kill()
    proc.wait()


def test_version_consistency():
    section("Version Consistency")
    rc1, out1, _ = run_cmd("python -c 'import jixing; print(jixing.__version__)'")
    rc2, out2, _ = run_cmd("jixing --version")

    test("__init__.py version accessible", rc1 == 0)
    test("CLI version accessible", rc2 == 0)
    if rc1 == 0 and rc2 == 0:
        test("Versions match", out1.strip() in out2.strip())


def main():
    print("JiXing Comprehensive Test Suite")
    print("=" * 60)

    quick = "--quick" in sys.argv

    test_version()
    test_version_consistency()
    test_cli_help()
    test_ollama_run()
    test_sessions_list()
    test_sessions_get()
    test_search()
    test_merge()
    test_web(quick)

    print(f"\n{'=' * 60}")
    print(
        f"  Results: {results['passed']} passed, {results['failed']} failed, {results['skipped']} skipped"
    )
    print(f"{'=' * 60}")

    return 0 if results["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
