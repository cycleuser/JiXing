#!/usr/bin/env python3
"""Build and publish JiXing to PyPI.

Usage:
    python publish.py          # Build only
    python publish.py test     # Build + upload to TestPyPI
    python publish.py release  # Build + upload to PyPI
"""

import subprocess
import sys
import shutil
from pathlib import Path

ROOT = Path(__file__).parent
DIST = ROOT / "dist"


def run(cmd, **kwargs):
    print(f"\n>>> {cmd}")
    result = subprocess.run(cmd, shell=True, **kwargs)
    if result.returncode != 0:
        print(f"FAILED (exit {result.returncode})")
        sys.exit(result.returncode)
    return result


def ensure_tools():
    for pkg in ("build", "twine"):
        try:
            __import__(pkg)
        except ImportError:
            print(f"Installing {pkg}...")
            run(f"{sys.executable} -m pip install {pkg}")


def clean():
    for d in (DIST, ROOT / "build", ROOT / "jixing.egg-info"):
        if d.exists():
            print(f"Removing {d}")
            shutil.rmtree(d)


def build():
    clean()
    run(f"{sys.executable} -m build")
    wheels = list(DIST.glob("*.whl"))
    tarballs = list(DIST.glob("*.tar.gz"))
    print(f"\nBuilt: {[f.name for f in wheels + tarballs]}")


def upload(repository=None):
    cmd = f"{sys.executable} -m twine upload"
    if repository:
        cmd += f" --repository {repository}"
    cmd += " dist/*"
    run(cmd)


def main():
    ensure_tools()

    action = sys.argv[1] if len(sys.argv) > 1 else "build"

    if action == "build":
        build()
        print("\nBuild complete. To upload run:")
        print("  python publish.py test      # TestPyPI")
        print("  python publish.py release   # PyPI")

    elif action == "test":
        build()
        print("\nUploading to TestPyPI...")
        upload("testpypi")
        print("\nDone! Install with:")
        print("  pip install --index-url https://test.pypi.org/simple/ jixing")

    elif action == "release":
        build()
        print("\nUploading to PyPI...")
        upload()
        print("\nDone! Install with:")
        print("  pip install jixing")

    else:
        print(f"Unknown action: {action}")
        print("Usage: python publish.py [build|test|release]")
        sys.exit(1)


if __name__ == "__main__":
    main()
