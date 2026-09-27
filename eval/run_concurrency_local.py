"""Start the CUDA API, benchmark it over loopback HTTP, then stop it."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from urllib import error, request


def wait_for_server(process: subprocess.Popen, base_url: str, seconds: float) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"API exited during startup with code {process.returncode}")
        try:
            with request.urlopen(f"{base_url}/health", timeout=2) as response:
                health = json.load(response)
            if health.get("status") == "ok" and health.get("device") == "cuda":
                print(f"CUDA API ready on {health.get('gpu')}", flush=True)
                return
        except (OSError, ValueError, error.HTTPError):
            pass
        time.sleep(0.5)
    raise TimeoutError("CUDA API did not become ready")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--startup-timeout", type=float, default=120)
    args, benchmark_args = parser.parse_known_args()
    base_url = f"http://127.0.0.1:{args.port}"
    server = subprocess.Popen([
        sys.executable, "-m", "uvicorn", "sentiment_api.app:app",
        "--host", "127.0.0.1", "--port", str(args.port), "--workers", "1",
        "--no-access-log",
    ])
    try:
        wait_for_server(server, base_url, args.startup_timeout)
        return subprocess.run([
            sys.executable, "-m", "eval.concurrency_bench", "--base-url", base_url,
            *benchmark_args,
        ], check=False).returncode
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=10)


if __name__ == "__main__":
    raise SystemExit(main())
