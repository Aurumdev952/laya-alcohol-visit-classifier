"""Benchmark the local sentiment API at increasing client concurrency."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib import error, request


DEFAULT_CASES = Path(__file__).with_name("cases.jsonl")
DEFAULT_OUTPUT = Path(__file__).with_name("concurrency_results.json")


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = (len(ordered) - 1) * fraction
    low, high = math.floor(index), math.ceil(index)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def summarize_level(concurrency: int, measurements: list[dict], elapsed: float) -> dict:
    successes = [item["latency_ms"] for item in measurements if item["status"] == 200]
    errors = Counter(str(item["status"]) for item in measurements if item["status"] != 200)
    return {
        "concurrency": concurrency,
        "requests": len(measurements),
        "successes": len(successes),
        "errors": dict(errors),
        "elapsed_seconds": round(elapsed, 3),
        "throughput_rps": round(len(successes) / elapsed, 2) if elapsed else 0.0,
        "p50_ms": round(percentile(successes, 0.50), 2),
        "p95_ms": round(percentile(successes, 0.95), 2),
        "p99_ms": round(percentile(successes, 0.99), 2),
        "max_ms": round(max(successes), 2) if successes else 0.0,
    }


def practical_limit(levels: list[dict], p95_target_ms: float) -> int | None:
    qualified = [level["concurrency"] for level in levels
                 if not level["errors"] and level["successes"] == level["requests"]
                 and level["p95_ms"] <= p95_target_ms]
    return max(qualified) if qualified else None


def get_health(base_url: str, timeout: float, required_gpu: str) -> dict:
    with request.urlopen(f"{base_url}/health", timeout=timeout) as response:
        health = json.load(response)
    if health.get("status") != "ok" or health.get("device") != "cuda":
        raise RuntimeError(f"API is not healthy on CUDA: {health}")
    if required_gpu and required_gpu.lower() not in health.get("gpu", "").lower():
        raise RuntimeError(f"API GPU is {health.get('gpu')!r}; expected {required_gpu!r}")
    return health


def send_one(base_url: str, text: str, timeout: float) -> dict:
    payload = json.dumps({"text": text}).encode("utf-8")
    req = request.Request(f"{base_url}/sentiment", data=payload,
                          headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    try:
        with request.urlopen(req, timeout=timeout) as response:
            body = json.load(response)
            status = response.status if body.get("sentiment") else "invalid_body"
    except error.HTTPError as exc:
        status = exc.code
    except (error.URLError, TimeoutError, ValueError):
        status = "connection_error"
    return {"status": status, "latency_ms": (time.perf_counter() - start) * 1000}


def run_level(base_url: str, texts: list[str], concurrency: int,
              request_count: int, timeout: float) -> dict:
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(send_one, base_url, texts[i % len(texts)], timeout)
                   for i in range(request_count)]
        measurements = [future.result() for future in as_completed(futures)]
    return summarize_level(concurrency, measurements, time.perf_counter() - started)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--concurrency", default="1,2,4,8,16,32,64")
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--p95-target-ms", type=float, default=200.0)
    parser.add_argument("--require-gpu", default="5090")
    args = parser.parse_args()
    try:
        concurrencies = [int(item) for item in args.concurrency.split(",")]
        if not concurrencies or any(n < 1 for n in concurrencies) or len(set(concurrencies)) != len(concurrencies):
            raise ValueError("concurrency levels must be unique positive integers")
        if args.requests < max(concurrencies) or args.warmup < 0 or args.p95_target_ms <= 0:
            raise ValueError("requests must cover the largest concurrency; warmup and target must be valid")
        base_url = args.base_url.rstrip("/")
        texts = [json.loads(line)["text"] for line in args.cases.read_text(encoding="utf-8").splitlines()
                 if line.strip()]
        if not texts:
            raise ValueError("no request texts found")
        health = get_health(base_url, args.timeout, args.require_gpu)
        for i in range(args.warmup):
            result = send_one(base_url, texts[i % len(texts)], args.timeout)
            if result["status"] != 200:
                raise RuntimeError(f"warmup request failed: {result['status']}")

        levels = []
        for concurrency in concurrencies:
            level = run_level(base_url, texts, concurrency, args.requests, args.timeout)
            levels.append(level)
            get_health(base_url, args.timeout, args.require_gpu)
            print(f"c={concurrency}: {level['throughput_rps']:.1f} req/s, "
                  f"p95={level['p95_ms']:.1f} ms, errors={sum(level['errors'].values())}", flush=True)

        best = max(levels, key=lambda item: item["throughput_rps"])
        output = {
            "model": health.get("model"), "gpu": health["gpu"],
            "endpoint": f"{base_url}/sentiment",
            "workload": {"texts": len(texts), "requests_per_level": args.requests,
                         "warmup_requests": args.warmup},
            "p95_target_ms": args.p95_target_ms,
            "practical_concurrency_at_target": practical_limit(levels, args.p95_target_ms),
            "highest_tested_error_free": max((level["concurrency"] for level in levels if not level["errors"]),
                                              default=None),
            "peak_throughput_level": best["concurrency"],
            "peak_throughput_rps": best["throughput_rps"],
            "levels": levels,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        parser.exit(1, f"Concurrency benchmark failed: {exc}\n")

    print(f"Practical concurrency at p95 <= {args.p95_target_ms:g} ms: "
          f"{output['practical_concurrency_at_target']}")
    print(f"Report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

