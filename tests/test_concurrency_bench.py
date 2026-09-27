import unittest

from eval.concurrency_bench import practical_limit, summarize_level


class ConcurrencyBenchTests(unittest.TestCase):
    def test_level_metrics_and_latency_target(self):
        first = summarize_level(2, [{"status": 200, "latency_ms": 20},
                                    {"status": 200, "latency_ms": 40}], 0.1)
        second = summarize_level(4, [{"status": 200, "latency_ms": 300},
                                     {"status": 503, "latency_ms": 5}], 0.1)
        self.assertEqual(first["throughput_rps"], 20.0)
        self.assertEqual(first["p95_ms"], 39.0)
        self.assertEqual(second["errors"], {"503": 1})
        self.assertEqual(practical_limit([first, second], 100), 2)


if __name__ == "__main__":
    unittest.main()

