"""
Baseline benchmarks — these run from Day 2 onward.
Every optimization you make will be measured against these numbers.
Run with: make test-bench
"""


def test_config_import_speed(benchmark):
    """How fast does settings load? Baseline for config overhead."""

    def load_settings():
        from config.settings import settings

        return settings

    result = benchmark(load_settings)
    assert result is not None


def test_logger_import_speed(benchmark):
    """How fast does the logger initialize?"""

    def get_log():
        from config.logging_config import get_logger

        return get_logger("benchmark")

    result = benchmark(get_log)
    assert result is not None


def test_pure_python_loop_baseline(benchmark):
    """
    Baseline: pure Python loop over 10K items.
    When we add NumPy vectorization in Week 2, we compare against this.
    """

    def python_sum():
        return sum(i * 2 for i in range(10_000))

    result = benchmark(python_sum)
    assert result == 99_990_000
