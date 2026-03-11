"""Lightweight unittest discovery and summary output for jobman-lite."""

from __future__ import annotations

import sys
import traceback
import unittest
from pathlib import Path


def _iter_suite(suite: unittest.TestSuite):
    for test in suite:
        if isinstance(test, unittest.TestSuite):
            yield from _iter_suite(test)
        else:
            yield test


class SummaryResult(unittest.TextTestResult):
    """Collect pass/fail information for compact summary output."""

    def __init__(self, stream, descriptions, verbosity):
        super().__init__(stream, descriptions, verbosity)
        self.successes: list[str] = []
        self.failures_info: list[tuple[str, str, str]] = []

    def addSuccess(self, test):
        super().addSuccess(test)
        self.successes.append(test.id())

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self.failures_info.append(("FAIL", test.id(), self._exc_info_to_string(err, test)))

    def addError(self, test, err):
        super().addError(test, err)
        self.failures_info.append(("ERROR", test.id(), self._exc_info_to_string(err, test)))

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self.failures_info.append(("SKIP", test.id(), reason))


class SummaryRunner(unittest.TextTestRunner):
    resultclass = SummaryResult


def run_all_tests(stream=None) -> int:
    """Discover and run all tests with a compact summary."""
    stream = stream or sys.stdout
    tests_dir = Path(__file__).resolve().parent.parent / "tests"
    suite = unittest.defaultTestLoader.discover(str(tests_dir), pattern="test_*.py")
    discovered = list(_iter_suite(suite))

    if not discovered:
        stream.write("No tests discovered.\n")
        return 1

    stream.write("Discovered tests:\n")
    for test in discovered:
        stream.write(f"  {test.id()}\n")

    runner = SummaryRunner(stream=stream, verbosity=0)
    result: SummaryResult = runner.run(suite)

    stream.write("\nResults:\n")
    for test_id in result.successes:
        stream.write(f"  PASS  {test_id}\n")
    for status, test_id, detail in result.failures_info:
        stream.write(f"  {status:<5} {test_id}\n")
        if status in {"FAIL", "ERROR"}:
            last_line = detail.strip().splitlines()[-1] if detail.strip() else ""
            if last_line:
                stream.write(f"         {last_line}\n")
        else:
            stream.write(f"         {detail}\n")

    total = result.testsRun
    passed = len(result.successes)
    stream.write(f"\nSummary: {passed}/{total} tests passed.\n")
    return 0 if result.wasSuccessful() else 1
