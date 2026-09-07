"""Minimal dependency-free test harness for v2 (no pytest in this project).

Each test file builds a Harness, calls .case(name, fn) for each test, and
returns .done() as the process exit code. A test function receives the harness
and reports checks via .expect(...) / .expect_raises(...).
"""

import os
import sys
import traceback

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


class Harness:
    def __init__(self, title: str):
        self.title = title
        self.passed = 0
        self.failed = 0
        print(f"=== {title} ===")

    def expect(self, name: str, cond: bool, detail: str = "") -> None:
        if cond:
            self.passed += 1
            print(f"  PASS  {name}")
        else:
            self.failed += 1
            suffix = f"  [{detail}]" if detail else ""
            print(f"  FAIL  {name}{suffix}")

    def expect_raises(self, name: str, exc_type, fn) -> None:
        try:
            fn()
        except exc_type:
            self.passed += 1
            print(f"  PASS  {name}")
        except Exception as e:
            self.failed += 1
            print(f"  FAIL  {name}  [wrong exception {type(e).__name__}: {e}]")
        else:
            self.failed += 1
            print(f"  FAIL  {name}  [no exception raised]")

    def case(self, name: str, fn) -> None:
        try:
            fn(self)
        except Exception as e:
            self.failed += 1
            print(f"  FAIL  {name}  [uncaught {type(e).__name__}: {e}]")
            traceback.print_exc()

    def done(self) -> int:
        total = self.passed + self.failed
        status = "OK" if self.failed == 0 else "FAILED"
        print(f"--- {self.title}: {self.passed}/{total} passed  [{status}]\n")
        return 0 if self.failed == 0 else 1
