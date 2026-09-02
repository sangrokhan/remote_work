#!/usr/bin/env python3
"""Run the pytest suite, merge the result with the latest STD (docs/std/*.json)
snapshot, and overwrite test-results/latest.json.

Usage:
    .venv/bin/python scripts/save_test_results.py [pytest args...] [--commit] [--no-push]

Behaviour:
  - Runs `pytest -m "not live" <pytest args>` (default target: the whole
    `tests/` dir, matching the invocation documented in
    docs/srs-jira-tickets-and-std-2026-09-01.md) with the pytest-json-report
    plugin, capturing a pass/fail summary.
  - Picks the *latest* STD snapshot under docs/std/std_*.json (by filename,
    which is date-stamped) and embeds it verbatim under the "std" key, so the
    STD-vs-actual comparison always travels together.
  - Overwrites test-results/latest.json (single file, no history kept in the
    JSON itself -- history lives in `git log -- test-results/latest.json`).
  - With --commit, stages + commits the file (and pushes unless --no-push).
    Without --commit, the script only writes the file locally.

Exit code mirrors pytest's exit code, so this is safe to use as a CI/local
gate as well as a result-recorder.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
STD_DIR = REPO_ROOT / "docs" / "std"
RESULTS_DIR = REPO_ROOT / "test-results"
RESULTS_FILE = RESULTS_DIR / "latest.json"


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True, **kwargs)


def git_info() -> dict:
    sha = _run(["git", "rev-parse", "HEAD"]).stdout.strip() or None
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]).stdout.strip() or None
    dirty = bool(_run(["git", "status", "--porcelain"]).stdout.strip())
    return {"commit": sha, "branch": branch, "dirty": dirty}


def latest_std_snapshot() -> dict | None:
    if not STD_DIR.exists():
        return None
    candidates = sorted(STD_DIR.glob("std_*.json"))
    if not candidates:
        return None
    latest = candidates[-1]
    with latest.open(encoding="utf-8") as f:
        data = json.load(f)
    data["_std_source_file"] = str(latest.relative_to(REPO_ROOT))
    return data


def run_pytest(pytest_args: list[str]) -> tuple[int, dict]:
    targets = pytest_args or ["tests/"]
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        report_path = Path(tmp.name)
    # NOTE: .venv/bin/python is a symlink that can resolve outside this venv
    # (e.g. to a uv-managed interpreter) on some machines, which silently
    # drops access to packages installed into .venv (pytest-json-report
    # included). python3.12 inside .venv/bin is the venv's own interpreter
    # and always sees its own site-packages.
    cmd = [
        str(REPO_ROOT / ".venv" / "bin" / "python3.12"),
        "-m",
        "pytest",
        "-m",
        "not live",
        f"--json-report-file={report_path}",
        "--json-report",
        *targets,
    ]
    proc = _run(cmd)
    print(proc.stdout)
    if proc.stderr:
        print(proc.stderr, file=sys.stderr)

    if not report_path.exists() or report_path.stat().st_size == 0:
        raise RuntimeError("pytest-json-report produced no output; is the plugin installed?")

    with report_path.open(encoding="utf-8") as f:
        raw = json.load(f)
    report_path.unlink(missing_ok=True)

    summary = raw.get("summary", {})
    pytest_result = {
        "exit_code": proc.returncode,
        "targets": targets,
        "duration_s": raw.get("duration"),
        "summary": {
            "total": summary.get("total", 0),
            "passed": summary.get("passed", 0),
            "failed": summary.get("failed", 0),
            "skipped": summary.get("skipped", 0),
            "error": summary.get("error", 0),
            "xfailed": summary.get("xfailed", 0),
            "xpassed": summary.get("xpassed", 0),
        },
        "failed_tests": [
            t["nodeid"] for t in raw.get("tests", []) if t.get("outcome") == "failed"
        ],
    }
    return proc.returncode, pytest_result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "pytest_args",
        nargs="*",
        help="Extra args/targets forwarded to pytest (default: tests/)",
    )
    parser.add_argument(
        "--commit",
        action="store_true",
        help="git add + commit test-results/latest.json after writing it",
    )
    parser.add_argument(
        "--no-push",
        action="store_true",
        help="With --commit, skip `git push` (commit locally only)",
    )
    args = parser.parse_args()

    exit_code, pytest_result = run_pytest(args.pytest_args)

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git": git_info(),
        "pytest": pytest_result,
        "std": latest_std_snapshot(),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {RESULTS_FILE.relative_to(REPO_ROOT)}")
    print(
        f"pytest: {pytest_result['summary']['passed']} passed, "
        f"{pytest_result['summary']['failed']} failed, "
        f"{pytest_result['summary']['skipped']} skipped "
        f"(exit={exit_code})"
    )

    if args.commit:
        _run(["git", "add", str(RESULTS_FILE.relative_to(REPO_ROOT))])
        msg = (
            f"chore(test-results): update latest pytest results "
            f"({pytest_result['summary']['passed']} passed, "
            f"{pytest_result['summary']['failed']} failed)"
        )
        commit = _run(["git", "commit", "-m", msg])
        print(commit.stdout)
        if commit.returncode != 0:
            print(commit.stderr, file=sys.stderr)
        elif not args.no_push:
            push = _run(["git", "push"])
            print(push.stdout)
            if push.returncode != 0:
                print(push.stderr, file=sys.stderr)

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
