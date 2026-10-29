#!/usr/bin/env python3
"""Chấm lại các transcript ASR mà không chạy mô hình.

Ví dụ:
    python src/rescore_asr.py models/asr_eval/results_phowhisper-base.csv
    python src/rescore_asr.py D:/tmp/asr_results --glob 'results_*.csv'

Việc tính điểm gọi lại ``asr.to_command`` và ``eval_asr.summarize`` để kết
quả khớp với pipeline đánh giá chính.
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
import asr  # noqa: E402
import eval_asr  # noqa: E402


def use_commands_ref(git_ref: str | None) -> None:
    """Nạp commands.py hiện tại hoặc một blob từ Git mà không sửa worktree."""
    if not git_ref:
        return
    source = subprocess.check_output(
        ["git", "show", f"{git_ref}:src/commands.py"], cwd=ROOT, text=True, encoding="utf-8"
    )
    module = types.ModuleType(f"commands_{git_ref.replace('/', '_')}")
    module.__file__ = str(ROOT / "src" / "commands.py")
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    asr.commands = module


def rescore_rows(rows: list[dict]) -> tuple[list[dict], dict]:
    rescored = []
    for row in rows:
        r = asr.to_command(row.get("hyp_text", ""))
        out = dict(row)
        out.update(
            pred_device=r["device"], pred_action=r["action"], pred_value=r["value"],
            reason=r["reason"], score=r["score"],
        )
        out["ok_device"] = r["device"] == (row.get("device") or None)
        out["ok_action"] = r["action"] == (row.get("action") or None)
        ref_value = row.get("value", "")
        ref_value = int(float(ref_value)) if str(ref_value).strip() else None
        out["ok_value"] = (r["value"] == ref_value) if ref_value is not None else None
        if row.get("device"):
            out["ok_exact"] = out["ok_device"] and out["ok_action"] and r["value"] == ref_value and r["reason"] is None
        else:
            out["ok_exact"] = r["device"] is None
        rescored.append(out)
    return rescored, eval_asr.summarize(rescored, [])


def _pct(value) -> str:
    return "–" if value is None else f"{100 * value:.1f}%"


def print_report(path: Path, rows: list[dict], report: dict) -> None:
    print(f"\n=== {path} ===")
    print("n={n} | thiết bị={acc_device} | hành động={acc_action} | giá trị={acc_value} | "
          "cả lệnh={acc_exact} | thực thi sai={wrong_exec} | từ chối={rejected_cmd}".format(
              n=report["n"], acc_device=_pct(report["acc_device"]),
              acc_action=_pct(report["acc_action"]), acc_value=_pct(report["acc_value"]),
              acc_exact=_pct(report["acc_exact"]), wrong_exec=_pct(report["wrong_exec"]),
              rejected_cmd=_pct(report["rejected_cmd"])))
    print("speaker | n | exact | wrong_exec | rejected")
    for speaker in sorted({r.get("speaker", "?") for r in rows}):
        sub = [r for r in rows if r.get("speaker", "?") == speaker]
        exact = sum(bool(r["ok_exact"]) for r in sub) / len(sub)
        wrong = sum(r["reason"] is None and not r["ok_exact"] for r in sub) / len(sub)
        commands = [r for r in sub if r.get("device")]
        rejected = (sum(r["reason"] is not None for r in commands) / len(commands)) if commands else None
        print(f"{speaker} | {len(sub)} | {_pct(exact)} | {_pct(wrong)} | {_pct(rejected)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", type=Path, help="CSV kết quả hoặc thư mục chứa CSV")
    ap.add_argument("--glob", default="results_*.csv", help="glob khi path là thư mục")
    ap.add_argument("--git-ref", help="dùng commands.py từ Git ref, ví dụ develop; không sửa worktree")
    args = ap.parse_args()
    use_commands_ref(args.git_ref)
    paths = sorted(args.path.glob(args.glob)) if args.path.is_dir() else [args.path]
    if not paths:
        raise SystemExit("Không tìm thấy CSV")
    for path in paths:
        with path.open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        rescored, report = rescore_rows(rows)
        print_report(path, rescored, report)


if __name__ == "__main__":
    main()
