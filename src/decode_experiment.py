#!/usr/bin/env python3
"""Chạy các cấu hình giải mã PhoWhisper vào một thư mục kết quả ngoài dự án.

Ví dụ:
  python src/decode_experiment.py --model phowhisper-small --out D:/tmp/decode_eval
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
import asr  # noqa: E402
import eval_asr as E  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="phowhisper-small")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip-simple", action="store_true")
    ap.add_argument("--skip-testset", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    configs = [("greedy", "greedy", False), ("nbest", "nbest", False),
               ("prompt", "greedy", True), ("nbest_prompt", "nbest", True),
               ("majority", "majority", False)]
    model = asr.get_asr(args.model)
    datasets = [] if args.skip_testset else [(False, E.load_testset(ROOT / "testset/sentences.csv", ROOT / "testset/audio"))]
    if not args.skip_simple:
        datasets.append((True, E.load_simple(ROOT / "features/manifest.csv", "test")))
    for name, decode, prompt in configs:
        for simple, items in datasets:
            if args.limit:
                items = items[:args.limit]
            out = args.out / (name + ("_simple" if simple else "_testset"))
            print(f"\n=== {name} {'simple' if simple else 'testset'} ({len(items)} mẫu) ===", flush=True)
            transcribe = lambda wav, m=model, d=decode, p=prompt: m.transcribe(wav, decode=d, prompt=p)
            rows, rep = E.evaluate_items(items, transcribe)
            rep["model"] = args.model; rep["decode"] = decode; rep["prompt"] = prompt
            E.write_outputs(args.model, rows, rep, out, simple)
            baseline = E.eval_baseline(items, ROOT / "models/command_model.joblib") if simple else None
            E.write_summary({args.model: rep}, out, simple, baseline)


if __name__ == "__main__":
    main()
