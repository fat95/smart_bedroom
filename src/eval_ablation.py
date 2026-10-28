# -*- coding: utf-8 -*-
"""
eval_ablation.py - Ablation cho câu 1.4: hệ thống CÓ và KHÔNG có xác thực người nói.

Chạy:  python src/eval_ablation.py [--asr-results models/asr_results.csv]

Hai cấu hình:
  A   : chỉ nhận dạng lệnh (PhoWhisper). Lệnh nhận đúng là được thực thi, bất kể ai nói.
  A+B : nhận dạng lệnh + xác thực người nói. Chỉ thực thi khi lệnh đúng VÀ là người đăng ký.

Cần biết lệnh của mỗi file có được nhận dạng đúng không. A xuất file CSV có 2 cột:
    file,command_ok          (file = tên file .wav, command_ok = 1 nếu nhận đúng, 0 nếu sai)
Nếu chưa có file đó, script giả định ASR đúng 100% (chỉ đo riêng tác dụng của xác thực);
khi A có kết quả thật thì chạy lại để số liệu phản ánh cả lỗi nhận dạng.

Hai chỉ số chính:
  - Người lạ được thực thi (càng thấp càng tốt)   = rủi ro bảo mật
  - Người đúng được thực thi (càng cao càng tốt) = độ dùng được
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from eval_speaker import build_pools
from speaker import FIG_DIR, MODEL_PATH, ROOT, UNKNOWN, load_bundle, score_with


def load_asr_results(path: Path):
    if not path.exists():
        return None
    out = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            out[Path(row["file"]).name] = str(row["command_ok"]).strip().lower() in ("1", "true", "yes")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asr-results", default=str(MODEL_PATH.parent / "asr_results.csv"))
    args = ap.parse_args()

    bundle = load_bundle()
    X, recs, pool = build_pools(bundle)
    is_enr = np.array([p.startswith("enrolled") for p in pool])
    if is_enr.all() or not is_enr.any():
        print("  ! Cần có cả người đăng ký lẫn người lạ trong dữ liệu để làm ablation.")

    asr = load_asr_results(Path(args.asr_results))
    if asr is None:
        print("  ! Chưa có file kết quả ASR của A -> giả định nhận dạng lệnh đúng 100%.")
        cmd_ok = np.ones(len(recs), dtype=bool)
    else:
        present = np.array([Path(r["path"]).name in asr for r in recs])
        if not present.any():
            raise SystemExit("Không file nào trong kết quả ASR khớp với dữ liệu. Kiểm tra cột 'file'.")
        if not present.all():
            print(f"  (chỉ dùng {int(present.sum())}/{len(recs)} file có trong kết quả ASR của A)")
        X, pool = X[present], pool[present]
        recs = [r for r, ok in zip(recs, present) if ok]
        is_enr = is_enr[present]
        cmd_ok = np.array([asr[Path(r["path"]).name] for r in recs])

    report = {"asr_assumed_perfect": asr is None, "n_enrolled": int(is_enr.sum()),
              "n_stranger": int((~is_enr).sum()), "configs": {}}
    rows = []
    for m, model in bundle["models"].items():
        names, sc = score_with(model, X, m)
        verified = (names != UNKNOWN) & (sc >= bundle["thresholds"][m])
        for cfg, accept in (("A (chỉ ASR)", cmd_ok), (f"A+B ({m})", cmd_ok & verified)):
            res = {"enrolled_executed": float(np.mean(accept[is_enr])) if is_enr.any() else None,
                   "stranger_executed": float(np.mean(accept[~is_enr])) if (~is_enr).any() else None}
            if cfg.startswith("A+B"):
                res["enrolled_blocked_by_speaker"] = (
                    float(np.mean((cmd_ok & ~verified)[is_enr])) if is_enr.any() else None)
            report["configs"][cfg] = res
            rows.append((cfg, res))

    print(f"\n{'cấu hình':22s} {'người đúng được chạy':>22s} {'người lạ được chạy':>20s}")
    for cfg, r in rows:
        f = lambda v: "-" if v is None else f"{v * 100:.1f}%"
        print(f"{cfg:22s} {f(r['enrolled_executed']):>22s} {f(r['stranger_executed']):>20s}")

    FIG_DIR.mkdir(exist_ok=True)
    labels = [c for c, _ in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(1.6 * len(rows) + 2, 4))
    ax.bar(x - 0.2, [(r["enrolled_executed"] or 0) * 100 for _, r in rows], 0.4, label="người đăng ký được thực thi")
    ax.bar(x + 0.2, [(r["stranger_executed"] or 0) * 100 for _, r in rows], 0.4, label="người lạ được thực thi")
    ax.set_xticks(x, labels, rotation=15)
    ax.set_ylabel("%")
    ax.set_title("Ablation: có và không có xác thực người nói")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig11_ablation.png", dpi=150)
    plt.close(fig)

    out = MODEL_PATH.parent / "ablation_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nĐã lưu {out.relative_to(ROOT)} và figures/fig11_ablation.png")


if __name__ == "__main__":
    main()