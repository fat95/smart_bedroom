# -*- coding: utf-8 -*-
"""
eval_ablation.py - Ablation cho câu 1.4: hệ thống CÓ và KHÔNG có xác thực người nói.

Chạy (gõ đúng như dưới đây, KHÔNG gõ dấu ngoặc vuông):
    python src/eval_ablation.py
    python src/eval_ablation.py --asr-model phowhisper-small
    python src/eval_ablation.py --asr-results models/asr_eval/results_phowhisper-small.csv models/asr_eval_simple/results_phowhisper-small.csv
Các tham số --asr-model và --asr-results đều tùy chọn.

Ba cấu hình:
  A   : chỉ nhận dạng lệnh (PhoWhisper). Lệnh nhận đúng là được thực thi, bất kể ai nói.
  B   : chỉ xác thực người nói (không quan tâm lệnh đúng hay sai).
  A+B : nhận dạng lệnh + xác thực người nói. Chỉ thực thi khi lệnh đúng VÀ là người đăng ký.
Cấu hình A không phụ thuộc phương pháp xác thực nên chỉ được tính và vẽ một lần;
B và A+B được tính riêng cho từng phương pháp (proba, margin, ...).

Cần biết lệnh của mỗi file có được nhận dạng đúng không. Script đọc thẳng results_<mô hình>.csv do
eval_asr.py ghi (cột `path` và `ok_exact`), mặc định models/asr_eval*/results_<--asr-model>.csv
(nên chạy eval_asr.py với --out khác nhau cho testset và --simple để hai lần chạy không ghi đè nhau).
Khớp file theo 2 thành phần cuối của đường dẫn (thư mục/tên file) nên không đụng nhau giữa các lệnh.
Nếu chưa có file nào, script giả định ASR đúng 100% (chỉ đo riêng tác dụng của xác thực).

Hai chỉ số chính:
  - Người lạ được thực thi (càng thấp càng tốt)   = rủi ro bảo mật
  - Người đúng được thực thi (càng cao càng tốt) = độ dùng được
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import textwrap
from pathlib import Path, PurePosixPath

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from eval_speaker import build_pools
from speaker import FIG_DIR, MODEL_PATH, ROOT, UNKNOWN, load_bundle, score_with


def key_of(path) -> str:
    """.../bat_tivi/vinh_001.wav -> 'bat_tivi/vinh_001.wav' (2 thành phần cuối, chịu được đường dẫn Windows)."""
    return "/".join(PurePosixPath(str(path).replace("\\", "/")).parts[-2:])


def load_asr_results(paths):
    """Đọc một hay nhiều results_<mô hình>.csv của eval_asr.py (cột path, ok_exact). Không có file nào -> None."""
    out = {}
    for p in paths:
        if not Path(p).exists():
            continue
        with open(p, encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                ok = row.get("ok_exact", row.get("command_ok", ""))
                out[key_of(row.get("path", row.get("file", "")))] = str(ok).strip().lower() in ("1", "true", "yes")
    return out or None


def plot_ablation(rows, path):
    """Biểu đồ cột: người đăng ký và người lạ được thực thi theo từng cấu hình."""
    labels = [textwrap.fill(c, 16) for c, _ in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(1.5 * len(rows) + 2, 4.6))
    b1 = ax.bar(x - 0.2, [(r["enrolled_executed"] or 0) * 100 for _, r in rows], 0.4,
                label="người đăng ký được thực thi")
    b2 = ax.bar(x + 0.2, [(r["stranger_executed"] or 0) * 100 for _, r in rows], 0.4,
                label="người lạ được thực thi")
    ax.bar_label(b1, fmt="%.1f", fontsize=8, padding=2)
    ax.bar_label(b2, fmt="%.1f", fontsize=8, padding=2)
    ax.set_xticks(x, labels, fontsize=9)
    ax.set_ylabel("%")
    ax.set_ylim(0, 125)
    ax.set_yticks(range(0, 101, 20))
    ax.set_title("Ablation: có và không có xác thực người nói")
    ax.legend(loc="upper center", ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asr-model", default="phowhisper-small")
    ap.add_argument("--asr-results", nargs="*", default=None,
                    help="một hay nhiều results_*.csv của eval_asr.py (mặc định: models/asr_eval*/results_<model>.csv)")
    args = ap.parse_args()
    paths = args.asr_results or sorted(glob.glob(str(MODEL_PATH.parent / "asr_eval*" / f"results_{args.asr_model}.csv")))

    bundle = load_bundle()
    X, recs, pool = build_pools(bundle)
    is_enr = np.array([p.startswith("enrolled") for p in pool])
    if is_enr.all() or not is_enr.any():
        print("  ! Cần có cả người đăng ký lẫn người lạ trong dữ liệu để làm ablation.")

    asr = load_asr_results(paths)
    print("  đọc kết quả ASR từ:", ", ".join(Path(p).name for p in paths) if asr else "(không có)")
    if asr is None:
        print("  ! Chưa có file kết quả ASR của A -> giả định nhận dạng lệnh đúng 100%.")
        cmd_ok = np.ones(len(recs), dtype=bool)
    else:
        present = np.array([key_of(r["path"]) in asr for r in recs])
        if not present.any():
            raise SystemExit("Không file nào trong kết quả ASR khớp với dữ liệu. Kiểm tra cột 'path' của results_*.csv.")
        if not present.all():
            print(f"  (chỉ dùng {int(present.sum())}/{len(recs)} file có trong kết quả ASR của A)")
        X, pool = X[present], pool[present]
        recs = [r for r, ok in zip(recs, present) if ok]
        is_enr = is_enr[present]
        cmd_ok = np.array([asr[key_of(r["path"])] for r in recs])

    report = {"asr_assumed_perfect": asr is None, "n_enrolled": int(is_enr.sum()),
              "n_stranger": int((~is_enr).sum()), "configs": {}}
    rows = []

    def rates(accept):
        return {"enrolled_executed": float(np.mean(accept[is_enr])) if is_enr.any() else None,
                "stranger_executed": float(np.mean(accept[~is_enr])) if (~is_enr).any() else None}

    # A (chỉ ASR) không phụ thuộc phương pháp xác thực nên chỉ tính một lần.
    res = rates(cmd_ok)
    report["configs"]["A (chỉ ASR)"] = res
    rows.append(("A (chỉ ASR)", res))

    for m, model in bundle["models"].items():
        names, sc = score_with(model, X, m)
        verified = (names != UNKNOWN) & (sc >= bundle["thresholds"][m])

        cfg = f"B ({m}, chỉ xác thực)"
        res = rates(verified)
        report["configs"][cfg] = res
        rows.append((cfg, res))

        cfg = f"A+B ({m})"
        res = rates(cmd_ok & verified)
        res["enrolled_blocked_by_speaker"] = (
            float(np.mean((cmd_ok & ~verified)[is_enr])) if is_enr.any() else None)
        report["configs"][cfg] = res
        rows.append((cfg, res))

    print(f"\nSố câu: người đăng ký = {report['n_enrolled']}, người lạ = {report['n_stranger']}")
    print(f"\n{'cấu hình':26s} {'người đúng được chạy':>22s} {'người lạ được chạy':>20s}")
    for cfg, r in rows:
        f = lambda v: "-" if v is None else f"{v * 100:.1f}%"
        print(f"{cfg:26s} {f(r['enrolled_executed']):>22s} {f(r['stranger_executed']):>20s}")

    FIG_DIR.mkdir(exist_ok=True)
    plot_ablation(rows, FIG_DIR / "fig11_ablation.png")

    out = MODEL_PATH.parent / "ablation_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nĐã lưu {out.relative_to(ROOT)} và figures/fig11_ablation.png")


if __name__ == "__main__":
    main()