# -*- coding: utf-8 -*-
"""
eval_speaker.py - Đánh giá xác thực người nói (câu 2.2 và 2.3 của B).

Chạy:  python src/eval_speaker.py

Dữ liệu đánh giá (không có file nào được dùng để huấn luyện hay chọn ngưỡng):
  enrolled_dataset_test : người đăng ký, tập test chia từ dataset/ (lệnh ngắn)
  enrolled_testset      : người đăng ký, câu dài trong testset/audio/
  stranger_dataset      : người lạ (la1, la2...) trong dataset/
  stranger_testset      : người lạ trong testset/audio/

Chỉ số: độ chính xác nhận diện người, FAR (người lạ lọt qua), FRR (người đúng bị chặn
hoặc nhận nhầm), EER kèm khoảng tin cậy 95% bằng bootstrap. Hình lưu ở figures/.
"""
from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from speaker import (FIG_DIR, MODEL_PATH, ROOT, UNKNOWN, bootstrap_ci, build_matrix,
                     collect_files, compute_eer, far_frr, far_frr_curve, load_bundle,
                     score_with, trial_scores)


def build_pools(bundle):
    recs = collect_files()
    X, recs = build_matrix(recs, bundle["feature_set"])
    enrolled = set(bundle["speakers"])
    test_rel = set(bundle["split"]["test"])
    keep, pool = [], []
    for i, r in enumerate(recs):
        is_enr = r["speaker"] in enrolled and not r["stranger"]
        if is_enr and r["group"] == "dataset":
            p = "enrolled_dataset_test" if r["rel"] in test_rel else None
        elif is_enr:
            p = "enrolled_testset"
        elif r["group"] == "dataset":
            p = "stranger_dataset"
        else:
            p = "stranger_testset"
        if p:
            keep.append(i)
            pool.append(p)
    recs = [recs[i] for i in keep]
    return X[keep], recs, np.array(pool)


def summarize(m, model, thr, X, recs, pool):
    y = np.array([r["speaker"] for r in recs])
    names, sc = score_with(model, X, m)
    is_gen = np.array([p.startswith("enrolled") for p in pool])
    out = {"method": m, "threshold": float(thr), "n_genuine": int(is_gen.sum()),
           "n_stranger": int((~is_gen).sum())}
    if is_gen.sum():
        out["closed_set_accuracy"] = float(np.mean(names[is_gen] == y[is_gen]))
    gen = trial_scores(y[is_gen], names[is_gen], sc[is_gen])
    imp = sc[~is_gen]
    if len(gen):
        out["FRR"] = float(np.mean(gen < thr))
    if len(imp):
        out["FAR"] = float(np.mean(imp >= thr))
    if len(gen) and len(imp):
        eer, thr_eer = compute_eer(gen, imp)
        lo, hi = bootstrap_ci(gen, imp, lambda g, i: compute_eer(g, i)[0])
        out.update({"EER": eer, "EER_threshold": thr_eer, "EER_ci95": [lo, hi]})
    by_pool = {}
    for p in sorted(set(map(str, pool))):
        mask = pool == p
        if p.startswith("enrolled"):
            g = trial_scores(y[mask], names[mask], sc[mask])
            by_pool[p] = {"n": int(mask.sum()), "FRR": float(np.mean(g < thr)),
                          "accuracy": float(np.mean(names[mask] == y[mask]))}
        else:
            by_pool[p] = {"n": int(mask.sum()), "FAR": float(np.mean(sc[mask] >= thr))}
    out["by_pool"] = by_pool
    out["FAR_per_stranger"] = {str(s): float(np.mean(sc[(y == s) & ~is_gen] >= thr))
                               for s in sorted(set(y[~is_gen]))}
    out["FRR_per_enrolled"] = {
        str(s): float(np.mean(trial_scores(y[is_gen & (y == s)], names[is_gen & (y == s)],
                                      sc[is_gen & (y == s)]) < thr))
        for s in sorted(set(y[is_gen]))}
    return out, (names, sc, y, is_gen, gen, imp)


def plot_det(results, path):
    fig, ax = plt.subplots(figsize=(5, 5))
    for m, (gen, imp) in results.items():
        _, far, frr = far_frr_curve(gen, imp)
        ax.plot(far * 100, frr * 100, label=m)
    ax.plot([0, 100], [0, 100], "k--", lw=0.8, label="FAR = FRR")
    ax.set_xlabel("FAR - người lạ lọt qua (%)")
    ax.set_ylabel("FRR - người đúng bị chặn (%)")
    ax.set_title("Đường FAR/FRR theo ngưỡng")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_scores(gen, imp, thr, method, path):
    fig, ax = plt.subplots(figsize=(6, 3.6))
    g = gen[np.isfinite(gen)]
    bins = np.linspace(min(g.min() if len(g) else 0, imp.min()), max(g.max() if len(g) else 1, imp.max()), 30)
    ax.hist(g, bins=bins, alpha=0.6, label="người đã đăng ký")
    ax.hist(imp, bins=bins, alpha=0.6, label="người lạ")
    ax.axvline(thr, color="k", ls="--", label=f"ngưỡng = {thr:.2f}")
    ax.set_xlabel(f"điểm người nói ({method})")
    ax.set_ylabel("số câu")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(names, sc, y, is_gen, thr, speakers, path):
    cols = speakers + ["từ chối"]
    rows = speakers + ["người lạ"]
    M = np.zeros((len(rows), len(cols)), dtype=int)
    for n, s, t, g in zip(names, sc, y, is_gen):
        pred = cols[-1] if (n == UNKNOWN or s < thr) else str(n)
        r = rows.index(t) if g else len(rows) - 1
        M[r, cols.index(pred) if pred in cols else len(cols) - 1] += 1
    fig, ax = plt.subplots(figsize=(1.2 * len(cols) + 2, 1.0 * len(rows) + 1.5))
    ax.imshow(M, cmap="Blues")
    ax.set_xticks(range(len(cols)), cols)
    ax.set_yticks(range(len(rows)), rows)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, M[i, j], ha="center", va="center",
                    color="white" if M[i, j] > M.max() / 2 else "black")
    ax.set_xlabel("hệ thống quyết định")
    ax.set_ylabel("người nói thật")
    ax.set_title("Ma trận nhầm lẫn xác thực")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default=None, help="phương pháp để vẽ hình (mặc định: đã chọn khi train)")
    args = ap.parse_args()

    bundle = load_bundle()
    X, recs, pool = build_pools(bundle)
    print("Số câu theo nhóm:", {str(p): int((pool == p).sum()) for p in sorted(set(pool))})
    if not any(p.startswith("stranger") for p in pool):
        print("  ! Chưa có dữ liệu người lạ (tên la1, la2, ...). FAR và EER sẽ không tính được.")

    report, curves, extra = {}, {}, {}
    for m, model in bundle["models"].items():
        res, ex = summarize(m, model, bundle["thresholds"][m], X, recs, pool)
        report[m] = res
        extra[m] = ex
        if len(ex[4]) and len(ex[5]):
            curves[m] = (ex[4], ex[5])

    print(f"\n{'phương pháp':12s} {'acc nhận diện':>14s} {'FRR':>7s} {'FAR':>7s} {'EER [CI95]':>24s}")
    for m, r in report.items():
        eer = (f"{r['EER'] * 100:5.1f}% [{r['EER_ci95'][0] * 100:4.1f}-{r['EER_ci95'][1] * 100:4.1f}]"
               if "EER" in r else "-")
        print(f"{m:12s} {r.get('closed_set_accuracy', float('nan')) * 100:13.1f}% "
              f"{r.get('FRR', float('nan')) * 100:6.1f}% {r.get('FAR', float('nan')) * 100:6.1f}% {eer:>24s}")
    chosen = args.method or bundle["method"]
    r = report[chosen]
    print(f"\nChi tiết phương pháp '{chosen}' (ngưỡng {r['threshold']:.3f}):")
    for p, v in r["by_pool"].items():
        print("  ", p, v)
    print("   FAR theo từng người lạ:", r["FAR_per_stranger"])
    print("   FRR theo từng người đăng ký:", r["FRR_per_enrolled"])

    FIG_DIR.mkdir(exist_ok=True)
    if curves:
        plot_det(curves, FIG_DIR / "fig8_speaker_far_frr.png")
        names, sc, y, is_gen, gen, imp = extra[chosen]
        plot_scores(gen, imp, r["threshold"], chosen, FIG_DIR / "fig9_speaker_scores.png")
    names, sc, y, is_gen, gen, imp = extra[chosen]
    plot_confusion(names, sc, y, is_gen, r["threshold"], bundle["speakers"],
                   FIG_DIR / "fig10_speaker_confusion.png")

    out = MODEL_PATH.parent / "speaker_eval_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nĐã lưu {out.relative_to(ROOT)} và hình fig8, fig9, fig10 trong figures/")
    if sum(p.startswith("stranger") for p in pool) < 60:
        print("Lưu ý: số câu người lạ ít (< 60) nên FAR/EER dao động mạnh; hãy ghi rõ khoảng tin cậy trong báo cáo.")


if __name__ == "__main__":
    main()