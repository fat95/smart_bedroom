# -*- coding: utf-8 -*-
"""
train_speaker.py - Huấn luyện SVM xác thực người nói và chọn ngưỡng từ chối người lạ.

Chạy:  python src/train_speaker.py
       python src/train_speaker.py --feature-set base            # so sánh đặc trưng (câu 1.3)
       python src/train_speaker.py --unknown-dir external_voices  # thêm lớp "unknown" từ giọng ngoài

Quy tắc chống rò rỉ dữ liệu:
  * Chỉ đọc dataset/ của người ĐÃ ĐĂNG KÝ. Người lạ (la1, la2, ...) không được đọc ở đây.
  * Ngưỡng chọn bằng leave-one-speaker-out: lần lượt coi 1 người đăng ký là "người lạ".
  * Người lạ thật chỉ xuất hiện ở eval_speaker.py và eval_ablation.py.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np

from speaker import (CACHE_PATH, MODEL_PATH, ROOT, UNKNOWN, FEATURE_VERSION,
                     build_matrix, build_model, collect_files, compute_eer,
                     make_rec, make_svm, score_with, split_indices, trial_scores)

C_GRID = (0.5, 1, 5, 20, 100)
GAMMA_GRID = ("scale", 0.003, 0.01, 0.03)


def load_unknown(dirpath: str, feature_set: str, use_cache: bool, limit: int = 400, seed: int = 0):
    files = sorted(Path(dirpath).rglob("*.wav"))
    rng = np.random.default_rng(seed)
    if len(files) > limit:
        files = [files[i] for i in rng.choice(len(files), limit, replace=False)]
    recs = [make_rec(f, "unknown", speaker=UNKNOWN) for f in files]
    X, kept = build_matrix(recs, feature_set, use_cache)
    print(f"  lớp unknown: {len(kept)} file từ {dirpath}")
    return X


def loso_scores(method, C, gamma, X, y, split, X_unk=None):
    """Điểm 'đúng người' (gen) và điểm người lạ (imp), gộp qua các fold leave-one-speaker-out."""
    gens, imps = [], []
    spks = sorted(set(y.tolist()))
    for s in spks:
        tr = [i for i in split["train"] if y[i] != s]
        va = [i for i in split["val"] if y[i] != s]
        imp_idx = np.where(y == s)[0]
        Xt, yt = X[tr], y[tr]
        if method == "unknown":
            Xt = np.vstack([Xt, X_unk])
            yt = np.concatenate([yt, np.array([UNKNOWN] * len(X_unk))])
        model = build_model(method, C, gamma).fit(Xt, yt)
        names_v, sc_v = score_with(model, X[va], method)
        gens.append(trial_scores(y[va], names_v, sc_v))
        _, sc_i = score_with(model, X[imp_idx], method)
        imps.append(sc_i)
    return np.concatenate(gens), np.concatenate(imps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-set", choices=["base", "plus"], default="plus")
    ap.add_argument("--unknown-dir", default=None, help="thư mục WAV giọng ngoài cho lớp unknown")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    use_cache = not args.no_cache

    # 1) Chỉ lấy dữ liệu đăng ký (người lạ bị loại ngay từ đầu)
    all_recs = collect_files()
    strangers = {}
    for r in all_recs:
        if r["stranger"]:
            strangers[r["speaker"]] = strangers.get(r["speaker"], 0) + 1
    if strangers:
        print("Người lạ (KHÔNG dùng để huấn luyện/chọn ngưỡng): "
              + ", ".join(f"{k}={v} file" for k, v in sorted(strangers.items())))
    recs = [r for r in all_recs if r["group"] == "dataset" and not r["stranger"]]
    X, recs = build_matrix(recs, args.feature_set, use_cache)
    if len(recs) == 0:
        raise SystemExit("Không có dữ liệu trong dataset/. Hãy thu âm trước (trang /record).")
    y = np.array([r["speaker"] for r in recs])
    spks = sorted(set(y.tolist()))
    print(f"Người đăng ký ({len(spks)}): " + ", ".join(f"{s}={int((y == s).sum())}" for s in spks))
    if len(spks) < 2:
        raise SystemExit("Cần ít nhất 2 người đăng ký.")
    for s in spks:
        if (y == s).sum() < 20:
            print(f"  ! {s} chỉ có {(y == s).sum()} mẫu, nên thu thêm (khuyến nghị >= 30)")

    split = split_indices(recs, seed=args.seed)
    tr, va, te = split["train"], split["val"], split["test"]
    print(f"Chia dữ liệu: train={len(tr)}, val={len(va)}, test={len(te)}")

    X_unk = load_unknown(args.unknown_dir, args.feature_set, use_cache) if args.unknown_dir else None
    methods = ["proba", "margin"] + (["unknown"] if X_unk is not None and len(X_unk) else [])

    # 2) Chọn siêu tham số trên tập val
    grid, best = [], None
    for C in C_GRID:
        for g in GAMMA_GRID:
            m = make_svm(C, g, probability=False).fit(X[tr], y[tr])
            acc = float(m.score(X[va], y[va])) if len(va) else float(m.score(X[tr], y[tr]))
            grid.append({"C": C, "gamma": g, "val_acc": acc})
            if best is None or acc > best[0] + 1e-9:
                best = (acc, C, g)
    _, C, gamma = best
    print(f"Siêu tham số tốt nhất: C={C}, gamma={gamma}, val_acc={best[0]:.3f}")

    # 3) Chọn ngưỡng cho từng phương pháp
    if len(spks) < 3:
        print("  ! Chỉ có 2 người đăng ký nên không làm được leave-one-speaker-out; "
              "ngưỡng lấy theo phân vị 5% của điểm đúng người trên tập val (chỉ tham khảo).")
    else:
        print(f"Leave-one-speaker-out với {len(spks)} người (mỗi fold chỉ còn {len(spks) - 1} lớp).")
        if len(spks) == 3:
            print("  ! Mỗi fold chỉ có 2 lớp nên điểm người lạ giả lập kém tin cậy hơn; "
                  "có thêm người đăng ký sẽ cho ngưỡng đáng tin hơn.")

    thresholds, loso_eer, models = {}, {}, {}
    for m in methods:
        if len(spks) >= 3:
            _, imp = loso_scores(m, C, gamma, X, y, split, X_unk)
            # Điểm "đúng người" lấy từ mô hình đủ số lớp (giống mô hình cuối), điểm người lạ giả lập
            # lấy từ các fold LOSO. Như vậy ngưỡng không bị lệch vì fold LOSO có ít lớp hơn.
            Xt, yt = X[tr], y[tr]
            if m == "unknown":
                Xt = np.vstack([Xt, X_unk])
                yt = np.concatenate([yt, np.array([UNKNOWN] * len(X_unk))])
            full = build_model(m, C, gamma).fit(Xt, yt)
            names_v, sc_v = score_with(full, X[va], m)
            gen = trial_scores(y[va], names_v, sc_v)
            eer, thr = compute_eer(gen, imp)
            loso_eer[m] = eer
            thresholds[m] = thr
            print(f"  [{m:7s}] EER chọn ngưỡng={eer * 100:5.1f}%  ngưỡng={thr:.3f}  "
                  f"(đúng người={len(gen)}, người lạ giả lập={len(imp)})")
        else:
            model = build_model(m, C, gamma).fit(X[tr], y[tr])
            names, sc = score_with(model, X[va], m)
            g = trial_scores(y[va], names, sc)
            g = g[np.isfinite(g)]
            thresholds[m] = float(np.percentile(g, 5)) if len(g) else 0.5
            loso_eer[m] = None

    # 4) Huấn luyện mô hình cuối trên train + val
    fit_idx = tr + va
    for m in methods:
        Xf, yf = X[fit_idx], y[fit_idx]
        if m == "unknown":
            Xf = np.vstack([Xf, X_unk])
            yf = np.concatenate([yf, np.array([UNKNOWN] * len(X_unk))])
        models[m] = build_model(m, C, gamma).fit(Xf, yf)
        if len(te):
            names, _ = score_with(models[m], X[te], m)
            acc = float(np.mean(names == y[te]))
            print(f"  [{m:7s}] độ chính xác nhận diện người (tập test đăng ký) = {acc * 100:.1f}%")

    valid = [m for m in methods if loso_eer.get(m) is not None]
    # Chỉ có 2 người đăng ký: không đo được EER, xác suất SVM 2 lớp luôn >= 0,5 nên ưu tiên điểm biên.
    chosen = min(valid, key=lambda k: loso_eer[k]) if valid else "margin"
    print(f"Phương pháp được chọn: {chosen}")

    rel = lambda idxs: [recs[i]["rel"] for i in idxs]
    bundle = {
        "feature_set": args.feature_set, "feature_version": FEATURE_VERSION,
        "speakers": spks, "models": models, "method": chosen,
        "thresholds": thresholds, "params": {"C": C, "gamma": gamma},
        "split": {"train": rel(tr), "val": rel(va), "test": rel(te)},
        "seed": args.seed, "has_unknown_class": X_unk is not None,
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH, compress=3)

    report = {
        "feature_set": args.feature_set, "speakers": spks,
        "samples_per_speaker": {s: int((y == s).sum()) for s in spks},
        "split_sizes": {"train": len(tr), "val": len(va), "test": len(te)},
        "grid": grid, "best": {"C": C, "gamma": gamma, "val_acc": best[0]},
        "loso_eer": loso_eer, "thresholds": thresholds, "chosen_method": chosen,
        "seconds": round(time.time() - t0, 1),
    }
    (MODEL_PATH.parent / "speaker_train_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Đã lưu {MODEL_PATH.relative_to(ROOT)} và models/speaker_train_report.json")


if __name__ == "__main__":
    main()