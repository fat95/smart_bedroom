# -*- coding: utf-8 -*-
"""
train_speaker.py - Huấn luyện SVM xác thực người nói và chọn ngưỡng từ chối người lạ.

Chạy:  python src/train_speaker.py
       python src/train_speaker.py --feature-set base            # so sánh đặc trưng (câu 1.3)
       python src/train_speaker.py --unknown-dir external_voices  # thêm lớp "unknown" từ giọng ngoài

Quy tắc chống rò rỉ dữ liệu:
  * Chỉ đọc dataset/ của người ĐÃ ĐĂNG KÝ. Người lạ (la1, la2, ...) không được đọc ở đây.
  * Ngưỡng chọn bằng leave-one-speaker-out (LOSO): lần lượt coi 1 người đăng ký là "người lạ".
    Trong mỗi fold, người bị bỏ KHÔNG xuất hiện ở bất kỳ dạng nào trong dữ liệu huấn luyện
    (kể cả dưới nhãn unknown), nếu không điểm người lạ sẽ đẹp một cách giả tạo (EER = 0%).
  * Chỉ phương pháp "unknown" mới có lớp unknown, và lớp đó chỉ chứa giọng ngoài (--unknown-dir).
  * Người lạ thật chỉ xuất hiện ở eval_speaker.py và eval_ablation.py.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import GroupKFold

from speaker import (MODEL_PATH, ROOT, UNKNOWN, FEATURE_VERSION,
                     build_matrix, build_model, collect_files, compute_eer,
                     make_rec, make_svm, score_with, split_indices, trial_scores)

C_GRID = (0.5, 1, 5, 20, 100)
GAMMA_GRID = ("scale", 0.003, 0.01, 0.03)
# Khi EER và AUC hòa nhau thì ưu tiên theo thứ tự này (ghi rõ trong báo cáo).
METHOD_PRIOR = {"margin": 0, "unknown": 1, "proba": 2}


def load_unknown(dirpath: str, feature_set: str, use_cache: bool, limit: int = 400, seed: int = 0):
    files = sorted(Path(dirpath).rglob("*.wav"))
    rng = np.random.default_rng(seed)
    if len(files) > limit:
        files = [files[i] for i in rng.choice(len(files), limit, replace=False)]
    recs = [make_rec(f, "unknown", speaker=UNKNOWN) for f in files]
    X, kept = build_matrix(recs, feature_set, use_cache)
    print(f"  lớp unknown: {len(kept)} file từ {dirpath}")
    return X


def auc_score(gen, imp) -> float:
    """P(điểm đúng người > điểm người lạ). 1.0 = tách hoàn toàn, 0.5 = ngẫu nhiên."""
    if len(gen) == 0 or len(imp) == 0:
        return float("nan")
    g = np.asarray(gen, dtype=float)[:, None]
    i = np.asarray(imp, dtype=float)[None, :]
    return float(np.mean(g > i) + 0.5 * np.mean(g == i))


def loso_scores(method, C, gamma, X, y, split, X_unk=None):
    """Điểm 'đúng người' (gen) và điểm người lạ (imp), gộp qua các fold leave-one-speaker-out.

    Mỗi fold: người s bị bỏ hoàn toàn khỏi huấn luyện. Điểm người lạ = điểm của MỌI mẫu của s
    (an toàn vì mô hình chưa từng thấy s). Điểm đúng người = mẫu val của những người còn lại.
    """
    gens, imps = [], []
    for s in sorted(set(y.tolist())):
        tr = [i for i in split["train"] if y[i] != s]
        va = [i for i in split["val"] if y[i] != s]
        imp_idx = np.where(y == s)[0]

        Xt, yt = X[tr], y[tr]
        # Chỉ phương pháp "unknown" có lớp unknown, và lớp đó chỉ gồm giọng ngoài.
        if method == "unknown" and X_unk is not None and len(X_unk) > 0:
            Xt = np.vstack([Xt, X_unk])
            yt = np.concatenate([yt, np.array([UNKNOWN] * len(X_unk))])

        model = build_model(method, C, gamma).fit(Xt, yt)

        names_v, sc_v = score_with(model, X[va], method)
        gens.append(trial_scores(y[va], names_v, sc_v))
        _, sc_i = score_with(model, X[imp_idx], method)
        imps.append(sc_i)
    return np.concatenate(gens), np.concatenate(imps)


def search_method(method, X, y, split, X_unk):
    """Duyệt lưới C x gamma theo LOSO; chọn theo (EER thấp, AUC cao, C nhỏ)."""
    rows = []
    for C in C_GRID:
        for g in GAMMA_GRID:
            gen, imp = loso_scores(method, C, g, X, y, split, X_unk)
            eer, thr = compute_eer(gen, imp)
            rows.append({"C": C, "gamma": g, "eer": float(eer), "auc": auc_score(gen, imp),
                         "thr": float(thr), "n_gen": int(len(gen)), "n_imp": int(len(imp))})
    best = min(rows, key=lambda r: (round(r["eer"], 6), -round(r["auc"], 6), r["C"]))
    return best, rows


def check_command_leak(recs, X, y, C, gamma, random_val_acc):
    """Kiểm tra rò rỉ theo lệnh/phiên thu: val là những lệnh mô hình CHƯA thấy.

    Nếu độ chính xác tụt mạnh so với chia ngẫu nhiên, mô hình đang nhận ra lệnh/phiên thu
    chứ chưa chắc là giọng người.
    """
    groups = np.array([Path(r["path"]).parent.name for r in recs])
    n_groups = len(set(groups.tolist()))
    if n_groups < 3:
        print("  (bỏ qua kiểm tra rò rỉ theo lệnh: có ít hơn 3 thư mục lệnh)")
        return None
    accs = []
    for tr_i, va_i in GroupKFold(n_splits=min(5, n_groups)).split(X, y, groups):
        if len(set(y[tr_i].tolist())) < len(set(y.tolist())):
            continue
        m = make_svm(C, gamma, probability=False).fit(X[tr_i], y[tr_i])
        accs.append(float(m.score(X[va_i], y[va_i])))
    if not accs:
        return None
    mean_acc = float(np.mean(accs))
    print(f"Kiểm tra rò rỉ: val ngẫu nhiên = {random_val_acc:.3f} | "
          f"val là lệnh chưa thấy ({n_groups} lệnh, {len(accs)} fold) = {mean_acc:.3f}")
    if mean_acc < random_val_acc - 0.05:
        print("  ! Độ chính xác tụt rõ khi val là lệnh lạ: chia ngẫu nhiên có thể đang rò rỉ theo lệnh/phiên thu.")
    return {"random_val_acc": float(random_val_acc), "command_holdout_acc": mean_acc,
            "n_commands": n_groups, "n_folds": len(accs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-set", choices=["base", "plus"], default="plus")
    ap.add_argument("--unknown-dir", default=None, help="thư mục WAV giọng ngoài cho lớp unknown")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--no-leak-check", action="store_true", help="bỏ qua kiểm tra rò rỉ theo lệnh")
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

    # 2) Lưới closed-set trên tập val (chỉ để báo cáo và làm phương án dự phòng)
    grid, grid_acc, best_cs = [], {}, None
    for C in C_GRID:
        for g in GAMMA_GRID:
            m = make_svm(C, g, probability=False).fit(X[tr], y[tr])
            acc = float(m.score(X[va], y[va])) if len(va) else float(m.score(X[tr], y[tr]))
            grid.append({"C": C, "gamma": g, "val_acc": acc})
            grid_acc[(C, g)] = acc
            if best_cs is None or acc > best_cs[0] + 1e-9:
                best_cs = (acc, C, g)
    cs_acc, cs_C, cs_gamma = best_cs
    print(f"Closed-set: C={cs_C}, gamma={cs_gamma}, val_acc={cs_acc:.3f} "
          f"({sum(1 for r in grid if r['val_acc'] >= cs_acc - 1e-9)}/{len(grid)} tổ hợp đạt mức này)")

    leak = None
    if not args.no_leak_check:
        leak = check_command_leak(recs, X, y, cs_C, cs_gamma, cs_acc)

    # 3) Chọn siêu tham số và ngưỡng cho từng phương pháp bằng LOSO
    params, thresholds, loso_eer, loso_auc, loso_rows = {}, {}, {}, {}, {}
    if len(spks) < 3:
        print("  ! Chỉ có 2 người đăng ký nên không làm được leave-one-speaker-out; "
              "ngưỡng lấy theo phân vị 5% của điểm đúng người trên tập val (chỉ tham khảo).")
        for m in methods:
            Xf, yf = X[tr], y[tr]
            if m == "unknown":
                Xf = np.vstack([Xf, X_unk])
                yf = np.concatenate([yf, np.array([UNKNOWN] * len(X_unk))])
            model = build_model(m, cs_C, cs_gamma).fit(Xf, yf)
            names, sc = score_with(model, X[va], m)
            g_ = trial_scores(y[va], names, sc)
            g_ = g_[np.isfinite(g_)]
            params[m] = (cs_C, cs_gamma)
            thresholds[m] = float(np.percentile(g_, 5)) if len(g_) else 0.5
            loso_eer[m], loso_auc[m] = None, None
    else:
        print(f"Leave-one-speaker-out với {len(spks)} người (mỗi fold chỉ còn {len(spks) - 1} lớp).")
        if len(spks) == 3:
            print("  ! Mỗi fold chỉ có 2 lớp nên điểm người lạ giả lập kém tin cậy hơn; "
                  "có thêm người đăng ký sẽ cho ngưỡng đáng tin hơn.")
        for m in methods:
            best_m, rows = search_method(m, X, y, split, X_unk)
            params[m] = (best_m["C"], best_m["gamma"])
            thresholds[m] = best_m["thr"]
            loso_eer[m], loso_auc[m] = best_m["eer"], best_m["auc"]
            loso_rows[m] = rows
            n_tie = sum(1 for r in rows if abs(r["eer"] - best_m["eer"]) < 1e-9)
            print(f"  [{m:7s}] C={best_m['C']} gamma={best_m['gamma']} | "
                  f"EER={best_m['eer'] * 100:5.1f}%  AUC={best_m['auc']:.3f}  ngưỡng={best_m['thr']:.3f}  "
                  f"(đúng người={best_m['n_gen']}, người lạ giả lập={best_m['n_imp']}; "
                  f"{n_tie}/{len(rows)} tổ hợp hòa EER)")

    # 4) Chọn phương pháp theo số liệu LOSO (hòa thì theo METHOD_PRIOR)
    valid = [m for m in methods if loso_eer.get(m) is not None]
    if valid:
        chosen = min(valid, key=lambda m: (round(loso_eer[m], 6), -round(loso_auc[m], 6), METHOD_PRIOR[m]))
        reason = "LOSO EER thấp nhất, hòa thì AUC cao nhất, hòa nữa thì ưu tiên margin > unknown > proba"
    else:
        chosen = "margin"
        reason = "không đủ người để chạy LOSO; mặc định margin (chưa có số liệu để so sánh)"
    print(f"Phương pháp được chọn: {chosen} ({reason})")

    # 5) Huấn luyện mô hình cuối trên train + val
    fit_idx = tr + va
    models = {}
    for m in methods:
        Cm, gm = params[m]
        Xf, yf = X[fit_idx], y[fit_idx]
        if m == "unknown":
            Xf = np.vstack([Xf, X_unk])
            yf = np.concatenate([yf, np.array([UNKNOWN] * len(X_unk))])
        models[m] = build_model(m, Cm, gm).fit(Xf, yf)
        if len(te):
            names, _ = score_with(models[m], X[te], m)
            acc = float(np.mean(names == y[te]))
            print(f"  [{m:7s}] độ chính xác nhận diện người (tập test đăng ký) = {acc * 100:.1f}%")

    C, gamma = params[chosen]
    rel = lambda idxs: [recs[i]["rel"] for i in idxs]
    bundle = {
        "feature_set": args.feature_set, "feature_version": FEATURE_VERSION,
        "speakers": spks, "models": models, "method": chosen,
        "thresholds": thresholds, "params": {"C": C, "gamma": gamma},
        "params_by_method": {m: {"C": p[0], "gamma": p[1]} for m, p in params.items()},
        "split": {"train": rel(tr), "val": rel(va), "test": rel(te)},
        "seed": args.seed, "has_unknown_class": X_unk is not None,
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH, compress=3)

    report = {
        "feature_set": args.feature_set, "speakers": spks,
        "samples_per_speaker": {s: int((y == s).sum()) for s in spks},
        "split_sizes": {"train": len(tr), "val": len(va), "test": len(te)},
        "grid": grid, "best": {"C": C, "gamma": gamma, "val_acc": grid_acc.get((C, gamma))},
        "params_by_method": {m: {"C": p[0], "gamma": p[1]} for m, p in params.items()},
        "loso_eer": loso_eer, "loso_auc": loso_auc, "loso_search": loso_rows,
        "thresholds": thresholds, "chosen_method": chosen, "selection_reason": reason,
        "leak_check": leak, "seconds": round(time.time() - t0, 1),
    }
    (MODEL_PATH.parent / "speaker_train_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Đã lưu {MODEL_PATH.relative_to(ROOT)} và models/speaker_train_report.json")


if __name__ == "__main__":
    main()