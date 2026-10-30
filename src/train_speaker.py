# -*- coding: utf-8 -*-
"""
train_speaker.py - Huấn luyện SVM xác thực người nói và chọn ngưỡng từ chối người lạ.

Chạy:  python src/train_speaker.py
       python src/train_speaker.py --feature-set base            # so sánh đặc trưng (câu 1.3)
       python src/train_speaker.py --unknown-dir external_voices  # thêm lớp "unknown" từ giọng ngoài

Chia dữ liệu theo PHIÊN (mặc định khi có >= 2 phiên câu dài trong dataset/):
  * Phiên được suy ra từ tên thư mục: dataset/cd05_p2/ -> phiên p2; thư mục lệnh đơn giản -> p0.
  * val = một phiên mà mô hình chưa thấy (mặc định phiên câu dài mới nhất đủ mẫu cho mọi người,
    đổi bằng --val-session p2). Siêu tham số, ngưỡng LOSO và điểm "đúng người" đều lấy trên val này.
  * train/test = các phiên còn lại, test chia ngẫu nhiên cùng điều kiện với train.
  * Mô hình cuối vẫn fit trên train + val, nên mọi phiên đều được dùng.
  * Chưa đủ 2 phiên câu dài thì tự chia ngẫu nhiên như cũ (kèm cảnh báo: val lạc quan).

Quy tắc chống rò rỉ dữ liệu:
  * Chỉ đọc dataset/ của người ĐÃ ĐĂNG KÝ. Người lạ (la1, la2, ...) không được đọc ở đây.
  * Ngưỡng chọn bằng leave-one-speaker-out (LOSO): lần lượt coi 1 người đăng ký là "người lạ".
    Khi chia theo phiên, điểm người lạ lấy từ cùng phiên val với điểm đúng người (điều kiện tương đương).
    Trong mỗi fold, người bị bỏ KHÔNG xuất hiện ở bất kỳ dạng nào trong dữ liệu huấn luyện
    (kể cả dưới nhãn unknown), nếu không điểm người lạ sẽ đẹp một cách giả tạo (EER = 0%).
  * Chỉ phương pháp "unknown" mới có lớp unknown, và lớp đó chỉ chứa giọng ngoài (--unknown-dir).
  * Người lạ thật chỉ xuất hiện ở eval_speaker.py và eval_ablation.py.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import GroupKFold

from speaker import (BASE_SESSION, MIN_SESSION_SAMPLES, MODEL_PATH, ROOT, UNKNOWN, FEATURE_VERSION,
                     build_matrix, build_model, candidate_val_sessions, collect_files, compute_eer,
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


def loso_scores(method, C, gamma, X, y, split, X_unk=None, imp_pool=None):
    """Điểm 'đúng người' (gen) và điểm người lạ (imp), gộp qua các fold leave-one-speaker-out.

    Mỗi fold: người s bị bỏ hoàn toàn khỏi huấn luyện. Điểm người lạ = điểm của MỌI mẫu của s
    (an toàn vì mô hình chưa từng thấy s). Điểm đúng người = mẫu val của những người còn lại.
    imp_pool: nếu có (tập chỉ số), điểm người lạ chỉ lấy từ các mẫu của s nằm trong tập này; khi chia
    theo phiên dùng chính tập val để điểm đúng người và điểm người lạ cùng điều kiện (cùng phiên, cùng loại câu).
    """
    gens, imps = [], []
    for s in sorted(set(y.tolist())):
        tr = [i for i in split["train"] if y[i] != s]
        va = [i for i in split["val"] if y[i] != s]
        imp_idx = np.where(y == s)[0]
        if imp_pool is not None:
            imp_idx = np.array([i for i in imp_idx if i in imp_pool], dtype=int)
        if len(imp_idx) == 0 or len(va) == 0:
            continue

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


def search_method(method, X, y, split, X_unk, imp_pool=None):
    """Duyệt lưới C x gamma theo LOSO; chọn theo (EER thấp, AUC cao, C nhỏ)."""
    rows = []
    for C in C_GRID:
        for g in GAMMA_GRID:
            gen, imp = loso_scores(method, C, g, X, y, split, X_unk, imp_pool)
            eer, thr = compute_eer(gen, imp)
            rows.append({"C": C, "gamma": g, "eer": float(eer), "auc": auc_score(gen, imp),
                         "thr": float(thr), "n_gen": int(len(gen)), "n_imp": int(len(imp))})
    best = min(rows, key=lambda r: (round(r["eer"], 6), -round(r["auc"], 6), r["C"]))
    return best, rows


def choose_val_session(recs, mode, wanted):
    """Trả về tên phiên dùng làm val, hoặc None nghĩa là chia ngẫu nhiên."""
    if mode == "random":
        return None
    sessions = sorted({r["session"] for r in recs} - {BASE_SESSION}, key=lambda x: int(x[1:]))
    cands = candidate_val_sessions(recs)
    if wanted:
        if wanted not in cands:
            raise SystemExit(f"Phiên {wanted} không đủ mẫu cho mọi người (cần >= {MIN_SESSION_SAMPLES} mẫu mỗi người). "
                             f"Phiên đủ điều kiện: {', '.join(cands) or 'không có'}.")
        return wanted
    if len(sessions) >= 2 and cands:
        return cands[-1]
    msg = (f"Chưa đủ để chia theo phiên: có {len(sessions)} phiên câu dài ({', '.join(sessions) or 'không có'}), "
           f"phiên đủ mẫu cho mọi người: {', '.join(cands) or 'không có'} "
           f"(cần >= 2 phiên câu dài và >= {MIN_SESSION_SAMPLES} mẫu mỗi người ở phiên val).")
    if mode == "session":
        raise SystemExit(msg)
    print("  ! " + msg + " Chuyển sang chia ngẫu nhiên: val cùng phiên với train nên lạc quan hơn thực tế.")
    return None


def check_command_leak(recs, X, y, C, gamma, ref_acc, ref_name):
    """Kiểm tra rò rỉ theo lệnh/câu: val là những lệnh/câu mô hình CHƯA thấy (nhóm theo thư mục).

    Nếu độ chính xác tụt mạnh so với val hiện tại, mô hình đang nhận ra lệnh/câu
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
    print(f"Kiểm tra rò rỉ: {ref_name} = {ref_acc:.3f} | "
          f"val là lệnh/câu chưa thấy ({n_groups} nhóm, {len(accs)} fold) = {mean_acc:.3f}")
    if mean_acc < ref_acc - 0.05:
        print("  ! Độ chính xác tụt rõ khi val là lệnh/câu lạ: mô hình có thể đang nhận ra nội dung câu.")
    return {"ref_name": ref_name, "val_acc": float(ref_acc), "command_holdout_acc": mean_acc,
            "n_commands": n_groups, "n_folds": len(accs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-set", choices=["base", "plus"], default="plus")
    ap.add_argument("--unknown-dir", default=None, help="thư mục WAV giọng ngoài cho lớp unknown")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--no-leak-check", action="store_true", help="bỏ qua kiểm tra rò rỉ theo lệnh")
    ap.add_argument("--split", choices=["auto", "session", "random"], default="auto",
                    help="auto: chia theo phiên nếu có >= 2 phiên câu dài, không thì ngẫu nhiên; "
                         "session: bắt buộc theo phiên; random: ngẫu nhiên như cũ")
    ap.add_argument("--val-session", default=None,
                    help="phiên làm val (vd p2); mặc định là phiên câu dài mới nhất đủ mẫu cho mọi người")
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

    sess_cnt = Counter((r["speaker"], r["session"]) for r in recs)
    sess_all = sorted({r["session"] for r in recs}, key=lambda x: (x != BASE_SESSION, int(x[1:])))
    print("Mẫu theo phiên (p0 = lệnh ngắn gốc): "
          + " | ".join(f"{s}: " + ", ".join(f"{p}={sess_cnt[(s, p)]}" for p in sess_all) for s in spks))

    val_session = choose_val_session(recs, args.split, args.val_session)
    split = split_indices(recs, seed=args.seed, val_session=val_session)
    tr, va, te = split["train"], split["val"], split["test"]
    for s in spks:
        n_tr = sum(1 for i in tr if y[i] == s)
        if n_tr < 10:
            raise SystemExit(f"{s} chỉ còn {n_tr} mẫu huấn luyện sau khi chia; hãy thu thêm hoặc dùng --split random.")
    if val_session:
        print(f"Chia theo PHIÊN: val = {val_session}; train/test = các phiên còn lại "
              f"(test chia ngẫu nhiên, cùng điều kiện với train)")
    else:
        print("Chia NGẪU NHIÊN theo từng người (val cùng phiên với train)")
    print(f"Chia dữ liệu: train={len(tr)}, val={len(va)}, test={len(te)}")
    imp_pool = set(va) if val_session else None

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
        leak = check_command_leak(recs, X, y, cs_C, cs_gamma, cs_acc,
                                  f"val theo phiên {val_session}" if val_session else "val ngẫu nhiên")

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
            best_m, rows = search_method(m, X, y, split, X_unk, imp_pool)
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
        "split_mode": "session" if val_session else "random", "val_session": val_session,
        "seed": args.seed, "has_unknown_class": X_unk is not None,
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH, compress=3)

    report = {
        "feature_set": args.feature_set, "speakers": spks,
        "samples_per_speaker": {s: int((y == s).sum()) for s in spks},
        "split_sizes": {"train": len(tr), "val": len(va), "test": len(te)},
        "split_info": {"mode": "session" if val_session else "random", "val_session": val_session,
                       "samples_by_session": {s: {p: int(sess_cnt[(s, p)]) for p in sess_all} for s in spks}},
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