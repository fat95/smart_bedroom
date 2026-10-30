# -*- coding: utf-8 -*-
"""
diag_speaker.py - Chẩn đoán vì sao EER cao: lệch miền giữa dataset/ và testset/ hay do đặc trưng.

Chạy:  python src/diag_speaker.py          (cần models/speaker_model.joblib từ train_speaker.py)

Chỉ để CHẨN ĐOÁN. Phần 4 có dùng testset để huấn luyện nên KHÔNG đưa con số đó vào báo cáo.
"""
from __future__ import annotations

from collections import Counter, defaultdict

import librosa
import numpy as np
import soundfile as sf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from speaker import (SR, build_matrix, collect_files, load_audio, load_bundle,
                     make_svm, score_with, speech_part, UNKNOWN)


def pool_of(r):
    return ("stranger_" if r["stranger"] else "enrolled_") + r["group"]


def audio_props(path):
    y = load_audio(path)
    fr = librosa.feature.rms(y=y, frame_length=400, hop_length=160)[0]
    lo, hi = float(np.percentile(fr, 10)), float(np.percentile(fr, 90))
    return {"sr": sf.info(str(path)).samplerate, "dur": len(y) / SR,
            "speech": len(speech_part(y)) / SR, "peak": float(np.abs(y).max()),
            "rms": float(np.sqrt(np.mean(y ** 2))),
            "snr_db": 20 * np.log10((hi + 1e-9) / (lo + 1e-9))}


def main():
    b = load_bundle()
    recs = collect_files()
    X, recs = build_matrix(recs, b["feature_set"], True)
    pools = np.array([pool_of(r) for r in recs])
    spk = np.array([r["speaker"] for r in recs])
    print(f"{len(recs)} file | pool: " + ", ".join(f"{p}={int((pools == p).sum())}" for p in sorted(set(pools))))

    # 1) Tính chất âm thanh theo pool
    print("\n[1] Trung vị tính chất âm thanh theo pool (khác nhau nhiều = nghi lệch kênh/phiên thu)")
    props = defaultdict(list)
    for r, p in zip(recs, pools):
        props[p].append(audio_props(r["path"]))
    keys = ["sr", "dur", "speech", "peak", "rms", "snr_db"]
    print(f"{'pool':<22}" + "".join(f"{k:>9}" for k in keys))
    for p in sorted(props):
        print(f"{p:<22}" + "".join(f"{np.median([d[k] for d in props[p]]):>9.3f}" for k in keys))

    # 2) Đặc trưng có tách được dataset và testset của CÙNG một người không?
    print("\n[2] AUC phân biệt dataset vs testset trong cùng một người (1.0 = đặc trưng chứa dấu vết phiên thu)")
    for s in sorted(set(spk[[not r["stranger"] for r in recs]])):
        idx = np.where((spk == s) & np.isin(pools, ["enrolled_dataset", "enrolled_testset"]))[0]
        lab = (pools[idx] == "enrolled_testset").astype(int)
        if min(lab.sum(), (1 - lab).sum()) < 5:
            print(f"  {s}: thiếu mẫu ở một pool"); continue
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        pr = cross_val_predict(clf, X[idx], lab, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                               method="predict_proba")[:, 1]
        print(f"  {s:<10} AUC = {roc_auc_score(lab, pr):.3f}  (dataset={int((1 - lab).sum())}, testset={int(lab.sum())})")

    # 3) Mô hình đang dự đoán gì cho từng nhóm?
    method = b["method"]
    thr = b["thresholds"][method]
    names, sc = score_with(b["models"][method], X, method)
    print(f"\n[3] Dự đoán của mô hình ({method}, ngưỡng {thr:.3f}); 'dưới ngưỡng' = bị từ chối")
    for p in sorted(set(pools)):
        for s in sorted(set(spk[pools == p])):
            m = (pools == p) & (spk == s)
            guess = Counter(str(n) if sc_ >= thr else "<từ chối>" for n, sc_ in zip(names[m], sc[m]))
            print(f"  {p:<20} {s:<8} n={int(m.sum()):>3} điểm TB={sc[m].mean():6.3f} | "
                  + ", ".join(f"{k}:{v}" for k, v in guess.most_common()))

    # 4) Nếu train thấy cả hai pool thì sao? (chỉ để chẩn đoán, KHÔNG báo cáo)
    print("\n[4] Chẩn đoán: huấn luyện trên nửa mỗi (người, pool), thi nửa còn lại của testset")
    en = np.where(~np.array([r["stranger"] for r in recs]))[0]
    rng = np.random.default_rng(0)
    tr_i, te_i = [], []
    for s in set(spk[en]):
        for p in ("enrolled_dataset", "enrolled_testset"):
            idx = [i for i in en if spk[i] == s and pools[i] == p]
            rng.shuffle(idx)
            h = len(idx) // 2
            tr_i += idx[:h]
            te_i += idx[h:] if p == "enrolled_testset" else []
    C, g = b["params"]["C"], b["params"]["gamma"]
    m = make_svm(C, g, probability=False).fit(X[tr_i], spk[tr_i])
    print(f"  độ chính xác trên testset khi train thấy cả hai pool = {m.score(X[te_i], spk[te_i]) * 100:.1f}%")
    only = [i for i in tr_i if pools[i] == "enrolled_dataset"]
    m0 = make_svm(C, g, probability=False).fit(X[only], spk[only])
    print(f"  cùng tập thi, nhưng train chỉ có dataset      = {m0.score(X[te_i], spk[te_i]) * 100:.1f}%")
    print("  (chênh lệch lớn = vấn đề là độ phủ dữ liệu đăng ký / lệch miền, không phải thuật toán)")


if __name__ == "__main__":
    main()