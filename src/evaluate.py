#!/usr/bin/env python3
"""Đánh giá mô hình nhận dạng lệnh trên tập test (kết quả thực nghiệm cho mục 3.4).

    python src/evaluate.py                      # accuracy, báo cáo theo lớp, ma trận nhầm lẫn, độ trễ
    python src/evaluate.py --loso               # thêm: huấn luyện bỏ một người nói, thử trên người đó
    python src/evaluate.py --snr 20 10 5 0      # thêm: thêm nhiễu trắng ở các mức SNR (dB) vào file test
Chạy từ thư mục gốc dự án (manifest.csv lưu đường dẫn tương đối).
"""
import argparse
import csv
import json
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import preprocess as pp
from train import load_features
from visualize import LABELS

name = lambda c: LABELS.get(c, c)


def add_noise(y, snr_db, rng):
    """Thêm nhiễu trắng; SNR tính so với công suất đoạn tiếng nói (theo VAD)."""
    v = pp.detect_speech(y)
    seg = y[v["start"]:v["end"]] if v["found"] else y
    p = float(np.mean(seg ** 2)) + 1e-12
    return (y + rng.standard_normal(len(y)) * np.sqrt(p / 10 ** (snr_db / 10))).astype(np.float32)


def plot_confusion(cm, classes, out):
    labels = [name(c) for c in classes]
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.8))
    for a, M, t, fmt in [(ax[0], cm, "Số mẫu", "d"), (ax[1], norm, "Tỉ lệ theo hàng", ".2f")]:
        a.imshow(M, cmap="Blues", vmin=0)
        a.set_xticks(range(len(classes))); a.set_xticklabels(labels, rotation=40, ha="right")
        a.set_yticks(range(len(classes))); a.set_yticklabels(labels)
        a.set_xlabel("Dự đoán"); a.set_ylabel("Thực tế"); a.set_title(f"Ma trận nhầm lẫn – {t}")
        for i in range(len(classes)):
            for j in range(len(classes)):
                a.text(j, i, format(M[i, j], fmt), ha="center", va="center", color="white" if M[i, j] > M.max() / 2 else "black", fontsize=9)
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", default="features"); ap.add_argument("--model", default="models/command_model.joblib")
    ap.add_argument("--out", default="figures"); ap.add_argument("--loso", action="store_true")
    ap.add_argument("--snr", type=float, nargs="*", default=None); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    bundle = joblib.load(a.model); model = bundle["model"]; classes = list(bundle["classes"])
    d = load_features(Path(a.features) / "features.npz")
    X, y, spk, sp, files = d["X_stat"], d["y_cmd"], d["y_spk"], d["split"], d["files"]
    te = sp == "test"
    if te.sum() == 0:
        raise SystemExit("Không có mẫu test.")
    pred = model.predict(X[te])
    acc = accuracy_score(y[te], pred)
    report = dict(model=bundle["name"], n_test=int(te.sum()), test_acc=float(acc))
    print(f"Mô hình: {bundle['name']}\nTập test: {te.sum()} mẫu  ->  ACCURACY = {acc:.3f}\n")
    print(classification_report(y[te], pred, labels=classes, target_names=[name(c) for c in classes], digits=3, zero_division=0))

    cm = confusion_matrix(y[te], pred, labels=classes)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    plot_confusion(cm, classes, Path(a.out) / "fig6_confusion_matrix.png")
    off = [(cm[i, j], classes[i], classes[j]) for i in range(len(classes)) for j in range(len(classes)) if i != j and cm[i, j] > 0]
    if off:
        print("Các cặp hay nhầm nhất (thực tế -> dự đoán):")
        for n, t, p in sorted(off, reverse=True)[:5]:
            print(f"  {name(t)} -> {name(p)}: {n} lần")
        print()
    print("Accuracy theo người nói (tập test):")
    report["by_speaker"] = {}
    for s in sorted(set(spk[te])):
        m = spk[te] == s
        report["by_speaker"][str(s)] = float(accuracy_score(y[te][m], pred[m]))
        print(f"  {s:<10} {report['by_speaker'][str(s)]:.3f}  ({m.sum()} mẫu)")

    proba = model.predict_proba(X[te]); conf = proba.max(axis=1); ok = pred == y[te]
    report["mean_conf_correct"] = float(conf[ok].mean()) if ok.any() else None
    report["mean_conf_wrong"] = float(conf[~ok].mean()) if (~ok).any() else None
    fmt = lambda v: "n/a" if v is None else f"{v:.2f}"
    print(f"\nĐộ tin cậy trung bình: đúng = {fmt(report['mean_conf_correct'])}, sai = {fmt(report['mean_conf_wrong'])}")

    # Độ trễ end-to-end: đọc file + tiền xử lý + MFCC + dự đoán
    ts = []
    for f in files[te][:30]:
        t0 = time.perf_counter(); yy, _ = pp.load_wav(f); r = pp.process_signal(yy); model.predict_proba(r["stat"][None, :]); ts.append(time.perf_counter() - t0)
    report["latency_ms"] = float(np.mean(ts) * 1000)
    print(f"Độ trễ xử lý mỗi mẫu (đọc file -> nhãn): {report['latency_ms']:.1f} ms (trung bình {len(ts)} mẫu)")

    if a.loso:
        print("\nLeave-one-speaker-out (huấn luyện bỏ một người, thử trên người đó):")
        report["loso"] = {}
        for s in sorted(set(spk)):
            m = spk == s
            if m.all():
                continue
            mdl = clone(model).fit(X[~m], y[~m])
            report["loso"][str(s)] = float(accuracy_score(y[m], mdl.predict(X[m])))
            print(f"  bỏ {s:<10} acc = {report['loso'][str(s)]:.3f}")
        if report["loso"]:
            print(f"  trung bình = {np.mean(list(report['loso'].values())):.3f}")

    if a.snr:
        print("\nẢnh hưởng của nhiễu trắng (file test, thêm nhiễu vào tín hiệu gốc):")
        rng = np.random.default_rng(a.seed); report["snr"] = {}
        clean = [pp.load_wav(f)[0] for f in files[te]]
        for snr in a.snr:
            p = [model.predict(pp.process_signal(add_noise(yy, snr, rng))["stat"][None, :])[0] for yy in clean]
            report["snr"][str(snr)] = float(accuracy_score(y[te], p))
            print(f"  SNR {snr:>5.0f} dB: acc = {report['snr'][str(snr)]:.3f}")
        xs = [f"{k} dB" for k in report["snr"]]
        plt.figure(figsize=(6, 4)); plt.plot(xs, list(report["snr"].values()), "o-"); plt.ylim(0, 1.02)
        plt.xlabel("SNR"); plt.ylabel("Accuracy"); plt.title("Accuracy theo mức nhiễu"); plt.grid(alpha=.3); plt.tight_layout()
        plt.savefig(Path(a.out) / "fig7_noise_robustness.png", dpi=140); plt.close()

    Path(a.model).parent.joinpath("eval_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nĐã lưu: {Path(a.out) / 'fig6_confusion_matrix.png'}, {Path(a.model).parent / 'eval_report.json'}")


if __name__ == "__main__":
    main()