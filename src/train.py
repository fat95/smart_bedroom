#!/usr/bin/env python3
"""Huấn luyện mô hình nhận dạng lệnh từ features/features.npz (tạo bởi preprocess.py).

Thử 3 họ mô hình trên đặc trưng thống kê MFCC (78 chiều), chọn theo độ chính xác trên tập val:
  - SVM (RBF), lưới C x gamma
  - k-NN
  - Random Forest
Mô hình tốt nhất được lưu vào models/command_model.joblib để evaluate.py và app.py dùng.

    python src/train.py --features features --out models
"""
import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import preprocess as pp


def svm(C, gamma, prob=False):
    return make_pipeline(StandardScaler(), SVC(kernel="rbf", C=C, gamma=gamma, probability=prob, random_state=0))


def knn(k):
    return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k))


def forest():
    return make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1))


def candidates():
    """(họ, tên, tham số, hàm tạo(prob))"""
    out = []
    for C in (1, 10, 100):
        for g in ("scale", 0.003, 0.01):
            out.append(("SVM", f"SVM-RBF C={C} gamma={g}", dict(C=C, gamma=g), lambda prob, C=C, g=g: svm(C, g, prob)))
    for k in (3, 5):
        out.append(("kNN", f"kNN k={k}", dict(k=k), lambda prob, k=k: knn(k)))
    out.append(("RandomForest", "Random Forest 300 cây", {}, lambda prob: forest()))
    return out


def load_features(path):
    d = np.load(path, allow_pickle=False)
    return {k: d[k] for k in d.files}


def predict_signal(bundle, y):
    """Dùng trong app.py: y là tín hiệu mono 16 kHz -> (nhãn, độ tin cậy, dict xác suất)."""
    r = pp.process_signal(y)
    proba = bundle["model"].predict_proba(r["stat"][None, :])[0]
    i = int(np.argmax(proba))
    return str(bundle["classes"][i]), float(proba[i]), {str(c): float(p) for c, p in zip(bundle["classes"], proba)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", default="features")
    ap.add_argument("--out", default="models")
    a = ap.parse_args()

    d = load_features(Path(a.features) / "features.npz")
    X, y, sp = d["X_stat"], d["y_cmd"], d["split"]
    tr, va = sp == "train", sp == "val"
    if tr.sum() == 0 or va.sum() == 0:
        raise SystemExit("Cần có cả tập train và val. Hãy thu thêm dữ liệu rồi chạy lại preprocess.py.")
    print(f"Train {tr.sum()} | Val {va.sum()} | Test {(sp == 'test').sum()} | {len(set(y))} lớp | {X.shape[1]} đặc trưng\n")

    results = []
    for fam, name, params, make in candidates():
        t0 = time.time()
        m = make(False).fit(X[tr], y[tr])
        acc = accuracy_score(y[va], m.predict(X[va]))
        results.append(dict(family=fam, name=name, params=params, val_acc=acc, fit_s=time.time() - t0, make=make))

    print(f"{'Mô hình':<28}{'Val acc':>10}{'Thời gian fit (s)':>20}")
    for r in sorted(results, key=lambda r: -r["val_acc"]):
        print(f"{r['name']:<28}{r['val_acc']:>10.3f}{r['fit_s']:>20.2f}")

    # mô hình tốt nhất của từng họ -> bảng so sánh cho báo cáo
    best_by_family = {}
    for r in results:
        if r["family"] not in best_by_family or r["val_acc"] > best_by_family[r["family"]]["val_acc"]:
            best_by_family[r["family"]] = r
    print("\nTốt nhất mỗi họ:")
    for r in sorted(best_by_family.values(), key=lambda r: -r["val_acc"]):
        print(f"  {r['name']:<28} val acc = {r['val_acc']:.3f}")

    best = max(results, key=lambda r: (r["val_acc"], r["family"] == "SVM"))  # hòa điểm thì ưu tiên SVM
    model = best["make"](True).fit(X[tr], y[tr])      # fit lại với predict_proba để có độ tin cậy
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    bundle = dict(model=model, classes=model.classes_, name=best["name"], params=best["params"], val_acc=best["val_acc"])
    joblib.dump(bundle, out / "command_model.joblib")
    with open(out / "train_report.json", "w", encoding="utf-8") as f:
        json.dump(dict(chosen=best["name"], val_acc=best["val_acc"],
                       all=[{k: (v if k != "params" else {kk: str(vv) for kk, vv in v.items()}) for k, v in r.items() if k != "make"} for r in results]),
                  f, ensure_ascii=False, indent=2)
    print(f"\nĐã chọn: {best['name']} (val acc {best['val_acc']:.3f}) -> {out / 'command_model.joblib'}")
    print("Chạy tiếp: python src/evaluate.py")


if __name__ == "__main__":
    main()