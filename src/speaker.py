# -*- coding: utf-8 -*-
"""
speaker.py - Xác thực người nói bằng MFCC + SVM (phần việc của B).

Ý tưởng: SVM chỉ học các người ĐÃ ĐĂNG KÝ. Khi có một câu nói mới, mô hình cho ra
(tên người đoán, điểm). Nếu điểm < ngưỡng thì coi là NGƯỜI LẠ -> hệ thống từ chối lệnh.

Quy ước dữ liệu (có thể đổi ở các hằng số bên dưới):
  dataset/<lệnh>/<người>_...wav        dữ liệu đăng ký (dùng để huấn luyện)
  testset/audio/<người>_...wav         câu có tham số, chỉ dùng để kiểm thử
  Tên người = phần trước dấu "_" đầu tiên, KHÔNG chứa "_".
  Người lạ = tên dạng la1, la2, ... HOẶC tên ghi trong strangers.txt (vd: vinh).
  Người lạ KHÔNG bao giờ được dùng để huấn luyện hay chọn ngưỡng.

Dùng trong app (hợp đồng với C):
    from speaker import verify_speaker
    res = verify_speaker(wav_bytes)   # hoặc đường dẫn file / mảng numpy
    # -> {"speaker": "vinh" | "unknown", "score": float, "is_enrolled": bool, ...}
"""
from __future__ import annotations

import io
import os
import re
from pathlib import Path

import joblib
import librosa
import numpy as np
import soundfile as sf
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# ----------------------------------------------------------------------------
# Cấu hình
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
SR = 16000
DATASET_DIR = ROOT / "dataset"
TESTSET_DIR = ROOT / "testset" / "audio"
MODEL_PATH = ROOT / "models" / "speaker_model.joblib"
CACHE_PATH = ROOT / "features" / "speaker_cache.joblib"
FIG_DIR = ROOT / "figures"

STRANGER_RE = re.compile(r"^la\d*$", re.I)     # la, la1, la2, ...
STRANGERS_FILE = ROOT / "strangers.txt"          # thêm tên người lạ khác (vd: vinh)
SKIP_DIRS = {"noise"}                          # thư mục không phải giọng nói
MIN_SPEECH_S = 0.3                             # ngắn hơn -> không đủ để xác thực
FEATURE_VERSION = 1                            # đổi khi sửa extract_features (xóa cache cũ)
UNKNOWN = "__unknown__"                        # nhãn lớp "người lạ" (nếu dùng)
METHODS = ("proba", "margin", "unknown")


# ----------------------------------------------------------------------------
# Tên file -> người nói
# ----------------------------------------------------------------------------
def speaker_of(path) -> str:
    """vinh_bat_dieu_hoa_03.wav -> 'vinh'; la1_c05_01.wav -> 'la1'."""
    return Path(path).stem.split("_", 1)[0].lower()


def load_strangers(path: Path = STRANGERS_FILE) -> set[str]:
    """Đọc strangers.txt (mỗi dòng một tên, # là ghi chú). Dùng cho người lạ không đặt tên la1, la2."""
    if not Path(path).exists():
        return set()
    names = set()
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        line = line.split("#", 1)[0].strip().lower()
        if line:
            names.add(line)
    return names


STRANGER_NAMES = load_strangers()


def is_stranger(name: str) -> bool:
    return bool(STRANGER_RE.match(name)) or name in STRANGER_NAMES


def _rel(path) -> str:
    p = Path(path).resolve()
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def make_rec(path, group: str, speaker: str | None = None) -> dict:
    name = speaker or speaker_of(path)
    return {"path": str(path), "rel": _rel(path), "speaker": name,
            "group": group, "stranger": is_stranger(name)}


def collect_files(dataset_dir: Path = DATASET_DIR, testset_dir: Path = TESTSET_DIR) -> list[dict]:
    recs = []
    if dataset_dir.exists():
        for sub in sorted(p for p in dataset_dir.iterdir() if p.is_dir() and p.name not in SKIP_DIRS):
            recs += [make_rec(w, "dataset") for w in sorted(sub.glob("*.wav"))]
    if testset_dir.exists():
        recs += [make_rec(w, "testset") for w in sorted(testset_dir.glob("*.wav"))]
    return recs


# ----------------------------------------------------------------------------
# Âm thanh và đặc trưng
# ----------------------------------------------------------------------------
def load_audio(audio, sr: int | None = None) -> np.ndarray:
    """Nhận đường dẫn, bytes WAV, file-like hoặc mảng numpy -> mono float32 16 kHz."""
    if isinstance(audio, (str, Path)):
        y, fs = sf.read(str(audio), dtype="float32", always_2d=False)
    elif isinstance(audio, (bytes, bytearray)):
        y, fs = sf.read(io.BytesIO(bytes(audio)), dtype="float32", always_2d=False)
    elif hasattr(audio, "read"):
        y, fs = sf.read(audio, dtype="float32", always_2d=False)
    else:
        y, fs = np.asarray(audio, dtype=np.float32), (sr or SR)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if fs != SR:
        y = librosa.resample(y, orig_sr=fs, target_sr=SR)
    return y.astype(np.float32)


def speech_part(y: np.ndarray) -> np.ndarray:
    """Giữ lại các đoạn có tiếng nói (cắt khoảng lặng)."""
    if y.size < 400 or float(np.max(np.abs(y))) < 0.01:
        return y[:0]
    intervals = librosa.effects.split(y, top_db=30, frame_length=400, hop_length=160)
    if len(intervals) == 0:
        return y[:0]
    return np.concatenate([y[a:b] for a, b in intervals])


def _f0_stats(v: np.ndarray) -> np.ndarray:
    """Trung vị và độ lệch của log F0 (ước lượng bằng YIN) trên các khung có năng lượng cao."""
    try:
        f0 = librosa.yin(v, fmin=60, fmax=400, sr=SR, frame_length=1024, hop_length=160)
        rms = librosa.feature.rms(y=v, frame_length=1024, hop_length=160)[0]
        n = min(len(f0), len(rms))
        f0, rms = f0[:n], rms[:n]
        ok = (rms > np.median(rms)) & (f0 > 62) & (f0 < 398)
        if ok.sum() < 5:
            return np.zeros(2)
        lf = np.log(f0[ok])
        return np.array([np.median(lf), lf.std()])
    except Exception:
        return np.zeros(2)


def extract_features(y: np.ndarray, feature_set: str = "plus"):
    """
    base: MFCC c1..c12, mean + std                                  (24 chiều)
    plus: MFCC c1..c19 mean + std, std của delta, log-F0 (median, std) (59 chiều)
    Không trừ trung bình (CMN) ở mức câu, vì mean của MFCC chính là thông tin người nói.
    Trả về None nếu đoạn tiếng nói quá ngắn.
    """
    v = speech_part(y)
    if v.size < MIN_SPEECH_S * SR:
        return None
    n_mfcc = 13 if feature_set == "base" else 20
    mf = librosa.feature.mfcc(y=v, sr=SR, n_mfcc=n_mfcc, n_fft=512,
                              hop_length=160, win_length=400, n_mels=40)[1:]
    parts = [mf.mean(axis=1), mf.std(axis=1)]
    if feature_set == "plus":
        delta = librosa.feature.delta(mf, width=9, mode="nearest")
        parts += [delta.std(axis=1), _f0_stats(v)]
    return np.concatenate(parts).astype(np.float32)


def build_matrix(recs: list[dict], feature_set: str = "plus", use_cache: bool = True):
    """Trích đặc trưng cho danh sách file (có cache). Trả về (X, danh sách rec giữ lại)."""
    cache = {}
    if use_cache and CACHE_PATH.exists():
        try:
            cache = joblib.load(CACHE_PATH)
        except Exception:
            cache = {}
    X, kept, skipped, dirty = [], [], 0, False
    for r in recs:
        p = Path(r["path"])
        st = p.stat()
        key = (feature_set, FEATURE_VERSION, str(p.resolve()))
        sig = (st.st_mtime_ns, st.st_size)
        hit = cache.get(key)
        if hit is not None and hit[0] == sig:
            vec = hit[1]
        else:
            try:
                vec = extract_features(load_audio(p), feature_set)
            except Exception as e:  # file hỏng
                print(f"  ! bỏ qua {r['rel']}: {e}")
                vec = None
            cache[key] = (sig, vec)
            dirty = True
        if vec is None:
            skipped += 1
            continue
        X.append(vec)
        kept.append(r)
    if dirty and use_cache:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(cache, CACHE_PATH)
    if skipped:
        print(f"  (bỏ qua {skipped} file quá ngắn hoặc không có tiếng nói)")
    return (np.vstack(X) if X else np.zeros((0, 0), dtype=np.float32)), kept


def split_indices(recs: list[dict], seed: int = 42, ratios=(0.70, 0.15, 0.15)) -> dict:
    """Chia train/val/test theo từng người đăng ký (cố định theo seed)."""
    rng = np.random.default_rng(seed)
    out = {"train": [], "val": [], "test": []}
    for spk in sorted({r["speaker"] for r in recs}):
        idx = np.array([i for i, r in enumerate(recs) if r["speaker"] == spk])
        rng.shuffle(idx)
        n = len(idx)
        n_va = max(1, int(round(n * ratios[1]))) if n >= 6 else 0
        n_te = max(1, int(round(n * ratios[2]))) if n >= 6 else 0
        n_tr = n - n_va - n_te
        out["train"] += idx[:n_tr].tolist()
        out["val"] += idx[n_tr:n_tr + n_va].tolist()
        out["test"] += idx[n_tr + n_va:].tolist()
    return out


# ----------------------------------------------------------------------------
# Mô hình và điểm số
# ----------------------------------------------------------------------------
def make_svm(C, gamma, probability: bool = True) -> Pipeline:
    return Pipeline([
        ("sc", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=C, gamma=gamma, probability=probability,
                    class_weight="balanced", random_state=0)),
    ])


def make_ovr(C, gamma) -> Pipeline:
    """Một-chống-tất-cả: cho điểm biên (margin) thật sự của từng người."""
    return Pipeline([
        ("sc", StandardScaler()),
        ("svm", OneVsRestClassifier(SVC(kernel="rbf", C=C, gamma=gamma, class_weight="balanced"))),
    ])


def build_model(method: str, C, gamma) -> Pipeline:
    return make_ovr(C, gamma) if method == "margin" else make_svm(C, gamma, probability=True)


def score_with(model, X, method: str):
    """
    Trả về (tên người đoán, điểm). Điểm càng cao càng chắc chắn là người đã đăng ký.
      proba  : xác suất lớn nhất của SVM (Platt)
      margin : điểm biên lớn nhất của SVM một-chống-tất-cả (decision_function)
      unknown: như proba nhưng có thêm lớp "người lạ"; đoán ra lớp đó thì điểm = 0
    """
    if method == "margin":
        D = model.decision_function(X)
        if D.ndim == 1:                      # chỉ 2 lớp
            D = np.column_stack([-D, D])
        idx = D.argmax(axis=1)
        return np.asarray(model.classes_)[idx], D.max(axis=1)
    P = model.predict_proba(X)
    idx = P.argmax(axis=1)
    names = np.asarray(model.classes_)[idx]
    sc = P.max(axis=1)
    if method == "unknown":
        sc = np.where(names == UNKNOWN, 0.0, sc)
    return names, sc


# ----------------------------------------------------------------------------
# Chỉ số xác thực: FAR, FRR, EER
# ----------------------------------------------------------------------------
def trial_scores(true_names, pred_names, scores) -> np.ndarray:
    """Điểm 'đúng người': đoán sai người thì coi như bị từ chối ở mọi ngưỡng (-inf)."""
    t = np.asarray(true_names)
    p = np.asarray(pred_names)
    return np.where(p == t, np.asarray(scores, dtype=float), -np.inf)


def far_frr(gen: np.ndarray, imp: np.ndarray, thr: float):
    far = float(np.mean(imp >= thr)) if len(imp) else float("nan")
    frr = float(np.mean(gen < thr)) if len(gen) else float("nan")
    return far, frr


def far_frr_curve(gen: np.ndarray, imp: np.ndarray):
    vals = np.unique(np.concatenate([gen[np.isfinite(gen)], imp]))
    if len(vals) == 0:
        return np.array([0.0]), np.array([0.0]), np.array([0.0])
    thr = np.concatenate([[vals.min() - 1e-6], vals, [vals.max() + 1e-6]])
    far = np.array([np.mean(imp >= t) for t in thr])
    frr = np.array([np.mean(gen < t) for t in thr])
    return thr, far, frr


def compute_eer(gen: np.ndarray, imp: np.ndarray):
    """Trả về (EER, ngưỡng tại EER)."""
    thr, far, frr = far_frr_curve(gen, imp)
    i = int(np.argmin(np.abs(far - frr)))
    return float((far[i] + frr[i]) / 2), float(thr[i])


def bootstrap_ci(gen, imp, fn, n: int = 1000, seed: int = 0):
    """Khoảng tin cậy 95% bằng bootstrap (lấy mẫu lại có hoàn lại). fn(gen, imp) -> số."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        g = gen[rng.integers(0, len(gen), len(gen))]
        i = imp[rng.integers(0, len(imp), len(imp))]
        vals.append(fn(g, i))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# ----------------------------------------------------------------------------
# API chạy thật (app.py gọi)
# ----------------------------------------------------------------------------
_BUNDLE = None


def load_bundle(path: Path = MODEL_PATH):
    global _BUNDLE
    if _BUNDLE is None:
        if not Path(path).exists():
            raise FileNotFoundError(
                f"Chưa có {Path(path).name}. Hãy chạy: python src/train_speaker.py")
        _BUNDLE = joblib.load(path)
    return _BUNDLE


def verify_speaker(audio, sr: int | None = None, method: str | None = None,
                   threshold: float | None = None) -> dict:
    """
    audio: đường dẫn, bytes WAV, file-like hoặc mảng numpy.
    Trả về {"speaker", "score", "is_enrolled", "best_guess", "threshold", "method"}.
    speaker = "unknown" nếu không vượt ngưỡng (người lạ) hoặc câu quá ngắn.
    """
    b = load_bundle()
    method = method or b["method"]
    y = load_audio(audio, sr)
    vec = extract_features(y, b["feature_set"])
    if vec is None:
        return {"speaker": "unknown", "score": 0.0, "is_enrolled": False,
                "best_guess": None, "threshold": None, "method": method, "reason": "too_short"}
    names, sc = score_with(b["models"][method], vec[None, :], method)
    thr = threshold
    if thr is None and os.environ.get("SPEAKER_THRESHOLD"):
        thr = float(os.environ["SPEAKER_THRESHOLD"])
    if thr is None:
        thr = b["thresholds"][method]
    guess, score = str(names[0]), float(sc[0])
    ok = guess != UNKNOWN and score >= thr
    return {"speaker": guess if ok else "unknown", "score": score, "is_enrolled": bool(ok),
            "best_guess": None if guess == UNKNOWN else guess,
            "threshold": float(thr), "method": method}