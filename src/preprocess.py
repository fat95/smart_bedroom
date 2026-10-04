#!/usr/bin/env python3
"""Tiền xử lý + trích đặc trưng cho đồ án Smart Bedroom (nhận dạng lệnh tiếng Việt).

Pipeline tự cài bằng NumPy/SciPy (không dùng librosa để tính MFCC):
  đọc WAV -> mono, 16 kHz -> VAD (năng lượng + ZCR) -> cắt đoạn tiếng nói
  -> chuẩn hóa biên độ -> pad/cắt về 1,5 s -> tiền nhấn -> phân khung 25/10 ms
  -> cửa sổ Hamming -> |FFT|^2 -> bộ lọc Mel -> log -> DCT -> MFCC(13) + delta + delta-delta

Chạy (từ thư mục gốc dự án):
    python src/preprocess.py --data dataset --out features

Đầu ra trong thư mục --out:
    features.npz   X_seq (N,148,39), X_stat (N,78), y_cmd, y_spk, split, files
    manifest.csv   mỗi dòng một file: nhãn, người nói, split, sr gốc, thời lượng, ...
Hàm process_signal() cũng dùng lại được trong app.py khi suy luận.
"""
import argparse
import csv
import random
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.fft import dct

SR = 16000
PREEMPH = 0.97
FRAME_LEN = int(0.025 * SR)   # 400 mẫu = 25 ms
HOP_LEN = int(0.010 * SR)     # 160 mẫu = 10 ms
N_FFT = 512
N_MELS = 26
N_MFCC = 13
FIX_LEN = int(1.5 * SR)       # độ dài cố định sau xử lý
MARGIN = int(0.05 * SR)       # chừa 50 ms hai đầu đoạn tiếng nói
PAD_FRAMES = 8                # mở rộng vùng tiếng nói 80 ms mỗi phía (bắt phụ âm yếu)
TARGET_RMS = 0.05              # chuẩn hóa RMS của đoạn tiếng nói (bền hơn chuẩn hóa đỉnh khi có tiếng 'tạch')


# ---------- Các bước xử lý tín hiệu ----------
def pre_emphasis(y, a=PREEMPH):
    """y[n] - a*y[n-1]: nhấn mạnh tần số cao."""
    return np.append(y[0], y[1:] - a * y[:-1])


def frame_signal(y, frame_len=FRAME_LEN, hop=HOP_LEN):
    if len(y) < frame_len:
        y = np.pad(y, (0, frame_len - len(y)))
    n = 1 + (len(y) - frame_len) // hop
    idx = np.arange(frame_len)[None, :] + hop * np.arange(n)[:, None]
    return y[idx]


def hamming(n):
    return 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(n) / (n - 1))


@lru_cache(maxsize=1)
def mel_filterbank():
    hz2mel = lambda f: 2595 * np.log10(1 + f / 700.0)
    mel2hz = lambda m: 700 * (10 ** (m / 2595.0) - 1)
    pts = mel2hz(np.linspace(0, hz2mel(SR / 2), N_MELS + 2))
    bins = np.floor((N_FFT + 1) * pts / SR).astype(int)
    fb = np.zeros((N_MELS, N_FFT // 2 + 1))
    for m in range(1, N_MELS + 1):
        l, c, r = bins[m - 1], bins[m], bins[m + 1]
        for k in range(l, c):
            fb[m - 1, k] = (k - l) / max(c - l, 1)
        for k in range(c, r):
            fb[m - 1, k] = (r - k) / max(r - c, 1)
    return fb


def power_spectrogram(y):
    """(số khung, N_FFT/2+1) – phổ công suất sau cửa sổ Hamming."""
    frames = frame_signal(y) * hamming(FRAME_LEN)
    return np.abs(np.fft.rfft(frames, n=N_FFT)) ** 2 / N_FFT


def mfcc(y):
    """(số khung, 13). Có tiền nhấn bên trong."""
    spec = power_spectrogram(pre_emphasis(y))
    logmel = np.log(spec @ mel_filterbank().T + 1e-10)
    return dct(logmel, type=2, axis=1, norm="ortho")[:, :N_MFCC]


def delta(feat, N=2):
    T = len(feat)
    padded = np.pad(feat, ((N, N), (0, 0)), mode="edge")
    denom = 2 * sum(n * n for n in range(1, N + 1))
    return sum(n * (padded[N + n:N + n + T] - padded[N - n:N - n + T]) for n in range(1, N + 1)) / denom


def detect_speech(y):
    """VAD theo năng lượng ngắn hạn + ZCR. Trả về dict (dùng cả để vẽ hình)."""
    fr = frame_signal(y)
    energy = 10 * np.log10(np.mean(fr ** 2, axis=1) + 1e-10)           # dB
    zcr = np.mean(np.abs(np.diff((fr >= 0).astype(np.int8), axis=1)), axis=1)
    noise = np.percentile(energy, 10)                                   # ước lượng nền nhiễu
    thr = max(noise + 10, energy.max() - 30)
    core = energy > thr
    out = dict(energy=energy, zcr=zcr, thr=thr, noise=noise, found=False, start=0, end=len(y))
    if not core.any():
        return out
    dil = np.convolve(core.astype(int), np.ones(2 * PAD_FRAMES + 1), mode="same") > 0
    ext = core | (dil & ((energy > noise + 6) | (zcr > 0.25)))         # phụ âm xát: năng lượng thấp, ZCR cao
    idx = np.flatnonzero(ext)
    out.update(found=True,
               start=max(0, idx[0] * HOP_LEN - MARGIN),
               end=min(len(y), idx[-1] * HOP_LEN + FRAME_LEN + MARGIN))
    return out


def fix_length(y, n=FIX_LEN):
    if len(y) >= n:
        s = (len(y) - n) // 2
        return y[s:s + n]
    left = (n - len(y)) // 2
    return np.pad(y, (left, n - len(y) - left))


def extract_features(y_fixed):
    """-> (seq (T,39), stat (78,))."""
    m = mfcc(y_fixed)
    seq = np.hstack([m, delta(m), delta(delta(m))])
    stat = np.concatenate([seq.mean(axis=0), seq.std(axis=0)])
    return seq.astype(np.float32), stat.astype(np.float32)


def process_signal(y):
    """Toàn bộ pipeline cho một tín hiệu 16 kHz mono. Dùng chung cho huấn luyện và suy luận."""
    vad = detect_speech(y)
    seg = y[vad["start"]:vad["end"]] if vad["found"] else y
    rms = float(np.sqrt(np.mean(seg ** 2))) if len(seg) else 0.0
    if vad["found"] and rms > 1e-4:                  # chỉ chuẩn hóa khi có tiếng nói (giữ nguyên mức của lớp noise)
        seg = np.clip(seg * (TARGET_RMS / rms), -1.0, 1.0)
    fixed = fix_length(seg)
    seq, stat = extract_features(fixed)
    return dict(signal=fixed, vad=vad, seq=seq, stat=stat)


# ---------- Đọc dữ liệu ----------
def load_wav(path):
    """-> (y mono 16 kHz float32, sr gốc)."""
    y, sr = sf.read(str(path), dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    if sr != SR:
        import librosa
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    return y, sr


def scan_dataset(data_dir):
    """dataset/<lệnh>/<người>_<số>.wav -> [(path, lệnh, người)]"""
    items = []
    for cdir in sorted(p for p in Path(data_dir).iterdir() if p.is_dir()):
        for f in sorted(cdir.glob("*.wav")):
            items.append((f, cdir.name, f.stem.split("_")[0]))
    return items


def assign_splits(items, seed=42, ratios=(0.7, 0.15, 0.15)):
    """Chia train/val/test phân tầng theo (lệnh, người nói)."""
    groups = defaultdict(list)
    for i, (_, c, s) in enumerate(items):
        groups[(c, s)].append(i)
    rng = random.Random(seed)
    split = [None] * len(items)
    for ids in groups.values():
        rng.shuffle(ids)
        n = len(ids)
        n_val = max(1, round(n * ratios[1])) if n >= 3 else 0
        n_test = max(1, round(n * ratios[2])) if n >= 3 else 0
        for k, i in enumerate(ids):
            split[i] = "val" if k < n_val else "test" if k < n_val + n_test else "train"
    return split


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--out", default="features")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    if not Path(a.data).is_dir():
        raise SystemExit(f"Không thấy thư mục dữ liệu: {a.data}")
    items = scan_dataset(a.data)
    if not items:
        raise SystemExit("Chưa có file .wav nào trong dataset/<lệnh>/")
    splits = assign_splits(items, a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    seqs, stats, rows = [], [], []
    for n, ((path, cmd, spk), sp) in enumerate(zip(items, splits), 1):
        y, sr0 = load_wav(path)
        r = process_signal(y)
        v = r["vad"]
        seqs.append(r["seq"]); stats.append(r["stat"])
        rows.append(dict(
            file=str(path), command=cmd, speaker=spk, split=sp, orig_sr=sr0,
            duration_s=round(len(y) / SR, 3),
            speech_s=round((v["end"] - v["start"]) / SR, 3) if v["found"] else 0.0,
            peak=round(float(np.max(np.abs(y))), 4),
            rms_db=round(float(20 * np.log10(np.sqrt(np.mean(y ** 2)) + 1e-10)), 2),
            vad_found=int(v["found"]),
            tail_zero_s=round(float(np.argmax(np.abs(y[::-1]) > 1e-6)) / SR if np.any(np.abs(y) > 1e-6) else len(y) / SR, 3)))
        if n % 50 == 0:
            print(f"  đã xử lý {n}/{len(items)} file...")

    np.savez_compressed(out / "features.npz",
                        X_seq=np.stack(seqs), X_stat=np.stack(stats),
                        y_cmd=np.array([r["command"] for r in rows]),
                        y_spk=np.array([r["speaker"] for r in rows]),
                        split=np.array(splits), files=np.array([r["file"] for r in rows]))
    with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # ---- Thống kê dùng cho slide mục 2.1, 2.2 ----
    cmds = sorted({r["command"] for r in rows}); spks = sorted({r["speaker"] for r in rows})
    cnt = Counter((r["command"], r["speaker"]) for r in rows)
    print("\nSố mẫu theo lệnh x người nói")
    print(f"{'lệnh':<16}" + "".join(f"{s:>8}" for s in spks) + f"{'tổng':>8}")
    for c in cmds:
        print(f"{c:<16}" + "".join(f"{cnt[(c, s)]:>8}" for s in spks) + f"{sum(cnt[(c, s)] for s in spks):>8}")
    print(f"{'TỔNG':<16}" + "".join(f"{sum(cnt[(c, s)] for c in cmds):>8}" for s in spks) + f"{len(rows):>8}")
    print(f"\nChia tập: {dict(Counter(splits))}")
    print(f"Tần số lấy mẫu gốc: {sorted({r['orig_sr'] for r in rows})} Hz (đã đưa về {SR} Hz)")
    print(f"Tổng thời lượng: {sum(r['duration_s'] for r in rows) / 60:.1f} phút; "
          f"thời lượng mỗi file: {min(r['duration_s'] for r in rows)}–{max(r['duration_s'] for r in rows)} s")
    miss = [r["file"] for r in rows if not r["vad_found"] and r["command"] != "noise"]
    if miss:
        print(f"\nCảnh báo: VAD không thấy tiếng nói trong {len(miss)} file lệnh (nên nghe lại/thu lại):")
        for m in miss[:10]:
            print("  ", m)
    tz = [r["file"] for r in rows if r["tail_zero_s"] > 0.1]
    if tz:
        print(f"\nCảnh báo: {len(tz)} file có đuôi toàn số 0 > 0,1 s (bản thu bị hụt, có thể mất cuối câu lệnh) – nên thu lại:")
        for m in tz[:10]:
            print("  ", m)
    print(f"\nĐã lưu: {out / 'features.npz'}  |  {out / 'manifest.csv'}  |  X_seq {np.stack(seqs).shape}, X_stat {np.stack(stats).shape}")


if __name__ == "__main__":
    main()