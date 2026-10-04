#!/usr/bin/env python3
"""Vẽ hình minh họa cho báo cáo (mục 2.3 – 2.5 của phiếu chấm).

Chạy sau khi đã có dataset/ và đã chạy preprocess.py:
    python src/visualize.py --data dataset --features features --out figures    

Hình tạo ra:
  fig1_pipeline.png          waveform thô -> tiền nhấn -> năng lượng/ZCR + VAD -> tín hiệu sau xử lý
  fig2_commands.png          mỗi lệnh một hàng: waveform, spectrogram, MFCC
  fig3_mfcc_delta.png        MFCC, delta, delta-delta của một mẫu
  fig4_dataset_summary.png   số mẫu theo lệnh/người, thời lượng tiếng nói, mức năng lượng
  fig5_pca.png               PCA của đặc trưng thống kê (tô theo lệnh và theo người nói)
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import preprocess as pp

LABELS = {"bat_dieu_hoa": "Bật điều hòa", "tat_dieu_hoa": "Tắt điều hòa", "bat_den_tran": "Bật đèn trần",
          "tat_den_tran": "Tắt đèn trần", "mo_rem": "Mở rèm", "bat_tivi": "Bật tivi", "noise": "Nền/im lặng"}
name = lambda c: LABELS.get(c, c)


def show_cep(ax, m, extent):
    """Vẽ hệ số cepstral: bỏ c0 (năng lượng, quá lớn nên làm phẳng các hệ số còn lại), trừ trung bình theo hệ số."""
    z = m[:, 1:] - m[:, 1:].mean(axis=0, keepdims=True)
    lim = np.percentile(np.abs(z), 98) + 1e-6
    return ax.imshow(z.T, origin="lower", aspect="auto", extent=extent, cmap="coolwarm", vmin=-lim, vmax=lim)


def first_file(data, cmd, speaker=None):
    files = sorted((Path(data) / cmd).glob(f"{speaker or '*'}_*.wav"))
    return files[0] if files else None


def fig_pipeline(path, out):
    y, _ = pp.load_wav(path)
    r = pp.process_signal(y)
    v = r["vad"]
    t = np.arange(len(y)) / pp.SR
    tf = np.arange(len(v["energy"])) * pp.HOP_LEN / pp.SR + 0.0125
    fig, ax = plt.subplots(4, 1, figsize=(10, 10))
    ax[0].plot(t, y, lw=.6); ax[0].set_title(f"1. Tín hiệu gốc – {path.name}")
    ax[1].plot(t, pp.pre_emphasis(y), lw=.6, color="tab:orange"); ax[1].set_title(f"2. Sau tiền nhấn (hệ số {pp.PREEMPH})")
    ax[2].plot(tf, v["energy"], label="Năng lượng (dB)"); ax[2].axhline(v["thr"], color="r", ls="--", lw=.8, label="Ngưỡng VAD")
    ax[2].set_ylabel("dB"); b = ax[2].twinx(); b.plot(tf, v["zcr"], color="tab:green", lw=.8, label="ZCR"); b.set_ylabel("ZCR")
    h1, l1 = ax[2].get_legend_handles_labels(); h2, l2 = b.get_legend_handles_labels()
    ax[2].legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8); ax[2].set_title("3. Năng lượng ngắn hạn + ZCR và vùng tiếng nói (VAD)")
    for a in (ax[0], ax[1], ax[2]):
        if v["found"]:
            a.axvspan(v["start"] / pp.SR, v["end"] / pp.SR, color="gold", alpha=.3)
        a.set_xlim(0, t[-1])
    ax[0].set_xlabel("Thời gian (s)"); ax[2].set_xlabel("Thời gian (s)")
    ax[3].plot(np.arange(len(r["signal"])) / pp.SR, r["signal"], lw=.6, color="tab:purple")
    ax[3].set_title("4. Sau cắt đoạn tiếng nói, chuẩn hóa biên độ, pad/cắt về 1,5 s"); ax[3].set_xlabel("Thời gian (s)")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def fig_commands(data, speaker, out):
    cmds = [d.name for d in sorted(Path(data).iterdir()) if d.is_dir() and first_file(data, d.name, speaker)]
    if not cmds:
        return False
    fig, ax = plt.subplots(len(cmds), 3, figsize=(14, 2.1 * len(cmds)), squeeze=False)
    for i, c in enumerate(cmds):
        y, _ = pp.load_wav(first_file(data, c, speaker)); sig = pp.process_signal(y)["signal"]
        spec = 10 * np.log10(pp.power_spectrogram(pp.pre_emphasis(sig)).T + 1e-10)
        m = pp.mfcc(sig)
        ax[i, 0].plot(np.arange(len(sig)) / pp.SR, sig, lw=.5); ax[i, 0].set_ylabel(name(c), rotation=0, ha="right", va="center")
        ax[i, 1].imshow(spec, origin="lower", aspect="auto", extent=[0, 1.5, 0, 8000], cmap="magma", vmin=spec.max() - 80)
        show_cep(ax[i, 2], m, [0, 1.5, 1, pp.N_MFCC])
    for j, t in enumerate(["Waveform", "Spectrogram (Hz)", "MFCC c1–c12 (đã trừ trung bình)"]):
        ax[0, j].set_title(t)
    for j in range(3):
        ax[-1, j].set_xlabel("Thời gian (s)")
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    return True


def fig_deltas(path, out):
    y, _ = pp.load_wav(path); seq = pp.process_signal(y)["seq"]
    fig, ax = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    for k, (title, sl) in enumerate([("MFCC", slice(0, 13)), ("Delta", slice(13, 26)), ("Delta-delta", slice(26, 39))]):
        im = show_cep(ax[k], seq[:, sl], [0, 1.5, 1, 13])
        ax[k].set_title(f"{title} – {path.name}"); ax[k].set_ylabel("Hệ số"); fig.colorbar(im, ax=ax[k])
    ax[-1].set_xlabel("Thời gian (s)")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def read_manifest(feat_dir):
    with open(Path(feat_dir) / "manifest.csv", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fig_summary(rows, out):
    cmds = sorted({r["command"] for r in rows}); spks = sorted({r["speaker"] for r in rows})
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    bottom = np.zeros(len(cmds))
    for s in spks:
        n = np.array([sum(1 for r in rows if r["command"] == c and r["speaker"] == s) for c in cmds])
        ax[0].bar([name(c) for c in cmds], n, bottom=bottom, label=s); bottom += n
    ax[0].set_title("Số mẫu theo lệnh và người nói"); ax[0].legend(title="Người nói"); ax[0].tick_params(axis="x", rotation=40)
    for a, key, t in [(ax[1], "speech_s", "Thời lượng đoạn tiếng nói sau VAD (s)"), (ax[2], "rms_db", "Mức năng lượng RMS (dB)")]:
        data = [[float(r[key]) for r in rows if r["command"] == c] for c in cmds]
        a.boxplot(data); a.set_xticklabels([name(c) for c in cmds]); a.set_title(t); a.tick_params(axis="x", rotation=40)
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)


def fig_pca(npz_path, out):
    d = np.load(npz_path); X = d["X_stat"]
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    _, S, Vt = np.linalg.svd(X, full_matrices=False)
    Z = X @ Vt[:2].T; var = (S ** 2) / np.sum(S ** 2)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
    for a, key, t in [(ax[0], "y_cmd", "Tô theo lệnh"), (ax[1], "y_spk", "Tô theo người nói")]:
        for lab in sorted(set(d[key])):
            m = d[key] == lab
            a.scatter(Z[m, 0], Z[m, 1], s=14, alpha=.7, label=name(lab) if key == "y_cmd" else lab)
        a.set_title(f"PCA đặc trưng MFCC thống kê – {t}"); a.set_xlabel(f"PC1 ({var[0]:.0%})"); a.set_ylabel(f"PC2 ({var[1]:.0%})"); a.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="dataset"); ap.add_argument("--features", default="features")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--speaker", default=None, help="chỉ lấy mẫu minh họa của người nói này")
    ap.add_argument("--cmd", default=None, help="lệnh dùng cho hình pipeline/delta (mặc định: lệnh đầu tiên khác noise)")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    cmds = [d.name for d in sorted(Path(a.data).iterdir()) if d.is_dir() and first_file(a.data, d.name, a.speaker)] if Path(a.data).is_dir() else []
    if not cmds:
        raise SystemExit("Chưa có dữ liệu trong dataset/ (hoặc không có mẫu của người nói đã chọn).")
    cmd = a.cmd or next((c for c in cmds if c != "noise"), cmds[0])
    sample = first_file(a.data, cmd, a.speaker)

    fig_pipeline(sample, out / "fig1_pipeline.png"); print("✓ fig1_pipeline.png")
    if fig_commands(a.data, a.speaker, out / "fig2_commands.png"): print("✓ fig2_commands.png")
    fig_deltas(sample, out / "fig3_mfcc_delta.png"); print("✓ fig3_mfcc_delta.png")
    if (Path(a.features) / "manifest.csv").exists():
        fig_summary(read_manifest(a.features), out / "fig4_dataset_summary.png"); print("✓ fig4_dataset_summary.png")
        fig_pca(Path(a.features) / "features.npz", out / "fig5_pca.png"); print("✓ fig5_pca.png")
    else:
        print("(bỏ qua fig4, fig5: chạy src/preprocess.py trước để có features/)")
    print(f"Hình đã lưu trong {out}/")


if __name__ == "__main__":
    main()