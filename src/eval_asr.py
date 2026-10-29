#!/usr/bin/env python3
"""Đánh giá nhận dạng lệnh (câu 2.3 và 2.4): WER, CER, độ chính xác thiết bị/hành động/giá trị, ma trận nhầm lẫn, độ trễ.

Hai chế độ:

  1) Câu có tham số (testset/)   ->  câu 2.3 và so sánh PhoWhisper với Whisper
         python src/eval_asr.py --models phowhisper-small whisper-small
     Cần  testset/sentences.csv  (cột: id,text,device,action,value)
          testset/audio/<người>_<mã câu>_<số lần>.wav   (ví dụ vinh_s01_1.wav, la1_s07_2.wav)

  2) Lệnh đơn giản (dataset/)    ->  câu 2.4: so với baseline MFCC trên cùng tập test
         python src/eval_asr.py --simple --models phowhisper-small whisper-small --baseline
     Dùng đúng các file tập test trong features/manifest.csv (bỏ lớp mo_rem vì đã bỏ rèm cửa).

Kết quả ghi vào --out (mặc định models/asr_eval):
  summary.csv / summary.md   bảng so sánh các mô hình (dán thẳng vào báo cáo)
  report_<mô hình>.json      toàn bộ số liệu của một mô hình
  results_<mô hình>.csv      từng câu: văn bản chuẩn, văn bản ASR, đúng/sai (để soi lỗi)
  confusion_*.png/.csv       ma trận nhầm lẫn

Chạy từ thư mục gốc dự án.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
import commands  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

# lớp lệnh đơn giản (thư mục trong dataset/) -> (thiết bị, hành động) mong đợi; noise = phải bị từ chối
SIMPLE_LABELS = {
    "bat_dieu_hoa": ("ac", "on"), "tat_dieu_hoa": ("ac", "off"),
    "bat_den_tran": ("light-main", "on"), "tat_den_tran": ("light-main", "off"),
    "bat_tivi": ("tv", "on"), "tat_tivi": ("tv", "off"),
    "noise": (None, None),
}
SIMPLE_TEXT = {  # câu chuẩn của từng lớp, dùng để tính WER (chỉ tham khảo)
    "bat_dieu_hoa": "bật điều hòa", "tat_dieu_hoa": "tắt điều hòa", "bat_den_tran": "bật đèn trần",
    "tat_den_tran": "tắt đèn trần", "bat_tivi": "bật tivi", "tat_tivi": "tắt tivi",
}


# ------------------------------------------------------------------ khoảng cách sửa (WER/CER)
def edit_distance(ref: list, hyp: list) -> int:
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1]


def wer_cer(refs: list[str], hyps: list[str]) -> tuple[float, float]:
    """WER và CER gộp toàn tập (tổng lỗi / tổng độ dài tham chiếu). Số đọc bằng chữ và chữ số được coi là một."""
    we = wn = ce = cn = 0
    for r, h in zip(refs, hyps):
        rt, ht = commands.normalize_for_wer(r), commands.normalize_for_wer(h)
        we += edit_distance(rt, ht)
        wn += len(rt)
        rc, hc = " ".join(rt), " ".join(ht)
        ce += edit_distance(list(rc), list(hc))
        cn += len(rc)
    return (we / wn if wn else 0.0), (ce / cn if cn else 0.0)


# ------------------------------------------------------------------ nạp dữ liệu
def load_testset(sent_csv: Path, audio_dir: Path) -> list[dict]:
    """sentences.csv + file wav -> danh sách mẫu. File không có trong csv bị bỏ qua và báo ra."""
    sents = {}
    with open(sent_csv, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            v = (row.get("value") or "").strip()
            sents[row["id"].strip()] = dict(
                text=row["text"].strip(), device=(row.get("device") or "").strip() or None,
                action=(row.get("action") or "").strip() or None, value=int(float(v)) if v else None)
    items, unknown = [], []
    for p in sorted(audio_dir.glob("*.wav")):
        parts = p.stem.split("_")
        if len(parts) < 3:
            unknown.append(p.name); continue
        speaker, sid, rep = parts[0], "_".join(parts[1:-1]), parts[-1]
        if sid not in sents:
            unknown.append(p.name); continue
        items.append(dict(path=str(p), speaker=speaker, sid=sid, rep=rep, ref_text=sents[sid]["text"],
                          device=sents[sid]["device"], action=sents[sid]["action"], value=sents[sid]["value"]))
    if unknown:
        print(f"[cảnh báo] bỏ qua {len(unknown)} file không khớp mã câu trong csv: {unknown[:5]}{' ...' if len(unknown) > 5 else ''}")
    return items


def load_simple(manifest: Path, split: str) -> list[dict]:
    """Các file lệnh đơn giản thuộc `split` trong features/manifest.csv (cùng tập mà baseline được đo)."""
    items = []
    with open(manifest, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["split"] != split or row["command"] not in SIMPLE_LABELS:
                continue
            dev, act = SIMPLE_LABELS[row["command"]]
            rel = Path(row["file"].replace("\\", "/"))
            path = ROOT / rel
            if not path.exists():                  # nhánh phan-B đổi tên file: phat_001.wav -> phat_bat_tivi_001.wav
                alt = path.with_name(f"{row['speaker']}_{row['command']}_{path.stem.split('_')[-1]}.wav")
                if alt.exists():
                    path = alt
            items.append(dict(path=str(path), speaker=row["speaker"], sid=row["command"], rep="",
                              ref_text=SIMPLE_TEXT.get(row["command"], ""), device=dev, action=act, value=None))
    return items


# ------------------------------------------------------------------ đánh giá một mô hình
def evaluate_items(items: list[dict], transcribe, warmup: int = 1) -> tuple[list[dict], dict]:
    """Chạy `transcribe(path) -> text` trên từng mẫu, so với đáp án. Trả về (kết quả từng câu, số liệu tổng)."""
    rows, lat = [], []
    for k, it in enumerate(items):
        t0 = time.perf_counter()
        text = transcribe(it["path"])
        ms = (time.perf_counter() - t0) * 1000
        if k >= warmup:                      # bỏ lần đầu: có thể gồm thời gian khởi động
            lat.append(ms)
        import asr  # to_command nằm trong asr.py; import muộn để chế độ test không cần torch
        r = asr.to_command(text)
        ok_dev = r["device"] == it["device"]
        ok_act = r["action"] == it["action"]
        ok_val = (r["value"] == it["value"]) if it["value"] is not None else None
        rows.append(dict(it, hyp_text=text, pred_device=r["device"], pred_action=r["action"], pred_value=r["value"],
                         reason=r["reason"], score=r["score"], ok_device=ok_dev, ok_action=ok_act, ok_value=ok_val,
                         ok_exact=ok_dev and ok_act and r["value"] == it["value"] and r["reason"] is None
                         if it["device"] else (r["device"] is None), latency_ms=round(ms, 1)))
    return rows, summarize(rows, lat)


def _rate(xs: list) -> float | None:
    return round(sum(xs) / len(xs), 4) if xs else None


def summarize(rows: list[dict], lat: list[float]) -> dict:
    refs = [r["ref_text"] for r in rows]
    hyps = [r["hyp_text"] for r in rows]
    cmd_rows = [r for r in rows if r["device"]]                  # câu thật sự là lệnh (không tính lớp noise)
    wer, cer = wer_cer(refs, hyps) if all(refs) else (None, None)
    out = {
        "n": len(rows),
        "wer": None if wer is None else round(wer, 4), "cer": None if cer is None else round(cer, 4),
        "acc_device": _rate([r["ok_device"] for r in rows]),
        "acc_action": _rate([r["ok_action"] for r in rows]),
        "acc_value": _rate([r["ok_value"] for r in rows if r["ok_value"] is not None]),
        "n_value": sum(r["ok_value"] is not None for r in rows),
        "acc_exact": _rate([r["ok_exact"] for r in rows]),
        # thực thi sai = hệ thống ĐÃ chấp nhận một lệnh (reason rỗng) nhưng lệnh đó sai; nguy hiểm hơn việc từ chối nhầm
        "wrong_exec": _rate([r["reason"] is None and not r["ok_exact"] for r in rows]),
        "rejected_cmd": _rate([r["reason"] is not None for r in rows if r["device"]]),
        "n_noise": len(rows) - len(cmd_rows),
        "noise_rejected": _rate([r["pred_device"] is None for r in rows if not r["device"]]),
        "latency_ms_mean": round(statistics.mean(lat), 1) if lat else None,
        "latency_ms_median": round(statistics.median(lat), 1) if lat else None,
        "latency_ms_p95": round(sorted(lat)[int(0.95 * (len(lat) - 1))], 1) if lat else None,
        "reject_reasons": dict(Counter(r["reason"] for r in rows if r["reason"])),
    }
    by_spk = {}
    for spk in sorted({r["speaker"] for r in rows}):
        sub = [r for r in rows if r["speaker"] == spk]
        w, c = wer_cer([r["ref_text"] for r in sub], [r["hyp_text"] for r in sub]) if all(r["ref_text"] for r in sub) else (None, None)
        by_spk[spk] = {"n": len(sub), "wer": None if w is None else round(w, 4), "acc_exact": _rate([r["ok_exact"] for r in sub])}
    out["by_speaker"] = by_spk
    return out


# ------------------------------------------------------------------ ma trận nhầm lẫn
def combo(device, action) -> str:
    return "từ chối" if device is None else f"{device}:{action or '?'}"


def confusion(rows: list[dict], key_true, key_pred) -> tuple[list[str], list[list[int]]]:
    labels = sorted({key_true(r) for r in rows} | {key_pred(r) for r in rows}, key=lambda s: (s == "từ chối", s))
    idx = {l: i for i, l in enumerate(labels)}
    m = [[0] * len(labels) for _ in labels]
    for r in rows:
        m[idx[key_true(r)]][idx[key_pred(r)]] += 1
    return labels, m


def save_confusion(labels, m, title: str, png: Path, csv_path: Path) -> None:
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["thực tế \\ dự đoán"] + labels)
        for l, row in zip(labels, m):
            w.writerow([l] + row)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    M = np.array(m)
    norm = M / np.maximum(M.sum(axis=1, keepdims=True), 1)
    n = len(labels)
    fig, ax = plt.subplots(1, 2, figsize=(max(9, n * 1.5), max(4.2, n * 0.7 + 2)))
    for a, A, t, fmt in [(ax[0], M, "Số mẫu", "d"), (ax[1], norm, "Tỉ lệ theo hàng", ".2f")]:
        a.imshow(A, cmap="Blues", vmin=0)
        a.set_xticks(range(n)); a.set_xticklabels(labels, rotation=40, ha="right")
        a.set_yticks(range(n)); a.set_yticklabels(labels)
        a.set_xlabel("Dự đoán"); a.set_ylabel("Thực tế"); a.set_title(f"{title} – {t}")
        for i in range(n):
            for j in range(n):
                a.text(j, i, format(A[i, j], fmt), ha="center", va="center",
                       color="white" if A[i, j] > A.max() / 2 else "black", fontsize=8)
    fig.tight_layout(); fig.savefig(png, dpi=140); plt.close(fig)


# ------------------------------------------------------------------ baseline MFCC (câu 2.4)
def eval_baseline(items: list[dict], model_path: Path) -> dict | None:
    """Chạy baseline MFCC đã huấn luyện (models/command_model.joblib) trên đúng các file `items`."""
    try:
        import joblib
        import numpy as np
        sys.path.insert(0, str(ROOT / "src"))
        import preprocess as pp
        bundle = joblib.load(model_path)
    except Exception as e:  # thiếu thư viện hoặc chưa có mô hình
        print(f"[baseline] bỏ qua: {e}")
        return None
    ok, lat = [], []
    for it in items:
        y, _ = pp.load_wav(it["path"])
        t0 = time.perf_counter()
        r = pp.process_signal(y)
        pred = str(bundle["classes"][int(np.argmax(bundle["model"].predict_proba(r["stat"][None, :])[0]))])
        lat.append((time.perf_counter() - t0) * 1000)
        ok.append(pred == it["sid"])
    return {"name": str(bundle.get("name") or "baseline MFCC"), "n": len(ok),
            "acc": round(sum(ok) / len(ok), 4), "latency_ms_mean": round(statistics.mean(lat), 1)}


# ------------------------------------------------------------------ ghi kết quả
def pct(x) -> str:
    return "–" if x is None else f"{100 * x:.1f}%"


def write_outputs(name: str, rows, rep, out: Path, simple: bool) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fields = ["path", "speaker", "sid", "rep", "ref_text", "hyp_text", "device", "action", "value", "pred_device", "pred_action",
              "pred_value", "reason", "score", "ok_device", "ok_action", "ok_value", "ok_exact", "latency_ms"]
    with open(out / f"results_{name}.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    if simple:
        labels, m = confusion(rows, lambda r: r["sid"], lambda r: _label_of(r))
        save_confusion(labels, m, f"{name}: lệnh đơn giản", out / f"confusion_{name}_simple.png", out / f"confusion_{name}_simple.csv")
    else:
        for tag, kt, kp in [("device", lambda r: r["device"] or "từ chối", lambda r: r["pred_device"] or "từ chối"),
                            ("action", lambda r: combo(r["device"], r["action"]), lambda r: combo(r["pred_device"], r["pred_action"]))]:
            labels, m = confusion(rows, kt, kp)
            save_confusion(labels, m, f"{name}: {'thiết bị' if tag == 'device' else 'thiết bị:hành động'}",
                           out / f"confusion_{name}_{tag}.png", out / f"confusion_{name}_{tag}.csv")
    with open(out / f"report_{name}.json", "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)


def _label_of(r: dict) -> str:
    inv = {v: k for k, v in SIMPLE_LABELS.items()}
    return inv.get((r["pred_device"], r["pred_action"]), "khác" if r["pred_device"] else "noise")


def write_summary(reports: dict, out: Path, simple: bool, baseline: dict | None) -> None:
    cols = ["model", "n", "acc_exact", "wrong_exec", "latency_ms_mean"] if simple else \
        ["model", "n", "wer", "cer", "acc_device", "acc_action", "acc_value", "acc_exact", "wrong_exec", "latency_ms_mean"]
    rows = [{"model": m, **{c: r.get(c) for c in cols[1:]}} for m, r in reports.items()]
    if simple and baseline:
        rows.append({"model": baseline.get("name", "baseline MFCC"), "n": baseline["n"], "acc_exact": baseline["acc"],
                     "latency_ms_mean": baseline["latency_ms_mean"]})
    with open(out / "summary.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    head = {"model": "Mô hình", "n": "Số mẫu", "wer": "WER", "cer": "CER", "acc_device": "Đúng thiết bị", "acc_action": "Đúng hành động",
            "acc_value": "Đúng giá trị", "acc_exact": "Đúng cả lệnh" if not simple else "Đúng lệnh", "wrong_exec": "Thực thi sai", "latency_ms_mean": "Độ trễ TB (ms)"}
    lines = ["| " + " | ".join(head[c] for c in cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        cells = [str(r.get(c)) if c in ("model", "n") else (f"{r.get(c):.1f}" if c == "latency_ms_mean" and r.get(c) is not None else pct(r.get(c)))
                 for c in cols]
        lines.append("| " + " | ".join(cells) + " |")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["phowhisper-small", "whisper-small"], help=f"trong {list(config.ASR_MODELS)}")
    ap.add_argument("--simple", action="store_true", help="đo trên lệnh đơn giản (dataset/) thay vì testset/")
    ap.add_argument("--split", default="test", help="với --simple: tập nào trong manifest.csv (mặc định test)")
    ap.add_argument("--baseline", action="store_true", help="với --simple: đo thêm baseline MFCC trên cùng các file")
    ap.add_argument("--sentences", default="testset/sentences.csv")
    ap.add_argument("--audio", default="testset/audio")
    ap.add_argument("--manifest", default="features/manifest.csv")
    ap.add_argument("--baseline-model", default="models/command_model.joblib")
    ap.add_argument("--out", default="models/asr_eval")
    ap.add_argument("--limit", type=int, default=0, help="chỉ chạy N mẫu đầu (để thử nhanh)")
    a = ap.parse_args()

    items = load_simple(Path(a.manifest), a.split) if a.simple else load_testset(Path(a.sentences), Path(a.audio))
    if a.limit:
        items = items[:a.limit]
    if not items:
        raise SystemExit("Không có mẫu nào để đánh giá. Kiểm tra đường dẫn --sentences/--audio (hoặc --manifest với --simple).")
    print(f"{len(items)} mẫu, {len({i['speaker'] for i in items})} người nói: {sorted({i['speaker'] for i in items})}")

    import asr
    out = Path(a.out)
    reports = {}
    for name in a.models:
        print(f"\n=== {name} ===")
        model = asr.get_asr(name)
        rows, rep = evaluate_items(items, model.transcribe)
        rep["model"] = name
        reports[name] = rep
        write_outputs(name, rows, rep, out, a.simple)
        print(f"WER {pct(rep['wer'])} | CER {pct(rep['cer'])} | thiết bị {pct(rep['acc_device'])} | hành động {pct(rep['acc_action'])} | "
              f"giá trị {pct(rep['acc_value'])} | đúng cả lệnh {pct(rep['acc_exact'])} | {rep['latency_ms_mean']} ms/câu")
    baseline = eval_baseline(items, Path(a.baseline_model)) if (a.simple and a.baseline) else None
    write_summary(reports, out, a.simple, baseline)
    print(f"\nĐã ghi kết quả vào {out}/")


if __name__ == "__main__":
    main()
