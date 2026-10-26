#!/usr/bin/env python3
"""Nhận dạng lệnh bằng PhoWhisper: WAV -> văn bản -> {thiết bị, hành động, giá trị}.

PhoWhisper (VinAI) là Whisper được huấn luyện lại cho tiếng Việt, KHÔNG cần huấn luyện thêm:
chỉ tải mô hình có sẵn từ Hugging Face lần đầu chạy (khoảng 300 MB cho base, 1 GB cho small).

Hàm chính (đúng hợp đồng nhóm):
    recognize_command(wav) -> {"text", "device", "action", "value", "score",
                               "reason", "latency_ms", "model"}     # ba trường cuối là phần thêm
`wav` có thể là đường dẫn file, bytes của file WAV, hoặc mảng numpy (mono, 16 kHz).

Quan trọng khi ghép vào app.py:
  - mô hình chỉ nạp MỘT lần (cache theo tên); gọi `warmup()` lúc khởi động app để request đầu không bị chậm
  - decision.py phải kiểm tra `reason` TRƯỚC: nếu reason khác None thì không được thực thi lệnh

Thử nhanh:
    python src/asr.py dataset/bat_tivi/vinh_001.wav
    python src/asr.py file.wav --model phowhisper-small
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
import commands  # noqa: E402

SR = config.SAMPLE_RATE


# ------------------------------------------------------------------ đọc âm thanh
def load_audio(wav, sr: int | None = None) -> np.ndarray:
    """Đường dẫn / bytes / mảng numpy -> mảng float32 mono 16 kHz, biên độ trong [-1, 1].

    Không cần ffmpeg: đọc WAV bằng soundfile (hoặc scipy nếu thiếu), đổi tần số bằng librosa (hoặc scipy).
    """
    if isinstance(wav, np.ndarray):
        y, in_sr = wav.astype("float32"), (sr or SR)
    else:
        src = io.BytesIO(wav) if isinstance(wav, (bytes, bytearray)) else str(wav)
        try:
            import soundfile as sf
            y, in_sr = sf.read(src, dtype="float32", always_2d=True)
        except ImportError:
            from scipy.io import wavfile
            in_sr, y = wavfile.read(src)
            y = y.astype("float32") / (np.iinfo(y.dtype).max if np.issubdtype(y.dtype, np.integer) else 1.0)
            y = y.reshape(len(y), -1)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if in_sr != SR:
        try:
            import librosa
            y = librosa.resample(y, orig_sr=in_sr, target_sr=SR)
        except ImportError:
            from math import gcd
            from scipy.signal import resample_poly
            g = gcd(int(in_sr), SR)
            y = resample_poly(y, SR // g, int(in_sr) // g).astype("float32")
    return np.ascontiguousarray(y, dtype="float32")


# ------------------------------------------------------------------ mô hình
def resolve_model(name: str) -> str:
    """Tên ngắn (phowhisper-base) hoặc id Hugging Face (vinai/PhoWhisper-base) -> id Hugging Face."""
    return config.ASR_MODELS.get(name, name)


class ASR:
    """Bọc pipeline nhận dạng tiếng nói của Hugging Face. Dùng get_asr(name) để có bản cache, đừng tạo mới mỗi request."""

    def __init__(self, name: str = config.ASR_DEFAULT, device: str | None = None):
        import torch
        from transformers import pipeline

        self.name, self.hf_id = name, resolve_model(name)
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.pipe = pipeline("automatic-speech-recognition", model=self.hf_id, device=device, torch_dtype=dtype)
        # PhoWhisper chỉ có tiếng Việt nên không cần chỉ định ngôn ngữ; Whisper gốc thì phải ép tiếng Việt
        self.gen_kwargs = {} if "phowhisper" in self.hf_id.lower() else {"language": "vi", "task": "transcribe"}

    def transcribe(self, wav) -> str:
        y = load_audio(wav)
        out = self.pipe({"raw": y, "sampling_rate": SR}, generate_kwargs=self.gen_kwargs)
        return out["text"].strip()

    def transcribe_many(self, wavs: list, batch_size: int = 8) -> list[str]:
        """Nhận dạng nhiều file một lượt (nhanh hơn gọi từng file trên GPU)."""
        inputs = [{"raw": load_audio(w), "sampling_rate": SR} for w in wavs]
        outs = self.pipe(inputs, batch_size=batch_size, generate_kwargs=self.gen_kwargs)
        return [o["text"].strip() for o in outs]


_CACHE: dict[str, ASR] = {}


def get_asr(name: str = config.ASR_DEFAULT) -> ASR:
    if name not in _CACHE:
        t0 = time.perf_counter()
        _CACHE[name] = ASR(name)
        print(f"[asr] nạp {_CACHE[name].hf_id} trên {_CACHE[name].device}: {time.perf_counter() - t0:.1f}s", flush=True)
    return _CACHE[name]


def warmup(name: str = config.ASR_DEFAULT) -> None:
    """Gọi một lần khi khởi động app.py: nạp mô hình và chạy thử 1 giây im lặng để lần gọi đầu không bị chậm."""
    get_asr(name).transcribe(np.zeros(SR, dtype="float32"))


# ------------------------------------------------------------------ hàm theo hợp đồng
def to_command(text: str) -> dict:
    """Văn bản ASR -> kết quả lệnh, áp ngưỡng COMMAND_THRESHOLD. Tách riêng để eval_asr.py dùng lại."""
    r = commands.parse_command(text)
    if r["reason"] is None and r["score"] < config.COMMAND_THRESHOLD:
        r.update(device=None, action=None, value=None, reason="unknown_command")   # khớp quá yếu thì không đoán
    return r


def recognize_command(wav, model: str = config.ASR_DEFAULT) -> dict:
    t0 = time.perf_counter()
    text = get_asr(model).transcribe(wav)
    r = to_command(text)
    r["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    r["model"] = model
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Nhận dạng lệnh từ một file WAV")
    ap.add_argument("wav")
    ap.add_argument("--model", default=config.ASR_DEFAULT, help=f"một trong {list(config.ASR_MODELS)}")
    a = ap.parse_args()
    recognize_command(a.wav, a.model)                       # lần đầu gồm cả thời gian nạp mô hình
    print(json.dumps(recognize_command(a.wav, a.model), ensure_ascii=False, indent=2))
