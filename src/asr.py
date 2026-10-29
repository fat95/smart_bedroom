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
    python src/asr.py dataset/bat_tivi/vinh_bat_tivi_001.wav
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
            from math import gcd
            from scipy.signal import resample_poly
            g = gcd(int(in_sr), SR)
            y = resample_poly(y, SR // g, int(in_sr) // g).astype("float32")
        except ImportError:
            import librosa
            y = librosa.resample(y, orig_sr=in_sr, target_sr=SR)
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

    PROMPT_TEXT = "bật tắt điều hòa, đèn trần, tivi, kênh, âm lượng, đầu giường, độ, phần trăm."

    def _prompt_ids(self):
        tokenizer = self.pipe.tokenizer
        if not hasattr(tokenizer, "get_prompt_ids"):
            return None
        ids = tokenizer.get_prompt_ids(self.PROMPT_TEXT, return_tensors="pt")
        return ids.to(self.pipe.model.device) if hasattr(ids, "to") else ids

    def _decode_direct(self, y: np.ndarray, decode: str = "greedy", prompt: bool = False) -> list[str]:
        """Decode trực tiếp để hỗ trợ beam/n-best; greedy vẫn đi qua pipeline cũ."""
        import torch

        fe = self.pipe.feature_extractor(y, sampling_rate=SR, return_tensors="pt")
        model_device = self.pipe.model.device
        model_dtype = next(self.pipe.model.parameters()).dtype
        model_inputs = {}
        for key, value in fe.items():
            if not hasattr(value, "to"):
                continue
            value = value.to(model_device)
            if torch.is_floating_point(value):
                value = value.to(model_dtype)
            model_inputs[key] = value
        kwargs = dict(self.gen_kwargs)
        if decode in {"nbest", "majority"}:
            kwargs.update(num_beams=5, num_return_sequences=5, do_sample=False)
        if prompt:
            ids = self._prompt_ids()
            if ids is not None:
                kwargs["prompt_ids"] = ids
        with torch.inference_mode():
            sequences = self.pipe.model.generate(**model_inputs, **kwargs)
        texts = self.pipe.tokenizer.batch_decode(sequences, skip_special_tokens=True)
        return [self._strip_prompt_text(t).strip() for t in texts]

    @classmethod
    def _strip_prompt_text(cls, text: str) -> str:
        """Xóa prompt nếu model chép nó ở đầu; không xóa các từ tương tự ở giữa lệnh."""
        value = (text or "").strip()
        prompt = cls.PROMPT_TEXT.strip()
        if value.casefold().startswith(prompt.casefold()):
            value = value[len(prompt):].lstrip(" ,.;:")
        return value

    def _select_nbest(self, texts: list[str], mode: str) -> str:
        parsed = []
        for rank, text in enumerate(texts):
            result = to_command(text)
            if result["reason"] is None and result["score"] >= config.COMMAND_THRESHOLD:
                key = (result["device"], result["action"], result["value"])
                parsed.append((rank, key, text))
        if not parsed:
            return ""
        if mode == "majority":
            from collections import Counter
            counts = Counter(key for _, key, _ in parsed)
            key, count = counts.most_common(1)[0]
            if count <= len(parsed) / 2:
                return ""
            return next(text for _, candidate, text in parsed if candidate == key)
        keys = {key for _, key, _ in parsed}
        if len(keys) != 1:
            return ""
        return parsed[0][2]

    def transcribe(self, wav, decode: str = "greedy", prompt: bool = False) -> str:
        y = load_audio(wav)
        if decode in {"nbest", "majority"} or prompt:
            texts = self._decode_direct(y, decode=decode, prompt=prompt)
            return self._select_nbest(texts, decode) if decode in {"nbest", "majority"} else texts[0]
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
