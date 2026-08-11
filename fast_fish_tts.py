import os
import sys
import subprocess
import warnings
import queue
import threading
from contextlib import nullcontext

# Suppress noisy framework warnings (TensorFlow, Pydantic, Transformers)
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# -----------------------------------------------------------------------------
# 0A. GUARANTEED SYS.PATH INJECTION (Must be Line 1 before any imports)
# -----------------------------------------------------------------------------
_file_dir = os.path.dirname(os.path.abspath(__file__))
_repo_dir = os.path.join(_file_dir, "fish-speech") if os.path.exists(os.path.join(_file_dir, "fish-speech")) else _file_dir

for _p in ["/content/fish-speech", _repo_dir, _file_dir, os.getcwd()]:
    if os.path.exists(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

# -----------------------------------------------------------------------------
# 0B. Self-Healing Automatic Dependency Installer
# -----------------------------------------------------------------------------
REQUIRED_PACKAGES = [
    ("flatten_dict", "flatten-dict"),
    ("tokenizers", "tokenizers"),
    ("julius", "julius"),
    ("positional", "positional"),
    ("argbind", "argbind"),
    ("yaml", "pyyaml"),
    ("ffmpy", "ffmpy"),
    ("pydub", "pydub"),
    ("randomname", "randomname"),
    ("librosa", "librosa"),
    ("soundfile", "soundfile"),
    ("scipy", "scipy"),
    ("resampy", "resampy"),
    ("fastapi", "fastapi"),
    ("uvicorn", "uvicorn"),
    ("pydantic", "pydantic"),
    ("loguru", "loguru"),
    ("pyrootutils", "pyrootutils"),
    ("hydra", "hydra-core"),
    ("lightning", "lightning"),
    ("pytorch_lightning", "pytorch-lightning"),
    ("torchmetrics", "torchmetrics"),
    ("loralib", "loralib"),
    ("websockets", "websockets"),
    ("ormsgpack", "ormsgpack"),
    ("requests", "requests"),
    ("huggingface_hub", "huggingface_hub"),
    ("transformers", "transformers"),
]

for mod_name, pip_name in REQUIRED_PACKAGES:
    try:
        __import__(mod_name)
    except ImportError:
        try:
            print(f"📦 Auto-installing missing dependency '{pip_name}'...", flush=True)
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_name])
        except Exception as err:
            print(f"⚠️ Warning: Auto-install of '{pip_name}' skipped: {err}", flush=True)

try:
    import audiotools
except ImportError:
    try:
        print("📦 Auto-installing descript-audiotools and descript-audio-codec...", flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "descript-audiotools", "descript-audio-codec"])
    except Exception:
        pass


import argparse
import asyncio
import io
import re
import time
import traceback
from pathlib import Path
from typing import AsyncGenerator, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from loguru import logger
from pydantic import BaseModel, Field

# Ensure root import path
import pyrootutils
try:
    pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
except Exception:
    pass

from fish_speech.conversation import Conversation, Message, TextPart, VQPart
from fish_speech.inference_engine.utils import wav_chunk_header
from fish_speech.models.text2semantic.inference import (
    GenerateRequest,
    GenerateResponse,
    WrappedGenerateResponse,
    generate_long,
    launch_thread_safe_queue,
)

# Defensive import for load_model from text2semantic
try:
    from fish_speech.models.text2semantic.inference import load_model as load_llama_model
except Exception:
    load_llama_model = None

# Defensive imports for decoder loader & architecture
try:
    from fish_speech.models.vqgan.inference import load_model as load_decoder_model
except Exception:
    try:
        from fish_speech.models.dac.inference import load_model as load_decoder_model
    except Exception:
        from hydra.utils import instantiate
        from omegaconf import OmegaConf
        def load_decoder_model(config_name, checkpoint_path, device):
            cfg = OmegaConf.load(f"fish_speech/configs/{config_name}.yaml")
            model = instantiate(cfg.model)
            state_dict = torch.load(checkpoint_path, map_location=device)
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            model.load_state_dict(state_dict)
            model.eval().to(device)
            return model

try:
    from fish_speech.models.vqgan.modules.firefly import FireflyArchitecture
except Exception:
    FireflyArchitecture = None

from fish_speech.text import clean_text
from fish_speech.utils import set_seed

AMPLITUDE = 32768


# -----------------------------------------------------------------------------
# 0C. Text Preprocessing & Sentence Splitter Helper
# -----------------------------------------------------------------------------
def split_text_into_sentences(text: str, max_chunk_len: int = 80) -> List[str]:
    """
    Splits text by sentence boundaries (।, ., !, ?, \n) into smaller chunks
    so long prompts achieve < 1.0s TTFA on the first sentence.
    """
    if len(text.strip()) <= max_chunk_len:
        return [text.strip()]
    parts = re.split(r'([।.!?\n]+)', text)
    sentences = []
    curr = ""
    for p in parts:
        if not p:
            continue
        curr += p
        if re.search(r'[।.!?\n]', p) or len(curr) >= max_chunk_len:
            if curr.strip():
                sentences.append(curr.strip())
            curr = ""
    if curr.strip():
        sentences.append(curr.strip())
    return sentences if sentences else [text.strip()]


# -----------------------------------------------------------------------------
# 1. Native FastFishTTS Engine Class
# -----------------------------------------------------------------------------
class FastFishTTS:
    def __init__(
        self,
        llama_checkpoint: str = "checkpoints/s2-pro",
        decoder_checkpoint: str = "checkpoints/s2-pro/codec.pth",
        decoder_config: str = "modded_dac_vq",
        device: str = "cuda",
        half: bool = True,
        compile_model: bool = False,
        num_workers: int = 4,
    ):
        """
        Native High-Performance Fish Speech S2-Pro TTS Engine.
        """
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.precision = torch.half if half and self.device == "cuda" else torch.bfloat16
        self.compile_model = compile_model

        # Optimize PyTorch CUDA Settings
        if self.device == "cuda":
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            torch.backends.cudnn.benchmark = True
            torch.set_float32_matmul_precision("high")

        os.environ["NUM_WORKERS"] = str(num_workers)

        logger.info(f"⚡ Loading Llama DualAR Queue ({num_workers} Workers) on {self.device}...")
        self.llama_queue = launch_thread_safe_queue(
            checkpoint_path=llama_checkpoint,
            device=self.device,
            precision=self.precision,
            compile=compile_model,
        )

        logger.info("⚡ Loading Firefly DAC VQGAN Decoder...")
        self.decoder = load_decoder_model(
            config_name=decoder_config,
            checkpoint_path=decoder_checkpoint,
            device=self.device,
        )
        self.sample_rate = getattr(self.decoder, "sample_rate", getattr(getattr(self.decoder, "spec_transform", None), "sample_rate", 44100))

        # In-Memory Voice Prompt Cache: { voice_id: (prompt_tokens_list, prompt_texts_list) }
        self.voice_cache: Dict[str, Tuple[List[torch.Tensor], List[str]]] = {}

        logger.info("🎉 FastFishTTS Engine v3.0.0 loaded successfully into GPU memory!")

    def register_reference_voice(
        self,
        voice_id: str,
        audio_path_or_bytes: Union[str, bytes],
        reference_text: Optional[str] = None,
    ):
        """
        Pre-encodes a reference audio sample into VRAM/RAM cache.
        Subsequent TTS calls with `reference_id=voice_id` will have ZERO encoding delay!
        """
        logger.info(f"⚡ Pre-encoding reference voice '{voice_id}' into RAM/VRAM Cache...")
        
        # Load audio data
        if isinstance(audio_path_or_bytes, bytes):
            audio_data, sr = sf.read(io.BytesIO(audio_path_or_bytes), dtype="float32")
        else:
            audio_data, sr = sf.read(audio_path_or_bytes, dtype="float32")

        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)

        # Resample to model sample rate if needed
        if sr != self.sample_rate:
            import resampy
            audio_data = resampy.resample(audio_data, sr, self.sample_rate)

        tensor_audio = torch.from_numpy(audio_data).to(self.device)[None, None, :]
        audio_len = torch.tensor([tensor_audio.shape[2]], device=self.device, dtype=torch.long)

        # Encode VQ tokens via Firefly Architecture
        with torch.inference_mode():
            prompt_tokens = self.decoder.encode(tensor_audio, audio_len)[0][0]

        ref_text = reference_text or ""
        self.voice_cache[voice_id] = ([prompt_tokens], [ref_text])
        logger.info(f"✅ Voice '{voice_id}' registered & cached successfully! Token shape: {prompt_tokens.shape}")

    def preload_references_dir(self, references_dir: str = "references"):
        """Scans directory containing voice folders (with audio & lab text files) and pre-caches them."""
        ref_path = Path(references_dir)
        if not ref_path.exists():
            ref_path.mkdir(parents=True, exist_ok=True)
            return

        count = 0
        for folder in ref_path.iterdir():
            if folder.is_dir():
                voice_id = folder.name
                audio_files = list(folder.glob("*.wav")) + list(folder.glob("*.mp3")) + list(folder.glob("*.flac"))
                if audio_files:
                    audio_file = audio_files[0]
                    lab_file = audio_file.with_suffix(".lab")
                    text_content = lab_file.read_text(encoding="utf-8").strip() if lab_file.exists() else ""
                    try:
                        self.register_reference_voice(voice_id, str(audio_file), text_content)
                        count += 1
                    except Exception as e:
                        logger.warning(f"Failed to pre-load reference voice '{voice_id}': {e}")
        if count > 0:
            logger.info(f"🎉 Pre-loaded {count} reference voices into memory cache.")

    def _decode_vq_tokens(self, codes: torch.Tensor, feature_len: torch.Tensor) -> torch.Tensor:
        """
        Universal & Defensive VQ Token Decoder (Handles FireflyArchitecture, DAC, modded_dac & audiotools).
        """
        # 1. Try DAC / modded_dac from_indices signature: from_indices(indices)
        if hasattr(self.decoder, "from_indices"):
            out = self.decoder.from_indices(codes)
            return out[0] if isinstance(out, tuple) else out

        # 2. Try DAC decode_code signature: decode_code(codes)
        if hasattr(self.decoder, "decode_code"):
            out = self.decoder.decode_code(codes)
            return out[0] if isinstance(out, tuple) else out

        # 3. Try FireflyArchitecture decode signature: decode(indices=codes, feature_lengths=feature_len)
        if hasattr(self.decoder, "decode"):
            try:
                out = self.decoder.decode(indices=codes, feature_lengths=feature_len)
                return out[0] if isinstance(out, tuple) else out
            except TypeError:
                try:
                    out = self.decoder.decode(codes, feature_len)
                    return out[0] if isinstance(out, tuple) else out
                except TypeError:
                    pass

        raise TypeError(f"Decoder of type '{type(self.decoder).__name__}' has no compatible decode method for shape {codes.shape}")

def split_text_into_sentences(text: str, max_chunk_len: int = 80) -> List[str]:
    """
    Splits text by sentence boundaries (।, ., !, ?, \\n) into smaller chunks
    so long prompts achieve < 1.0s TTFA on the first sentence.
    """
    if len(text.strip()) <= max_chunk_len:
        return [text.strip()]
    parts = re.split(r'([।.!?\n]+)', text)
    sentences = []
    curr = ""
    for p in parts:
        if not p:
            continue
        curr += p
        if re.search(r'[।.!?\n]', p) or len(curr) >= max_chunk_len:
            if curr.strip():
                sentences.append(curr.strip())
            curr = ""
    if curr.strip():
        sentences.append(curr.strip())
    return sentences if sentences else [text.strip()]


    @torch.inference_mode()
    def generate_stream(
        self,
        text: str,
        reference_id: Optional[str] = None,
        max_new_tokens: int = 1024,
        chunk_length: int = 80,
        top_p: float = 0.7,
        temperature: float = 0.7,
        repetition_penalty: float = 1.2,
        seed: Optional[int] = None,
    ) -> Generator[bytes, None, None]:
        """
        Native Sentence-Streaming Generator: Yields WAV header at 0ms, followed by 
        sentence-by-sentence PCM audio chunks. Sub-1-second TTFA even on long text!
        """
        if seed is not None:
            set_seed(seed)

        prompt_tokens, prompt_texts = None, None
        if reference_id and reference_id in self.voice_cache:
            prompt_tokens, prompt_texts = self.voice_cache[reference_id]

        sentences = split_text_into_sentences(text, max_chunk_len=chunk_length if chunk_length > 0 else 80)
        logger.info(f"⚡ [SENTENCE STREAM] Split text into {len(sentences)} sentence chunks for instant TTFA")

        # Yield 44-byte WAV header first for instant streaming playback
        yield wav_chunk_header(sample_rate=self.sample_rate)

        feature_len = torch.tensor([0], device=self.device)

        for s_idx, sentence_text in enumerate(sentences):
            # Dynamic max new tokens for each sentence chunk
            sent_max_tokens = max(64, min(len(sentence_text) * 4, max_new_tokens))

            req_dict = dict(
                device=self.device,
                max_new_tokens=sent_max_tokens,
                text=sentence_text,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                temperature=temperature,
                compile=self.compile_model,
                iterative_prompt=True,
                chunk_length=chunk_length,
                prompt_tokens=prompt_tokens,
                prompt_text=prompt_texts,
            )

            response_queue = import_queue()
            self.llama_queue.put(GenerateRequest(request=req_dict, response_queue=response_queue))

            while True:
                wrapped = response_queue.get()
                if wrapped.status == "error":
                    logger.error(f"❌ Generation queue error: {wrapped.response}")
                    raise RuntimeError(f"Generation error: {wrapped.response}")

                res = wrapped.response
                if res.action != "next":
                    codes = res.codes
                    if codes is not None and codes.shape[-1] > 0:
                        if codes.ndim == 2:
                            indices = codes[None].to(self.device)
                        else:
                            indices = codes.to(self.device)

                        feature_len[0] = indices.shape[-1]
                        try:
                            ctx = torch.cuda.amp.autocast(dtype=self.precision) if self.device == "cuda" else nullcontext()
                            with ctx:
                                decoded_tensor = self._decode_vq_tokens(indices, feature_len).view(-1)

                            pcm_data = (decoded_tensor.float().cpu().numpy() * AMPLITUDE).astype(np.int16).tobytes()
                            if len(pcm_data) > 0:
                                yield pcm_data
                        except Exception as dec_err:
                            logger.error(f"❌ Decoder Error: {dec_err}\n{traceback.format_exc()}")
                            raise dec_err
                else:
                    break

    @torch.inference_mode()
    def generate(
        self,
        text: str,
        reference_id: Optional[str] = None,
        max_new_tokens: int = 1024,
        chunk_length: int = 150,
        format: str = "wav",
        **kwargs,
    ) -> bytes:
        """
        Generates full complete audio bytes (non-streaming).
        """
        chunks = []
        stream = self.generate_stream(text, reference_id=reference_id, max_new_tokens=max_new_tokens, chunk_length=chunk_length, **kwargs)
        # Skip the 44-byte WAV chunk header from stream
        next(stream, None)
        for chunk in stream:
            chunks.append(chunk)

        raw_pcm = b"".join(chunks)
        audio_np = np.frombuffer(raw_pcm, dtype=np.int16).astype(np.float32) / AMPLITUDE

        buffer = io.BytesIO()
        sf.write(buffer, audio_np, self.sample_rate, format=format)
        return buffer.getvalue()


def import_queue():
    import queue
    return queue.Queue()


# -----------------------------------------------------------------------------
# 2. Standalone Fast API Server
# -----------------------------------------------------------------------------
app = FastAPI(
    title="FastFishTTS Native Server",
    version="3.1.0-dev",
    description="Clean, native, low-latency Fish Speech S2-Pro API Server (Sentence Chunking Enabled).",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

tts_engine: Optional[FastFishTTS] = None


import base64 as _base64
import hashlib as _hashlib


class ReferenceAudioModel(BaseModel):
    audio: str = Field(..., description="Base64-encoded audio bytes")
    text: str = Field("", description="Transcript of the reference audio (improves cloning quality)")
    mime_type: str = Field("audio/wav", description="MIME type of the audio")


class RegisterVoiceRequest(BaseModel):
    voice_id: str = Field(..., description="Unique name for this voice (e.g. 'prakash_confident')")
    audio: str = Field(..., description="Base64-encoded audio bytes")
    text: str = Field("", description="Transcript of the reference audio")
    mime_type: str = Field("audio/wav", description="MIME type of the audio")


class TTSRequestModel(BaseModel):
    text: str = Field(..., json_schema_extra={"example": "नमस्ते"})
    reference_id: Optional[str] = Field(None, description="Pre-cached voice ID (from /v1/voices/register)")
    references: Optional[List[ReferenceAudioModel]] = Field(None, description="Inline audio reference (only needed if reference_id not yet cached)")
    max_new_tokens: int = Field(1024)
    chunk_length: int = Field(80)
    streaming: bool = Field(True)
    format: str = Field("wav")
    top_p: float = Field(0.7)
    temperature: float = Field(0.7)
    repetition_penalty: float = Field(1.2)
    seed: Optional[int] = Field(None)


@app.get("/v1/health")
@app.post("/v1/health")
async def health():
    return {"status": "ok", "engine": "FastFishTTS Native v3.1.0-dev (Sentence Chunking)"}


@app.get("/v1/voices")
async def list_voices():
    voices = list(tts_engine.voice_cache.keys()) if tts_engine else []
    return {"status": "ok", "cached_voice_count": len(voices), "cached_voices": voices}


@app.post("/v1/voices/register")
async def register_voice(req: RegisterVoiceRequest):
    """Upload a reference audio once → encodes & caches VQ tokens → returns voice_id for all future /v1/tts calls."""
    if not tts_engine:
        raise HTTPException(status_code=500, detail="TTS Engine not initialized.")
    if req.voice_id in tts_engine.voice_cache:
        logger.info(f"✅ Voice '{req.voice_id}' already in cache — skipping re-encode.")
        return {"status": "already_cached", "voice_id": req.voice_id}
    try:
        audio_bytes = _base64.b64decode(req.audio)
        tts_engine.register_reference_voice(req.voice_id, audio_bytes, req.text or None)
        return {"status": "cached", "voice_id": req.voice_id}
    except Exception as e:
        logger.error(f"❌ Voice register error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/tts")
async def tts_endpoint(req: TTSRequestModel):
    if not tts_engine:
        raise HTTPException(status_code=500, detail="TTS Engine not initialized.")

    # Auto-register inline references if reference_id is not yet in cache
    effective_ref_id = req.reference_id
    if req.references and (not effective_ref_id or effective_ref_id not in tts_engine.voice_cache):
        ref = req.references[0]
        # Auto-generate a cache key from audio hash if no reference_id given
        if not effective_ref_id:
            audio_bytes_preview = _base64.b64decode(ref.audio[:512] if len(ref.audio) > 512 else ref.audio)
            effective_ref_id = "auto_" + _hashlib.sha256(audio_bytes_preview).hexdigest()[:12]
        if effective_ref_id not in tts_engine.voice_cache:
            try:
                audio_bytes = _base64.b64decode(ref.audio)
                tts_engine.register_reference_voice(effective_ref_id, audio_bytes, ref.text or None)
                logger.info(f"⚡ Auto-cached inline reference as '{effective_ref_id}'")
            except Exception as enc_err:
                logger.error(f"❌ Inline reference encode failed: {enc_err}")
                effective_ref_id = None

    if req.streaming:
        async def stream_gen():
            start = time.perf_counter()
            first = True
            count = 0
            bytes_sent = 0
            text_preview = req.text[:40].replace("\n", " ")
            logger.info(f"📥 [NATIVE TTS START] Text: '{text_preview}' | Voice: {req.reference_id} | MaxTokens: {req.max_new_tokens}")

            try:
                for chunk in tts_engine.generate_stream(
                    text=req.text,
                    reference_id=effective_ref_id,
                    max_new_tokens=req.max_new_tokens,
                    chunk_length=req.chunk_length,
                    top_p=req.top_p,
                    temperature=req.temperature,
                    repetition_penalty=req.repetition_penalty,
                    seed=req.seed,
                ):
                    if len(chunk) > 0:
                        bytes_sent += len(chunk)
                        count += 1
                        now = time.perf_counter()
                        if first:
                            print(f"🔥 NATIVE FIRST CHUNK (TTFA): {(now - start) * 1000:.0f} ms", flush=True)
                            first = False
                        yield chunk
            except Exception as stream_err:
                logger.error(f"❌ [NATIVE TTS ERROR] {stream_err}\n{traceback.format_exc()}")
                raise HTTPException(status_code=500, detail=str(stream_err))

            logger.info(f"🎉 [NATIVE TTS COMPLETE] Sent {bytes_sent:,} bytes in {count} chunks | Elapsed: {time.perf_counter() - start:.2f}s")

        return StreamingResponse(stream_gen(), media_type="audio/wav")
    else:
        try:
            audio_bytes = tts_engine.generate(
                text=req.text,
                reference_id=effective_ref_id,
                max_new_tokens=req.max_new_tokens,
                chunk_length=req.chunk_length,
                format=req.format,
                top_p=req.top_p,
                temperature=req.temperature,
                repetition_penalty=req.repetition_penalty,
                seed=req.seed,
            )
            return Response(content=audio_bytes, media_type=f"audio/{req.format}")
        except Exception as e:
            logger.error(f"❌ [NATIVE TTS NON-STREAM ERROR] {e}\n{traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/v1/tts/live")
async def websocket_tts(ws: WebSocket):
    await ws.accept()
    if not tts_engine:
        await ws.close(code=1011, reason="Engine not initialized")
        return

    logger.info("🔌 WebSocket connected to /v1/tts/live")
    try:
        while True:
            data = await ws.receive_json()
            req = TTSRequestModel(**data)
            start = time.perf_counter()
            first = True

            for chunk in tts_engine.generate_stream(
                text=req.text,
                reference_id=req.reference_id,
                max_new_tokens=req.max_new_tokens,
                chunk_length=req.chunk_length,
                top_p=req.top_p,
                temperature=req.temperature,
                repetition_penalty=req.repetition_penalty,
                seed=req.seed,
            ):
                if len(chunk) > 0:
                    now = time.perf_counter()
                    if first:
                        print(f"🔥 NATIVE WS FIRST CHUNK (TTFA): {(now - start) * 1000:.0f} ms", flush=True)
                        first = False
                    await ws.send_bytes(chunk)

            await ws.send_json({"event": "done", "total_ms": (time.perf_counter() - start) * 1000})

    except WebSocketDisconnect:
        logger.info("🔌 WebSocket client disconnected.")
    except Exception as e:
        logger.error(f"❌ WebSocket error: {e}")


# -----------------------------------------------------------------------------
# CLI Entry Point
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="FastFishTTS Native UltraFast Server")
    parser.add_argument("--listen", type=str, default="127.0.0.1:8085", help="Host and port (default: 127.0.0.1:8085)")
    parser.add_argument("--llama-checkpoint", type=str, default="checkpoints/s2-pro")
    parser.add_argument("--decoder-checkpoint", type=str, default="checkpoints/s2-pro/codec.pth")
    parser.add_argument("--decoder-config", type=str, default="modded_dac_vq")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--half", action="store_true", default=True)
    parser.add_argument("--compile", action="store_true", default=False)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--references-dir", type=str, default="references")
    args = parser.parse_args()

    global tts_engine
    tts_engine = FastFishTTS(
        llama_checkpoint=args.llama_checkpoint,
        decoder_checkpoint=args.decoder_checkpoint,
        decoder_config=args.decoder_config,
        device=args.device,
        half=args.half,
        compile_model=args.compile,
        num_workers=args.num_workers,
    )

    tts_engine.preload_references_dir(args.references_dir)

    host, port = args.listen.split(":")
    logger.info(f"🚀 Native FastFishTTS Server listening on http://{host}:{port}")
    
    # Run uvicorn server cleanly
    uvicorn.run(app, host=host, port=int(port), log_level="info", access_log=False)


if __name__ == "__main__":
    main()
