import os
import subprocess
import tempfile
import time
from pathlib import Path

import groq
import imageio_ffmpeg
from dotenv import load_dotenv
from yt_dlp import YoutubeDL

load_dotenv()

MODEL = os.getenv("WHISPER_MODEL", "whisper-large-v3")
MAX_MINUTES = int(os.getenv("WHISPER_MAX_MINUTES", "90"))
CHUNK_SECONDS = 600

_client = groq.Groq(api_key=os.environ.get("GROQ_API_KEY"))
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


class WhisperError(Exception):
    pass


def _get(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def download_audio(video_id, folder):
    url = f"https://www.youtube.com/watch?v={video_id}"
    options = {
        "format": "bestaudio/best",
        "outtmpl": str(Path(folder) / "%(id)s.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }
    with YoutubeDL(options) as ydl:
        info = ydl.extract_info(url, download=False)
        if info.get("is_live"):
            raise WhisperError("Live streams are not supported.")
        duration = info.get("duration") or 0
        if duration > MAX_MINUTES * 60:
            raise WhisperError(
                f"This video is {duration // 60} minutes long, and the limit is {MAX_MINUTES} minutes."
            )
        ydl.download([url])

    files = [
        p
        for p in Path(folder).glob(f"{video_id}.*")
        if p.suffix not in (".part", ".ytdl")
    ]
    if not files:
        raise WhisperError("The audio download failed.")
    return files[0]


def split_audio(source, folder):
    pattern = str(Path(folder) / "chunk_%03d.flac")
    command = [
        FFMPEG, "-y", "-i", str(source),
        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "flac",
        "-f", "segment", "-segment_time", str(CHUNK_SECONDS),
        "-reset_timestamps", "1",
        pattern,
    ]
    done = subprocess.run(
        command, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    chunks = sorted(Path(folder).glob("chunk_*.flac"))
    if done.returncode != 0 or not chunks:
        raise WhisperError("The audio could not be converted.")
    return chunks


def transcribe_chunk(path):
    for attempt in range(4):
        try:
            with open(path, "rb") as f:
                return _client.audio.transcriptions.create(
                    file=(path.name, f.read()),
                    model=MODEL,
                    response_format="verbose_json",
                    temperature=0.0,
                )
        except groq.RateLimitError:
            if attempt == 3:
                raise
            print("  (Whisper rate limit, waiting 30 seconds)")
            time.sleep(30)


def to_lines(result, offset):
    lines = []
    for seg in _get(result, "segments", None) or []:
        text = (_get(seg, "text", "") or "").strip()
        if not text:
            continue
        if (_get(seg, "no_speech_prob", 0) or 0) > 0.8:
            continue
        start = float(_get(seg, "start", 0) or 0)
        end = float(_get(seg, "end", start) or start)
        lines.append(
            {"text": text, "start": offset + start, "duration": max(end - start, 0.1)}
        )
    return lines


def transcribe_video(video_id):
    with tempfile.TemporaryDirectory() as folder:
        source = download_audio(video_id, folder)
        chunks = split_audio(source, folder)

        lines, language = [], None
        for n, chunk in enumerate(chunks):
            print(f"  whisper {n + 1}/{len(chunks)}")
            result = transcribe_chunk(chunk)
            language = language or _get(result, "language", None)
            lines.extend(to_lines(result, n * CHUNK_SECONDS))

    if not lines:
        raise WhisperError("No speech could be recognised in this video.")
    info = {"language_code": f"{language or 'unknown'} (Whisper)", "is_generated": True}
    return lines, info