import os

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app import rag, store, whisper_fallback
from app.chunking import clean_lines, format_time, make_chunks
from app.embeddings import embed_texts
from app.youtube import extract_video_id, get_transcript

app = FastAPI(title="Momentwise")


class IngestRequest(BaseModel):
    url: str


class AskRequest(BaseModel):
    url: str
    question: str


def summarize(record, status):
    lines = record["lines"]
    last = lines[-1]
    return {
        "video_id": record["video_id"],
        "status": status,
        "language": record["info"]["language_code"],
        "auto_generated": record["info"]["is_generated"],
        "lines": len(lines),
        "chunks": record["chunk_count"],
        "duration": format_time(last["start"] + last["duration"]),
    }


def get_raw_lines(video_id):
    forced = os.getenv("FORCE_WHISPER") == "1"
    if not forced:
        try:
            return get_transcript(video_id)
        except Exception as e:
            name = type(e).__name__
            if "Blocked" in name or "TooManyRequests" in name:
                raise HTTPException(
                    status_code=503,
                    detail="YouTube is temporarily blocking transcript requests. Try again later.",
                )
            no_captions = name in ("TranscriptsDisabled", "NoTranscriptFound") or isinstance(e, ValueError)
            if not no_captions:
                raise HTTPException(
                    status_code=404,
                    detail=f"Could not get captions for this video ({name}). "
                    "It may be private, age-restricted or unavailable.",
                )

    print("Using Whisper (no captions found, or FORCE_WHISPER is on)...")
    try:
        return whisper_fallback.transcribe_video(video_id)
    except whisper_fallback.WhisperError as w:
        raise HTTPException(
            status_code=422,
            detail=f"This video has no captions, and audio transcription is not possible: {w}",
        )
    except Exception as w:
        raise HTTPException(
            status_code=502,
            detail=f"This video has no captions, and the audio fallback failed ({type(w).__name__}). Please try again later.",
        )


@app.get("/")
def home():
    return {"message": "Momentwise API is running. Open /ui/ or /docs."}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest")
def ingest(request: IngestRequest):
    video_id = extract_video_id(request.url)
    if not video_id:
        raise HTTPException(
            status_code=422,
            detail="That does not look like a valid YouTube link.",
        )

    existing = store.load_video(video_id)
    if existing:
        return summarize(existing, "already_ingested")

    raw_lines, info = get_raw_lines(video_id)

    lines = clean_lines(raw_lines)
    if not lines:
        raise HTTPException(status_code=422, detail="The captions were empty.")

    chunks = make_chunks(lines)

    try:
        vectors = embed_texts([chunk["text"] for chunk in chunks])
        record = store.save_video(video_id, info, lines, chunks, vectors)
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Could not store the video ({type(e).__name__}). Please try again.",
        )

    return summarize(record, "ingested")


@app.post("/ask")
def ask(request: AskRequest):
    video_id = extract_video_id(request.url)
    if not video_id:
        raise HTTPException(
            status_code=422,
            detail="That does not look like a valid YouTube link.",
        )

    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Please type a question.")

    record = store.load_video(video_id)
    if not record:
        raise HTTPException(
            status_code=404,
            detail="This video has not been prepared yet. Call /ingest with its link first.",
        )

    try:
        return rag.answer_question(video_id, question, record["lines"])
    except Exception as e:
        name = type(e).__name__
        status = getattr(e, "status_code", None)
        if status == 429 or "RateLimit" in name:
            raise HTTPException(
                status_code=429,
                detail="The AI service is busy (rate limit). Wait a minute and try again.",
            )
        if status == 413:
            raise HTTPException(
                status_code=413,
                detail="That request was too large for the free AI limit. Try a more specific question.",
            )
        raise HTTPException(
            status_code=502,
            detail=f"Could not answer right now ({name}). Please try again.",
        )


app.mount("/ui", StaticFiles(directory="frontend", html=True), name="ui")
