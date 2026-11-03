import re
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi

ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{11}$")

YOUTUBE_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "music.youtube.com",
    "youtube-nocookie.com",
    "www.youtube-nocookie.com",
}

PREFERRED_LANGUAGES = ["hi", "en"]


def extract_video_id(url_or_id):
    text = url_or_id.strip()

    if ID_PATTERN.match(text):
        return text

    if "://" not in text:
        text = "https://" + text

    parsed = urlparse(text)
    host = (parsed.hostname or "").lower()
    path_parts = [p for p in parsed.path.split("/") if p]

    candidate = None

    if host == "youtu.be":
        if path_parts:
            candidate = path_parts[0]
    elif host in YOUTUBE_HOSTS:
        if parsed.path == "/watch":
            candidate = parse_qs(parsed.query).get("v", [None])[0]
        elif len(path_parts) >= 2 and path_parts[0] in ("shorts", "embed", "live", "v"):
            candidate = path_parts[1]

    if candidate and ID_PATTERN.match(candidate):
        return candidate

    return None


def get_transcript(video_id):
    api = YouTubeTranscriptApi()
    available = api.list(video_id)

    try:
        transcript = available.find_transcript(PREFERRED_LANGUAGES)
    except Exception:
        tracks = list(available)
        if not tracks:
            raise ValueError("This video has no captions.")
        manual = [t for t in tracks if not t.is_generated]
        transcript = (manual or tracks)[0]

    fetched = transcript.fetch()
    lines = [
        {"text": s.text, "start": s.start, "duration": s.duration}
        for s in fetched
    ]
    info = {
        "language_code": transcript.language_code,
        "is_generated": transcript.is_generated,
    }
    return lines, info