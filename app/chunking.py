import re


def clean_lines(lines):
    cleaned = []
    for line in lines:
        text = re.sub(r"\[.*?\]", "", line["text"])
        text = text.replace("\n", " ").strip()
        if text:
            cleaned.append(
                {"text": text, "start": line["start"], "duration": line["duration"]}
            )
    return cleaned


def make_chunks(lines, window=40, overlap=10):
    chunks = []
    n = len(lines)
    i = 0

    while i < n:
        start_time = lines[i]["start"]
        end_limit = start_time + window

        j = i
        parts = []
        while j < n and lines[j]["start"] < end_limit:
            parts.append(lines[j]["text"])
            j += 1

        last = lines[j - 1]
        chunks.append(
            {
                "start": start_time,
                "end": last["start"] + last["duration"],
                "text": " ".join(parts),
            }
        )

        if j >= n:
            break

        next_start_time = end_limit - overlap
        k = i + 1
        while k < n and lines[k]["start"] < next_start_time:
            k += 1
        i = k

    return chunks


def format_time(seconds):
    seconds = int(seconds)
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"
