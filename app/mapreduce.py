import json
import os
import time
from pathlib import Path
from typing import List

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from app.chunking import format_time
from app.llm import BACKUP_MODEL, PRIMARY_MODEL, RETRY_ON, chat_model, make_model

TPM = int(os.getenv("GROQ_TPM", "6000"))
SEGMENT_CHARS = 6000
DIRECT_CHARS = 9000
MAX_NOTES_CHARS = 12000
GROUP = 6
MAX_ITEMS = 150
DATA_DIR = Path("data")

_recent_calls = []


def estimate_tokens(text):
    if not text:
        return 0
    devanagari = sum(1 for ch in text if "\u0900" <= ch <= "\u097F")
    chars_per_token = 2.0 if devanagari / len(text) > 0.3 else 3.5
    return int(len(text) / chars_per_token) + 700


def pace(tokens):
    while True:
        now = time.time()
        _recent_calls[:] = [(t, k) for t, k in _recent_calls if now - t < 60]
        used = sum(k for _, k in _recent_calls)
        if not _recent_calls or used + tokens <= TPM:
            break
        wait = 60 - (now - _recent_calls[0][0]) + 0.5
        print(f"  (waiting {wait:.0f}s to stay within the free rate limit)")
        time.sleep(max(wait, 1))
    _recent_calls.append((time.time(), tokens))


def structured(schema):
    strict = (
        make_model(PRIMARY_MODEL)
        .with_structured_output(schema, method="json_schema", strict=True)
        .with_retry(
            retry_if_exception_type=RETRY_ON,
            stop_after_attempt=3,
            wait_exponential_jitter=True,
        )
    )
    loose = make_model(PRIMARY_MODEL).with_structured_output(schema, method="json_schema")
    backup = make_model(BACKUP_MODEL).with_structured_output(schema, method="json_schema")
    return strict.with_fallbacks([loose, backup])


class Match(BaseModel):
    name: str = Field(description="Short name of the item, spelled as in the transcript if unsure")
    line: int = Field(description="The number in square brackets of the line where it is mentioned")
    detail: str = Field(description="One short sentence of context, in English")


class SegmentMatches(BaseModel):
    matches: List[Match] = Field(description="Every matching item in this part. Empty list if none.")


class Group(BaseModel):
    title: str = Field(description="Name of this distinct thing, in the reply language")
    item_numbers: List[int] = Field(description="Numbers of all raw items that refer to this same thing")


class FinalList(BaseModel):
    answer: str = Field(description="The final answer, in the reply language, without timestamps")
    groups: List[Group] = Field(description="One group per distinct thing. Empty if nothing was found.")


notes_chain = (
    ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You take notes on one part of a video transcript, covering {span} of the video. "
                "The transcript may be in any language, but write your notes in English. "
                "Write 3 to 5 short bullet points (under 90 words in total) with the main points, "
                "plus any names, numbers and specific things that are mentioned. "
                "Auto-captions may contain mistakes, so use judgment. Use ONLY the transcript. "
                "Output only the bullet points.",
            ),
            ("human", "{text}"),
        ]
    )
    | chat_model
    | StrOutputParser()
)

condense_chain = (
    ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Merge these notes from consecutive parts of a video ({span}) into 4 to 8 short "
                "bullet points in English. Keep names, numbers and specific things. "
                "Output only the bullet points.",
            ),
            ("human", "{text}"),
        ]
    )
    | chat_model
    | StrOutputParser()
)

final_chain = (
    ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You write the final answer to a request about a video, using ONLY the material below. "
                "{scope_text} The material is either notes from consecutive parts of the video "
                "(each labelled with its time span) or transcript excerpts, and it may be in a "
                "different language from the reply. "
                "LANGUAGE RULE (very important): write the answer ONLY in this language: {rule} "
                "Follow the format the user asks for (for example the number of points, the "
                "length, or a table). If no format is asked, write 3 to 6 sentences. "
                "Write only the answer, with no preface. Do not invent anything that is not "
                "in the material.",
            ),
            ("human", "Request: {question}\n\nMaterial:\n{material}"),
        ]
    )
    | chat_model
    | StrOutputParser()
)

extract_chain = (
    ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You scan one part of a video transcript ({span}) for everything the user wants "
                "to collect: {target}. Use ONLY the numbered lines. List each matching item once "
                "per place it is mentioned, with a short name (keep the transcript's spelling if "
                "unsure, because auto-captions can garble names), the line number, and one short "
                "sentence of context in English. If nothing matches, return an empty list. "
                "Do not guess items that are not in the lines.",
            ),
            ("human", "{text}"),
        ]
    )
    | structured(SegmentMatches)
)

merge_chain = (
    ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You merge raw findings from a video into a final answer. The user wants: "
                "{target}. Each raw item has a time, a name, a note and the original transcript "
                "line. The SAME thing is often mentioned several times under different names or "
                "wordings: it may be made or described in one place, and then given a different "
                "name, or a number (first, second), later in the video. Use the original lines, "
                "the notes and the order of events to decide which items are the same thing, "
                "and merge them into ONE group. "
                "LANGUAGE RULE (very important): write the answer and every group title ONLY in "
                "this language: {rule} Follow the format the user asks for in the request. "
                "Never write item numbers, line numbers or timestamps in the answer or titles. "
                "'groups' has one entry per distinct thing, listing the numbers of the raw items "
                "it merges. Only use numbers that appear in the list. If the list is empty, say "
                "in the answer that nothing was found, and return no groups.",
            ),
            ("human", "Request: {question}\n\nRaw items:\n{items}"),
        ]
    )
    | structured(FinalList)
)


def make_segments(lines, ids, max_chars=SEGMENT_CHARS):
    segments, current, size = [], [], 0
    for i in ids:
        n = len(lines[i]["text"]) + 1
        if current and size + n > max_chars:
            segments.append(current)
            current, size = [], 0
        current.append(i)
        size += n
    if current:
        segments.append(current)
    return segments


def span_of(lines, ids):
    start = lines[ids[0]]["start"]
    end = lines[ids[-1]]["start"] + lines[ids[-1]]["duration"]
    return start, end


def span_label(lines, ids):
    start, end = span_of(lines, ids)
    return f"{format_time(start)} to {format_time(end)}"


def notes_path(video_id):
    return DATA_DIR / f"{video_id}.notes.json"


def load_notes(video_id):
    path = notes_path(video_id)
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("segment_chars") == SEGMENT_CHARS:
                return data.get("notes", {})
        except Exception:
            pass
    return {}


def save_notes(video_id, notes):
    with open(notes_path(video_id), "w", encoding="utf-8") as f:
        json.dump({"segment_chars": SEGMENT_CHARS, "notes": notes}, f, ensure_ascii=False)


def combine_notes(items):
    while sum(len(x["text"]) for x in items) > MAX_NOTES_CHARS and len(items) > 1:
        merged = []
        for k in range(0, len(items), GROUP):
            group = items[k : k + GROUP]
            if len(group) == 1:
                merged.append(group[0])
                continue
            joined = "\n".join(
                f"[{format_time(x['start'])} to {format_time(x['end'])}] {x['text']}"
                for x in group
            )
            span = f"{format_time(group[0]['start'])} to {format_time(group[-1]['end'])}"
            pace(estimate_tokens(joined))
            text = condense_chain.invoke({"span": span, "text": joined})
            merged.append(
                {"start": group[0]["start"], "end": group[-1]["end"], "text": text.strip()}
            )
        items = merged
    return items


def make_stamp(video_id, lines, line_id, why, certainty="direct"):
    line = lines[line_id]
    seconds = int(line["start"])
    start_at = max(seconds - 3, 0)
    return {
        "time": format_time(line["start"]),
        "seconds": seconds,
        "start_at": start_at,
        "certainty": certainty,
        "text": line["text"],
        "why": why,
        "link": f"https://www.youtube.com/watch?v={video_id}&t={start_at}s",
    }


def overview(video_id, question, lines, plan, scope, result, rule_text):
    if not scope:
        result["answer"] = "There are no transcript lines in that part of the video."
        return result
    result["lines_used"] = len(scope)

    scope_text = "The material covers the whole video."
    if plan["time_range"]:
        start, end = plan["time_range"]
        scope_text = (
            f"The material covers only the part of the video from "
            f"{format_time(start)} to {format_time(end)}."
        )

    chars = sum(len(lines[i]["text"]) + 1 for i in scope)
    if chars <= DIRECT_CHARS:
        material = "\n".join(
            f"({format_time(lines[i]['start'])}) {lines[i]['text']}" for i in scope
        )
    else:
        segments = make_segments(lines, range(len(lines)))
        scope_set = set(scope)
        chosen = [
            k
            for k, ids in enumerate(segments)
            if sum(1 for i in ids if i in scope_set) * 2 >= len(ids)
        ]
        if not chosen:
            chosen = [
                max(
                    range(len(segments)),
                    key=lambda k: sum(1 for i in segments[k] if i in scope_set),
                )
            ]

        notes = load_notes(video_id)
        todo = [k for k in chosen if str(k) not in notes]
        for n, k in enumerate(todo, 1):
            ids = segments[k]
            text = "\n".join(lines[i]["text"] for i in ids)
            pace(estimate_tokens(text))
            notes[str(k)] = notes_chain.invoke(
                {"span": span_label(lines, ids), "text": text}
            ).strip()
            save_notes(video_id, notes)
            print(f"  notes {n}/{len(todo)}")

        items = []
        for k in chosen:
            start, end = span_of(lines, segments[k])
            items.append({"start": start, "end": end, "text": notes[str(k)]})
        items = combine_notes(items)
        material = "\n".join(
            f"[{format_time(x['start'])} to {format_time(x['end'])}] {x['text']}"
            for x in items
        )
        result["note"] = f"Built from notes on {len(chosen)} parts of the video."

    pace(estimate_tokens(material))
    result["answer"] = final_chain.invoke(
        {
            "scope_text": scope_text,
            "rule": rule_text,
            "question": question,
            "material": material,
        }
    ).strip()
    return result


def collect(video_id, question, lines, plan, scope, result, rule_text):
    if not scope:
        result["answer"] = "There are no transcript lines in that part of the video."
        return result
    result["lines_used"] = len(scope)

    target = plan["target"] or question
    segments = make_segments(lines, scope)

    raw, seen = [], set()
    for n, ids in enumerate(segments, 1):
        valid = set(ids)
        text = "\n".join(f"[{i}] {lines[i]['text']}" for i in ids)
        pace(estimate_tokens(text))
        found = extract_chain.invoke(
            {"span": span_label(lines, ids), "target": target, "text": text}
        )
        if found is not None:
            for match in found.matches:
                key = (match.name.strip().lower(), match.line)
                if match.line in valid and key not in seen:
                    seen.add(key)
                    raw.append(match)
        print(f"  scanned {n}/{len(segments)}")

    raw.sort(key=lambda m: m.line)
    truncated = len(raw) > MAX_ITEMS
    raw = raw[:MAX_ITEMS]
    listing = (
        "\n".join(
            f"#{k} [{format_time(lines[m.line]['start'])}] {m.name} - {m.detail}"
            f" | original line: {lines[m.line]['text']}"
            for k, m in enumerate(raw, 1)
        )
        or "(none)"
    )

    pace(estimate_tokens(listing))
    final = merge_chain.invoke(
        {"target": target, "rule": rule_text, "question": question, "items": listing}
    )
    if final is None:
        result["answer"] = "The AI reply could not be read. Please try again."
        return result
    result["answer"] = final.answer.strip()

    used, picked = set(), []
    for group in final.groups:
        for num in group.item_numbers:
            if 1 <= num <= len(raw) and num not in used:
                used.add(num)
                match = raw[num - 1]
                picked.append((match.line, group.title, match.detail))
    picked.sort()

    for line_id, title, detail in picked[:40]:
        result["timestamps"].append(
            make_stamp(video_id, lines, line_id, f"{title}: {detail}")
        )

    result["note"] = f"Scanned {len(segments)} section(s) of the selected part of the video."
    if truncated:
        result["note"] += f" Only the first {MAX_ITEMS} findings were kept."
    return result