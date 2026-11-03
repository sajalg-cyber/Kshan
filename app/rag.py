
from app import store
from app.chunking import format_time
from app.embeddings import embed_query
from app.llm import ask_llm, parse_json
from app.planner import make_plan
from app import mapreduce
MAX_LINES = 100
OVERVIEW_LINES = 100
OVERVIEW_BLOCK = 5
 
LANGUAGE_RULES = {
    "english": "English.",
    "hinglish": (
        "Hinglish, which means Hindi written in Roman (English) letters, the "
        "casual way people text in India. Example style: 'Haan, dry mixture "
        "banate waqt lal mirch powder daala gaya.' Do NOT use Devanagari letters."
    ),
    "hindi": "Hindi written in Devanagari script (देवनागरी).",
}
 
 
def range_label(time_range):
    if not time_range:
        return None
    start, end = time_range
    return {"start": format_time(start), "end": format_time(end)}
 
 
def scope_ids(lines, time_range):
    if not time_range:
        return list(range(len(lines)))
    start, end = time_range
    return [i for i, line in enumerate(lines) if start <= line["start"] < end]
 
 
def filter_keywords(lines, scope, keywords):
    limit = max(10, int(len(scope) * 0.10))
    kept = []
    for keyword in keywords:
        k = keyword.lower()
        count = sum(1 for i in scope if k in lines[i]["text"].lower())
        if 0 < count <= limit:
            kept.append(keyword)
    return kept
 
 
def collect_line_ids(video_id, question, lines, scope, keywords, time_range):
    scope_set = set(scope)
 
    keyword_ids = set()
    for i in scope:
        text = lines[i]["text"].lower()
        if any(k.lower() in text for k in keywords):
            for j in (i - 1, i, i + 1):
                if j in scope_set:
                    keyword_ids.add(j)
 
    warning = None
    top_k = 12 if time_range else 4
    try:
        chunks = store.search_chunks(video_id, embed_query(question), top_k=top_k)
    except Exception:
        chunks = []
        warning = "Semantic search was unavailable, so only keyword search was used."
 
    semantic_ids = set()
    for payload in chunks:
        for j in scope:
            if payload["start"] <= lines[j]["start"] <= payload["end"]:
                semantic_ids.add(j)
 
    ordered = sorted(keyword_ids) + sorted(semantic_ids - keyword_ids)
    return sorted(ordered[:MAX_LINES]), warning
 
 
def overview_ids(scope):
    n = len(scope)
    if n <= OVERVIEW_LINES:
        return list(scope)
    blocks = OVERVIEW_LINES // OVERVIEW_BLOCK
    step = n / blocks
    picked = []
    for b in range(blocks):
        begin = int(b * step)
        picked.extend(scope[begin : begin + OVERVIEW_BLOCK])
    return sorted(set(picked))
 
 
def base_result(video_id, question, plan):
    warning = None
    if not plan["planned"]:
        warning = "The question planner was unavailable, so a basic search was used."
    return {
        "video_id": video_id,
        "question": question,
        "language": plan["language"],
        "mode": plan["question_type"],
        "time_range": range_label(plan["time_range"]),
        "keywords": [],
        "lines_used": 0,
        "answer": "",
        "timestamps": [],
        "note": None,
        "warning": warning,
    }
 
 
def answer_overview(video_id, question, lines, plan, scope):
    result = base_result(video_id, question, plan)
    if not scope:
        result["answer"] = "There are no transcript lines in that part of the video."
        return result
 
    ids = overview_ids(scope)
    result["lines_used"] = len(ids)
    if len(ids) < len(scope):
        result["note"] = (
            "This answer is based on an evenly spaced sample of the transcript, "
            "so it may miss details."
        )
 
    context = "\n".join(
        f"({format_time(lines[i]['start'])}) {lines[i]['text']}" for i in ids
    )
 
    scope_text = "The excerpts cover the whole video."
    if plan["time_range"]:
        label = range_label(plan["time_range"])
        scope_text = (
            f"The excerpts cover only the part of the video from {label['start']} "
            f"to {label['end']}."
        )
 
    system = (
        "You describe a video using ONLY the transcript excerpts provided. "
        f"{scope_text} The excerpts are sampled evenly, so some parts are "
        "skipped. Auto-captions may contain mistakes, so use judgment. "
        "LANGUAGE RULE (very important): write the answer ONLY in this language: "
        f"{LANGUAGE_RULES[plan['language']]} "
        "The transcript may be in a different language. Understand it, but do "
        "not copy its language into your reply. "
        "Follow the format the user asks for (for example the number of points, "
        "the length, or a table). If no format is asked, write 2 to 4 sentences. "
        "Write only the answer text, with no JSON and no preface. "
        "Do not write timestamps. Do not invent details that are not in the excerpts."
    )
    user = (
        f"Request: {question}\n"
        f"(Reply language: {plan['language']}. Follow the language rule.)\n\n"
        f"Transcript excerpts:\n{context}"
    )
 
    result["answer"] = ask_llm(system, user).strip()
    return result
 
 
def answer_specific(video_id, question, lines, plan, scope):
    result = base_result(video_id, question, plan)
    if not scope:
        result["answer"] = "There are no transcript lines in that part of the video."
        return result
 
    keywords = filter_keywords(lines, scope, plan["keywords"])
    ids, warning = collect_line_ids(
        video_id, question, lines, scope, keywords, plan["time_range"]
    )
 
    result["keywords"] = keywords
    result["lines_used"] = len(ids)
    if warning:
        result["warning"] = warning
 
    if not ids:
        result["answer"] = "No matching lines were found in this video."
        return result
 
    extra = ""
    if plan["question_type"] == "collect":
        extra = (
            f"The user wants every distinct item of this kind: {plan['target']}. "
            "List each item once, even if it is mentioned several times, and "
            "cite the line where it is mentioned. "
        )
        result["note"] = (
            "This list is built from search results, not a scan of the whole "
            "video, so it may be incomplete."
        )
 
    context = "\n".join(
        f"[{i}] ({format_time(lines[i]['start'])}) {lines[i]['text']}" for i in ids
    )
 
    system = (
        "You answer questions about a video using ONLY the numbered transcript "
        "lines provided. Auto-captions may contain mistakes, so use judgment. "
        "LANGUAGE RULE (very important): write 'answer' and every 'why' ONLY in "
        f"this language: {LANGUAGE_RULES[plan['language']]} "
        "The transcript lines may be in a different language. Understand them, "
        "but do not copy their language into your reply. Describe the meaning in "
        "your own words instead of quoting them. "
        'Return ONLY JSON like {"found": true, "answer": "...", "evidence": [{"line": 12, "role": "direct", "why": "..."}]}. '
        "'found' is true only if the lines really show the answer to the "
        "question. If the thing asked about is not mentioned, set 'found' to "
        "false, say in 'answer' that it is not mentioned, and return an empty "
        "evidence list. A line about a different thing that only shares a word "
        "with the question is NOT a match. "
        "In 'answer', give a short, direct answer and do not write timestamps "
        "(they are added automatically). "
        f"{extra}"
        "Each evidence item needs a 'role': "
        "'direct' if that line itself says or shows the answer; "
        "'uncertain' if it might be relevant but is unclear or may be a different thing; "
        "'context' if it only introduces the topic or surrounds the answer. "
        "Only 'direct' and 'uncertain' lines are shown to the user. "
        "If two lines refer to the same thing, mark only the line where it "
        "actually happens as 'direct'. "
        "Never mention line numbers inside 'answer' or 'why'. "
        "Never invent line numbers."
    )
    user = (
        f"Question: {question}\n"
        f"(Reply language: {plan['language']}. Follow the language rule.)\n\n"
        f"Transcript lines:\n{context}"
    )
 
    data = parse_json(ask_llm(system, user))
    if not data:
        result["answer"] = "The AI reply could not be read. Please try again."
        return result
 
    result["answer"] = str(data.get("answer", ""))
 
    found = data.get("found", True)
    if isinstance(found, str):
        found = found.strip().lower() not in ("false", "no", "0")
    if not found:
        return result
 
    valid = set(ids)
    seen = set()
    evidence = []
    for item in data.get("evidence", []):
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "direct")).lower()
        if role not in ("direct", "uncertain"):
            continue
        try:
            line_id = int(item.get("line"))
        except (TypeError, ValueError):
            continue
        if line_id in valid and line_id not in seen:
            seen.add(line_id)
            evidence.append((line_id, role, str(item.get("why", ""))))
    evidence.sort()
 
    for line_id, role, why in evidence:
        line = lines[line_id]
        seconds = int(line["start"])
        start_at = max(seconds - 3, 0)
        result["timestamps"].append(
            {
                "time": format_time(line["start"]),
                "seconds": seconds,
                "start_at": start_at,
                "certainty": role,
                "text": line["text"],
                "why": why,
                "link": f"https://www.youtube.com/watch?v={video_id}&t={start_at}s",
            }
        )
 
    return result
 
 
def answer_question(video_id, question, lines):
    plan = make_plan(question, lines)
    scope = scope_ids(lines, plan["time_range"])
    kind = plan["question_type"]
    rule = LANGUAGE_RULES[plan["language"]]
 
    if kind == "overview":
        result = base_result(video_id, question, plan)
        return mapreduce.overview(video_id, question, lines, plan, scope, result, rule)
    if kind == "collect":
        result = base_result(video_id, question, plan)
        return mapreduce.collect(video_id, question, lines, plan, scope, result, rule)
    return answer_specific(video_id, question, lines, plan, scope)

