import re
from typing import List, Literal, Optional

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from app.llm import BACKUP_MODEL, PRIMARY_MODEL, RETRY_ON, make_model

DEVANAGARI = re.compile(r"[\u0900-\u097F]")


class QuestionPlan(BaseModel):
    language: Literal["english", "hinglish"] = Field(
        description="Language of the QUESTION: hinglish if Hindi written in Roman letters, otherwise english"
    )
    question_type: Literal["find", "overview", "collect"] = Field(
        description=(
            "find: locate specific moments or facts. "
            "overview: describe or summarize the video, or part of it. "
            "collect: gather every item of some kind that is mentioned anywhere in the video."
        )
    )
    keywords: List[str] = Field(
        description="Up to 12 short search words that could appear in the transcript. Empty list for overview questions."
    )
    time_start_seconds: Optional[int] = Field(
        description="Start of the part of the video the question is limited to, in seconds. null if the question covers the whole video."
    )
    time_end_seconds: Optional[int] = Field(
        description="End of the part of the video the question is limited to, in seconds. null if the question covers the whole video."
    )
    target: Optional[str] = Field(
        description="For collect questions: what to gather, in a few words. null otherwise."
    )


SYSTEM = (
    "You analyse a question about a video so that a search system can answer it. "
    "The video is {duration} seconds long. "
    "Here is a sample of how the transcript text looks (script and spelling):\n"
    "{sample}\n\n"
    "If the question limits the answer to part of the video (for example the first "
    "N minutes, the second half, between two times, or the last N minutes), convert "
    "that part into seconds. Otherwise use null for both time fields. "
    "Keywords must be written in the SAME script as the sample (for Hindi captions "
    "use Devanagari), plus common alternative spellings and English forms. "
    "Never use question words such as when, what, did, कब, क्या as keywords."
)

prompt = ChatPromptTemplate.from_messages(
    [("system", SYSTEM), ("human", "{question}")]
)


def _structured(model_name):
    return make_model(model_name).with_structured_output(
        QuestionPlan, method="json_schema", strict=True
    )


planner = prompt | _structured(PRIMARY_MODEL).with_retry(
    retry_if_exception_type=RETRY_ON,
    stop_after_attempt=3,
    wait_exponential_jitter=True,
).with_fallbacks([_structured(BACKUP_MODEL)])


def clean_plan(plan, question, duration):
    if plan is None:
        language = "english"
        question_type = "find"
        keywords = [w for w in question.split() if len(w) > 3]
        start = end = target = None
    else:
        language = plan.language
        question_type = plan.question_type
        keywords = list(plan.keywords)
        start = plan.time_start_seconds
        end = plan.time_end_seconds
        target = plan.target

    if DEVANAGARI.search(question):
        language = "hindi"

    keywords = [k.strip() for k in keywords if k and k.strip()][:12]
    if question_type == "overview":
        keywords = []

    time_range = None
    if start is not None or end is not None:
        s = 0 if start is None else start
        e = duration if end is None else end
        s = max(0, min(s, duration))
        e = max(0, min(e, duration))
        if s > e:
            s, e = e, s
        if e - s >= 1 and not (s == 0 and e >= duration):
            time_range = (s, e)

    if question_type == "collect" and not target:
        target = question

    return {
        "language": language,
        "question_type": question_type,
        "keywords": keywords,
        "time_range": time_range,
        "target": target,
        "planned": plan is not None,
    }


def make_plan(question, lines):
    duration = int(lines[-1]["start"] + lines[-1]["duration"])
    middle = len(lines) // 2
    sample = "\n".join(line["text"] for line in lines[middle : middle + 6])
    try:
        plan = planner.invoke(
            {"question": question, "duration": duration, "sample": sample}
        )
    except Exception:
        plan = None
    return clean_plan(plan, question, duration)