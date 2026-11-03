import json
import os

import groq
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

load_dotenv()

PRIMARY_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
BACKUP_MODEL = os.getenv("LLM_BACKUP_MODEL", "openai/gpt-oss-20b")

RETRY_ON = (
    groq.RateLimitError,
    groq.APIConnectionError,
    groq.InternalServerError,
)


def make_model(name):
    return ChatGroq(
        model=name,
        temperature=0,
        api_key=os.environ.get("GROQ_API_KEY"),
        timeout=60,
        max_retries=1,
    )


primary = make_model(PRIMARY_MODEL).with_retry(
    retry_if_exception_type=RETRY_ON,
    stop_after_attempt=3,
    wait_exponential_jitter=True,
)
backup = make_model(BACKUP_MODEL)

chat_model = primary.with_fallbacks([backup])
text_chain = chat_model | StrOutputParser()


def ask_llm(system, user):
    return text_chain.invoke([("system", system), ("human", user)])


def parse_json(text):
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None