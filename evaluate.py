import json
import time

from app import rag, store
from app.chunking import format_time

PAUSE_SECONDS = 25


def check(case, found):
    expected = case["expected_seconds"]
    tolerance = case.get("tolerance", 15)
    if not expected:
        return len(found) == 0
    return all(any(abs(f - e) <= tolerance for f in found) for e in expected)


def main():
    with open("eval_set.json", "r", encoding="utf-8") as f:
        cases = json.load(f)

    passed = ran = 0
    for n, case in enumerate(cases, 1):
        record = store.load_video(case["video_id"])
        if not record:
            print(f"{n}. SKIPPED (video not prepared): {case['question']}")
            continue
        ran += 1
        try:
            result = rag.answer_question(case["video_id"], case["question"], record["lines"])
            found = [t["seconds"] for t in result["timestamps"]]
            ok = check(case, found)
            passed += ok
            got = ", ".join(format_time(s) for s in found) or "none"
            want = ", ".join(format_time(s) for s in case["expected_seconds"]) or "none"
            print(f"{n}. {'PASS' if ok else 'FAIL'} | {case['question']}")
            print(f"   expected: {want} | got: {got}")
        except Exception as e:
            print(f"{n}. ERROR | {case['question']} ({type(e).__name__})")
        if n < len(cases):
            time.sleep(PAUSE_SECONDS)

    print(f"\n{passed} of {ran} passed")


if __name__ == "__main__":
    main()
    