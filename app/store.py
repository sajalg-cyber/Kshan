import json
import os
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

load_dotenv()

COLLECTION = "video_chunks"
VECTOR_SIZE = 1024
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    check_compatibility=False,
)


def with_retry(action, attempts=3):
    for attempt in range(attempts):
        try:
            return action()
        except Exception:
            if attempt == attempts - 1:
                raise
            time.sleep(2)


def ensure_collection():
    exists = with_retry(lambda: client.collection_exists(COLLECTION))
    if not exists:
        with_retry(
            lambda: client.create_collection(
                collection_name=COLLECTION,
                vectors_config=models.VectorParams(
                    size=VECTOR_SIZE, distance=models.Distance.COSINE
                ),
            )
        )
    try:
        client.create_payload_index(
            collection_name=COLLECTION,
            field_name="video_id",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
    except Exception:
        pass


def video_path(video_id):
    return DATA_DIR / f"{video_id}.json"


def load_video(video_id):
    path = video_path(video_id)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_video(video_id, info, lines, chunks, vectors):
    ensure_collection()

    points = []
    for i, chunk in enumerate(chunks):
        points.append(
            models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{video_id}-{i}")),
                vector=vectors[i].tolist(),
                payload={
                    "video_id": video_id,
                    "start": chunk["start"],
                    "end": chunk["end"],
                    "text": chunk["text"],
                },
            )
        )

    for begin in range(0, len(points), 64):
        batch = points[begin : begin + 64]
        with_retry(
            lambda b=batch: client.upsert(collection_name=COLLECTION, points=b)
        )

    record = {
        "video_id": video_id,
        "info": info,
        "lines": lines,
        "chunk_count": len(chunks),
    }
    with open(video_path(video_id), "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False)
    return record


def search_chunks(video_id, vector, top_k=4):
    result = with_retry(
        lambda: client.query_points(
            collection_name=COLLECTION,
            query=vector.tolist(),
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="video_id", match=models.MatchValue(value=video_id)
                    )
                ]
            ),
            limit=top_k,
            with_payload=True,
        )
    )
    return [point.payload for point in result.points]