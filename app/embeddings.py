from sentence_transformers import SentenceTransformer

_model = None


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("BAAI/bge-m3")
    return _model


def embed_texts(texts):
    return get_model().encode(texts, batch_size=8, normalize_embeddings=True)


def embed_query(text):
    return get_model().encode(text, normalize_embeddings=True)
