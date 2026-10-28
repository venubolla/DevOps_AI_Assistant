from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_model = None


def get_similarity_model() -> SentenceTransformer:
    global _model

    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)

    return _model


def calculate_semantic_similarity(
    text_a: str,
    text_b: str,
) -> float:

    model = get_similarity_model()

    embeddings = model.encode(
        [text_a, text_b],
        normalize_embeddings=True,
    )

    similarity = embeddings[0] @ embeddings[1]

    return float(similarity)
