from functools import lru_cache

from langchain_huggingface import (
    HuggingFaceEmbeddings,
)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# Loading sentence-transformers costs seconds, so keep one instance per process.
@lru_cache(maxsize=1)
def get_embedding_model() -> HuggingFaceEmbeddings:

    return HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={
            "device": "cpu",
        },
        encode_kwargs={
            "normalize_embeddings": True,
        },
    )


def embed_texts(
    texts: list[str],
    batch_size: int = 32,
) -> list[list[float]]:

    model = get_embedding_model()

    embeddings = []

    for start in range(
        0,
        len(texts),
        batch_size,
    ):
        batch = texts[start : start + batch_size]

        batch_embeddings = model.embed_documents(batch)

        embeddings.extend(batch_embeddings)

    return embeddings
