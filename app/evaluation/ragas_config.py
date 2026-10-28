from openai import OpenAI
from ragas.llms import llm_factory

from app.config import settings


def get_ragas_llm():
    """Create the configured RAGAS judge through Ollama's OpenAI API."""

    base_url = settings.ollama_base_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url += "/v1"
    client = OpenAI(
        base_url=base_url,
        api_key="ollama",
    )

    return llm_factory(
        settings.ragas_model,
        provider="openai",
        client=client,
        adapter="auto",
        temperature=0,
        max_tokens=1536,
    )
