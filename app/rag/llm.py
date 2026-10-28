from langchain_ollama import ChatOllama

from app.config import settings


def get_llm():
    """
    Main LLM used by the Agent.

    Keep the normal Qwen3 configuration so that
    tool calling and reasoning are not artificially
    restricted.
    """

    return ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=0,
    )


def get_fast_llm():
    """
    Lightweight LLM used for query rewriting.
    """

    return ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=0,
        num_predict=64,
        reasoning=False,
    )


def get_judge_llm():
    """Evaluator model kept separate from the answer-generation model."""

    return ChatOllama(
        model=settings.judge_model,
        base_url=settings.ollama_base_url,
        temperature=0,
    )
