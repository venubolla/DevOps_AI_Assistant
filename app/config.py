from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
)


class Settings(BaseSettings):
    app_name: str = "DevOps AI Assistant"

    app_version: str = "0.1.0"

    environment: str = "development"

    database_url: str

    ollama_base_url: str = "http://localhost:11434"

    ollama_model: str = "qwen3:8b"

    # RAGAS uses a separately configurable evaluator model.
    ragas_model: str = "qwen2.5:7b-instruct"

    # Separate model for the custom LLM answer judge.
    judge_model: str = "qwen2.5:7b-instruct"

    # Calibrated with scripts/calibrate_relevance_threshold.py: the 20 labeled
    # in-scope questions score at or above -0.44, while out-of-scope probes score
    # around -11.0. -5.0 sits inside that gap.
    relevance_threshold: float = -5.0

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
