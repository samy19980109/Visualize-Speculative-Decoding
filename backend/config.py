from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # Required
    cerebras_api_key: str = Field(..., description="Cerebras Cloud API key")
    cerebras_target_model: str = Field(
        ..., description="Target model on Cerebras (e.g. gpt-oss-120b)"
    )

    # Draft model
    draft_model: str = Field(
        default="mlx-community/Llama-3.2-3B-Instruct-4bit",
        description="MLX-LM draft model identifier",
    )

    # Speculation parameters
    speculation_k: int = Field(default=8, ge=1, le=16, description="Tokens to draft per round")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: int = Field(default=512, ge=1, le=4096, description="Max tokens to generate")

    # Server
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
