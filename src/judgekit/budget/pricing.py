from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPricing:
    input_per_1k: float
    output_per_1k: float


# Prices in USD per 1k tokens. Free-tier vendors set to 0.0.
PRICING: dict[str, ModelPricing] = {
    # Groq free tier
    "groq/llama-3.3-70b-versatile": ModelPricing(input_per_1k=0.0, output_per_1k=0.0),
    # Cerebras free tier
    "cerebras/llama-3.3-70b": ModelPricing(input_per_1k=0.0, output_per_1k=0.0),
    # SambaNova free tier
    "sambanova/DeepSeek-R1": ModelPricing(input_per_1k=0.0, output_per_1k=0.0),
    # OpenRouter free tier (Llama 3.3 70B)
    "openrouter/meta-llama/llama-3.3-70b-instruct:free": ModelPricing(
        input_per_1k=0.0, output_per_1k=0.0
    ),
    # Anthropic Claude Sonnet 4.5 (verify rates before headline run)
    "anthropic/claude-sonnet-4-5": ModelPricing(input_per_1k=0.003, output_per_1k=0.015),
    # Local vLLM
    "vllm/Qwen/Qwen3-8B": ModelPricing(input_per_1k=0.0, output_per_1k=0.0),
}


def lookup(vendor: str, model: str) -> ModelPricing:
    key = f"{vendor}/{model}"
    return PRICING.get(key, ModelPricing(input_per_1k=0.0, output_per_1k=0.0))
