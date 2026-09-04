"""Small LLM helper for research stages, with per-call usage tracking."""

import json
import time
from typing import Any, List, Optional, Tuple

from tenacity import retry, stop_after_attempt, wait_random_exponential

from app.llm import LLM
from app.research.models import LLMUsage


@retry(
    wait=wait_random_exponential(min=2, max=30),
    stop=stop_after_attempt(4),
)
async def call_llm(
    messages: List[dict],
    max_tokens: int = 4096,
    temperature: float = 0.0,
) -> Tuple[str, LLMUsage]:
    """Non-streaming chat completion returning (content, usage)."""
    llm = LLM()
    started = time.perf_counter()
    response = await llm.client.chat.completions.create(
        model=llm.model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=False,
        timeout=180,
    )
    latency_ms = (time.perf_counter() - started) * 1000
    if not response.choices or not response.choices[0].message:
        raise ValueError("Empty or invalid response from LLM")
    content = response.choices[0].message.content or ""
    usage = LLMUsage(
        model=llm.model,
        input_tokens=response.usage.prompt_tokens if response.usage else 0,
        output_tokens=response.usage.completion_tokens if response.usage else 0,
        latency_ms=round(latency_ms, 1),
    )
    return content, usage


def extract_json(text: str) -> Any:
    """Parse JSON from LLM output, tolerating markdown fences and prose.

    Raises ValueError if no JSON object/array can be extracted.
    """
    text = text.strip()
    lines = [line for line in text.splitlines() if not line.strip().startswith("```")]
    cleaned = "\n".join(lines).strip()
    candidates = [cleaned]
    start = min(
        (i for i in (cleaned.find("["), cleaned.find("{")) if i >= 0),
        default=-1,
    )
    if start >= 0:
        end = max(cleaned.rfind("]"), cleaned.rfind("}"))
        if end > start:
            candidates.append(cleaned[start : end + 1])
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise ValueError(f"No JSON found in LLM output: {text[:200]}")
