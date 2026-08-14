"""LLM access layer: one client, structured outputs, prompt assets, caching,
budgets. No agent talks to a provider SDK directly."""

from __future__ import annotations

from .budget import BudgetTracker
from .cache import ResponseCache
from .client import FakeLLMClient, LLMClient, LLMResponse, ToolCall
from .prompts import Prompt, PromptLibrary
from .structured import generate_structured, schema_for

__all__ = [
    "BudgetTracker",
    "FakeLLMClient",
    "LLMClient",
    "LLMResponse",
    "Prompt",
    "PromptLibrary",
    "ResponseCache",
    "ToolCall",
    "generate_structured",
    "schema_for",
]
