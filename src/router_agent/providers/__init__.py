"""LLM providers behind one ``LLMBackend`` contract, plus a registry to build them by name."""

from __future__ import annotations

from collections.abc import Mapping

from router_agent.providers.base import BackendError, BackendUnavailable, LLMBackend
from router_agent.providers.groq import GroqBackend
from router_agent.providers.huggingface import HuggingFaceBackend
from router_agent.providers.mock import LLMCall, MockBackend
from router_agent.providers.ollama import OllamaBackend
from router_agent.providers.watsonx import WatsonxBackend

BACKENDS = {
    "watsonx": WatsonxBackend,
    "groq": GroqBackend,
    "huggingface": HuggingFaceBackend,
    "ollama": OllamaBackend,
}


def backend_from_env(name: str, env: Mapping[str, str] | None = None) -> LLMBackend:
    """Build a real backend from environment variables."""
    if name not in BACKENDS:
        raise ValueError(f"unknown backend {name!r}; choose one of {sorted(BACKENDS)}")
    return BACKENDS[name].from_env(env)


__all__ = [
    "BACKENDS",
    "BackendError",
    "BackendUnavailable",
    "GroqBackend",
    "HuggingFaceBackend",
    "LLMBackend",
    "LLMCall",
    "MockBackend",
    "OllamaBackend",
    "WatsonxBackend",
    "backend_from_env",
]
