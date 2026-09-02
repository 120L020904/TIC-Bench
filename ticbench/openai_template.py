"""The repository's only model-service adapter.

All benchmark logic accepts a plain callable, so this module can be replaced
without changing prompt construction or evaluation.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def create_openai_caller(model: str) -> Callable[[list[dict[str, Any]]], str]:
    """Return a callable backed by the OpenAI Responses API.

    Authentication and client configuration are intentionally delegated to the
    official SDK. No credential or endpoint argument is accepted here.
    """
    from openai import OpenAI

    client = OpenAI()

    def call(messages: list[dict[str, Any]]) -> str:
        response = client.responses.create(
            model=model,
            input=messages,
            store=False,
        )
        return response.output_text.strip()

    return call
