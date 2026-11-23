"""Model backends for `tea run`. Each exposes

    complete(text, model, **options) -> {"text": str, "stop_reason": str | None, "usage": dict | None}

`text` is the request, sent as a single user message with no system prompt.
"""


def get(provider):
    if provider == 'anthropic':
        from . import anthropic_backend
        return anthropic_backend.complete
    if provider == 'openai':
        from . import openai_compatible
        return openai_compatible.complete
    raise ValueError(f"unknown provider {provider!r}; use 'anthropic' or 'openai'")
