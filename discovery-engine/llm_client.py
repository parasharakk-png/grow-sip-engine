"""Anthropic-compatible gateway client and answer streaming."""

from __future__ import annotations

from typing import Any, Iterator

REQUEST_TIMEOUT_S = 120
MAX_OUTPUT_TOKENS = 12000


def make_client(api_key: str, base_url: str | None) -> Any:
    import anthropic

    # One attempt with a hard timeout: SDK defaults (600s read, 2 retries) turned a slow
    # reasoning model into 5–8 minute waits.
    return anthropic.Anthropic(
        api_key=api_key, base_url=base_url, timeout=REQUEST_TIMEOUT_S, max_retries=0
    )


def stream_text(client: Any, model: str, prompt: str, max_tokens: int = MAX_OUTPUT_TOKENS) -> Iterator[str]:
    """Yield answer text as it arrives. Reasoning models think first, so max_tokens must leave room
    for both thinking and the answer."""
    got_text = False
    with client.messages.stream(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for event in stream:
            if event.type == "content_block_delta" and event.delta.type == "text_delta":
                got_text = True
                yield event.delta.text
        final = stream.get_final_message()
    if not got_text:
        raise RuntimeError(f"model returned no answer text (stop_reason={final.stop_reason})")
