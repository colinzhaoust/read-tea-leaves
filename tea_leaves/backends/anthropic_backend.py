"""Claude models through the official Anthropic SDK (`pip install anthropic`).

Credentials are resolved by the SDK (ANTHROPIC_API_KEY, or a profile from `ant auth login`).
"""
_client = None


def complete(text, model='claude-opus-5-5', max_tokens=16000, effort=None, **_):
    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover
        raise SystemExit('The anthropic provider needs the SDK:  pip install anthropic') from exc
    global _client
    if _client is None:
        _client = anthropic.Anthropic(max_retries=4)
    params = {'model': model, 'max_tokens': max_tokens, 'messages': [{'role': 'user', 'content': text}]}
    if effort:
        params['output_config'] = {'effort': effort}
    # No fallback model is configured on purpose: a benchmark answer must come from the model
    # under test. A refusal is recorded as an unanswered question.
    with _client.messages.stream(**params) as stream:
        message = stream.get_final_message()
    answer = ''.join(block.text for block in message.content if block.type == 'text')
    usage = {'input_tokens': message.usage.input_tokens, 'output_tokens': message.usage.output_tokens}
    return {'text': '' if message.stop_reason == 'refusal' else answer,
            'stop_reason': message.stop_reason, 'usage': usage}
