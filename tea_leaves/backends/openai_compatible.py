"""Any server that speaks the OpenAI Chat Completions protocol: a hosted API, or an open-weight
model served locally (for example with vLLM). Standard library only.

    --base-url   default: $OPENAI_BASE_URL or https://api.openai.com/v1
    key          read from $OPENAI_API_KEY (or the variable named by --api-key-env)
"""
import json
import os
import time
import urllib.error
import urllib.request


def complete(text, model, base_url=None, api_key_env='OPENAI_API_KEY', max_tokens=None, extra=None,
             timeout=600, retries=4, **_):
    base = (base_url or os.environ.get('OPENAI_BASE_URL') or 'https://api.openai.com/v1').rstrip('/')
    body = {'model': model, 'messages': [{'role': 'user', 'content': text}]}
    if max_tokens:
        body['max_tokens'] = max_tokens
    body.update(extra or {})
    headers = {'Content-Type': 'application/json'}
    key = os.environ.get(api_key_env or '')
    if key:
        headers['Authorization'] = f'Bearer {key}'
    request = urllib.request.Request(base + '/chat/completions', data=json.dumps(body).encode(), headers=headers)
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read())
            choice = payload['choices'][0]
            return {'text': choice['message'].get('content') or '', 'stop_reason': choice.get('finish_reason'),
                    'usage': payload.get('usage')}
        except urllib.error.HTTPError as exc:
            if exc.code not in (408, 409, 429) and exc.code < 500 or attempt == retries:
                raise RuntimeError(f'HTTP {exc.code}: {exc.read().decode(errors="replace")[:300]}') from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == retries:
                raise RuntimeError(f'connection failed: {exc}') from exc
        time.sleep(min(60, 2 ** attempt * 2))
