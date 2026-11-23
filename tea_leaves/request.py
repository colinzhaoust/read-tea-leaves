"""The exact text sent to a model.

    text = PROMPT + "\\nINPUT:\\n" + json.dumps(input, sort_keys=True, separators=(",", ":"))

The prompt is one line, ``INPUT:`` is on its own line, and the input object is one line of
compact JSON with sorted keys. The text is sent as a single user message with no system
prompt. Every request of the paper was built this way.
"""
import json
from pathlib import Path

PROMPTS = Path(__file__).resolve().parent / 'prompts'


def prompt(task='prediction'):
    return (PROMPTS / f'{task}.txt').read_text(encoding='utf-8')


def build_request(input_object, task='prediction'):
    """Return the request text for one input object (a dict)."""
    return prompt(task) + '\nINPUT:\n' + json.dumps(input_object, sort_keys=True, separators=(',', ':'),
                                                    ensure_ascii=False)


def split_request(text):
    """Inverse of build_request: (prompt, input object)."""
    head, payload = text.split('\nINPUT:\n', 1)
    return head, json.loads(payload)
