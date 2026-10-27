import json
import os
import re
import sys
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

URL = 'https://openrouter.ai/api/v1/chat/completions'
MAX_TOKENS = 32768
MAX_ATTEMPTS = 3
REQUEST_DEADLINE_SECONDS = 300
RETRY_BASE_DELAY_SECONDS = 10
RETRYABLE_STATUSES = {429, 500, 502, 503, 504}
JSON_LIST = re.compile(r'\[.*\]', re.S)


class NoValidAnswer(Exception):
    pass


def session():
    load_dotenv(Path(__file__).resolve().parents[1] / '.env')
    headers = {'Authorization': f'Bearer {os.environ["GRAPHRAG_API_KEY"]}'}
    return httpx.Client(headers=headers, timeout=REQUEST_DEADLINE_SECONDS)


def post(client, request):
    deadline = time.monotonic() + REQUEST_DEADLINE_SECONDS
    try:
        with client.stream('POST', URL, json=request) as response:
            body = b''
            for chunk in response.iter_bytes():
                if time.monotonic() > deadline:
                    return None
                body += chunk
    except httpx.TransportError:
        return None
    if response.status_code != 200:
        return response.status_code, None
    try:
        return response.status_code, json.loads(body)
    except json.JSONDecodeError:
        return None


def parse_list(content):
    found = JSON_LIST.search(content or '')
    try:
        return json.loads(found.group(0)) if found else None
    except json.JSONDecodeError:
        return None


def ask_list(client, model, prompt, is_valid):
    request = {**model, 'temperature': 0, 'max_tokens': MAX_TOKENS, 'messages': [{'role': 'user', 'content': prompt}]}
    for attempt in range(MAX_ATTEMPTS):
        result = post(client, request)
        if result is None or result[0] in RETRYABLE_STATUSES:
            print(f"{model['model']}: retry after {'deadline' if result is None else result[0]}", file=sys.stderr)
            time.sleep(RETRY_BASE_DELAY_SECONDS * 2 ** attempt)
            continue
        status, payload = result
        if status != 200:
            raise RuntimeError(f'openrouter returned {status}')
        choice = payload['choices'][0]
        answer = parse_list(choice['message']['content'])
        if is_valid(answer):
            return answer
        print(f"{model['model']}: retry after an invalid answer, finish {choice['finish_reason']}", file=sys.stderr)
    raise NoValidAnswer(f'no valid answer after {MAX_ATTEMPTS} attempts')
