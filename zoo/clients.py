"""Chat client with on-disk caching, retries, and SQL extraction.

Caching is keyed by (model, prompt-style, messages, temperature, max_tokens), so
re-running never re-pays for a completed call and the whole zoo is resumable.
Currently only the OpenAI provider is wired (the Together key returns 403 on this
machine); adding another provider is a new branch in `_raw_call`.
"""
import hashlib
import json
import os
import re
import time
import urllib.request
import urllib.error

from .config import ARTIFACT_ROOT

_CACHE = os.path.join(ARTIFACT_ROOT, "gen_cache")
os.makedirs(_CACHE, exist_ok=True)


def _key(provider, model, messages, temperature, max_tokens):
    blob = json.dumps([provider, model, messages, temperature, max_tokens],
                      sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def _raw_call(provider, model, messages, temperature, max_tokens):
    if provider != "openai":
        raise ValueError(f"provider {provider} not configured on this machine")
    key = os.environ["OPENAI_API_KEY"]
    body = {"model": model, "messages": messages}
    # o-series and gpt-5 reasoning models use max_completion_tokens and reject a custom
    # temperature (only the default is allowed); the SQL manifest avoids them, but the
    # gpt-5-mini JUDGE (zoo/judge.py) needs this branch.
    if model.startswith(("o1", "o3", "o4", "gpt-5")):
        body["max_completion_tokens"] = max(max_tokens, 2048)
    else:
        body["max_tokens"] = max_tokens
        body["temperature"] = temperature
    data = json.dumps(body).encode()
    req = urllib.request.Request("https://api.openai.com/v1/chat/completions",
                                 data=data,
                                 headers={"Authorization": f"Bearer {key}",
                                          "Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=90))
    return r["choices"][0]["message"]["content"]


def chat(provider, model, messages, temperature=0.0, max_tokens=512, retries=4):
    ck = _key(provider, model, messages, temperature, max_tokens)
    cpath = os.path.join(_CACHE, ck + ".json")
    if os.path.exists(cpath):
        with open(cpath) as f:
            return json.load(f)["content"]
    last = None
    for attempt in range(retries):
        try:
            content = _raw_call(provider, model, messages, temperature, max_tokens)
            with open(cpath, "w") as f:
                json.dump({"model": model, "content": content}, f)
            return content
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (429, 500, 502, 503, 529):
                time.sleep(2 ** attempt)
                continue
            raise
        except Exception as e:  # transient network
            last = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"chat failed for {model}: {last}")


_FENCE = re.compile(r"```(?:sql)?\s*(.*?)```", re.S | re.I)


def extract_sql(text: str) -> str:
    """Pull a single SQL statement out of a model response."""
    if not text:
        return ""
    m = _FENCE.search(text)
    if m:
        text = m.group(1)
    # drop a leading 'SQL:' label and surrounding whitespace
    text = re.sub(r"^\s*SQL\s*:\s*", "", text.strip(), flags=re.I)
    # take up to the first semicolon if present, else the first non-empty block
    if ";" in text:
        text = text.split(";")[0]
    # collapse to the first statement starting at SELECT/WITH if the model rambled
    mm = re.search(r"\b(SELECT|WITH)\b", text, re.I)
    if mm:
        text = text[mm.start():]
    return text.strip()
