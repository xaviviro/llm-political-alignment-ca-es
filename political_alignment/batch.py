"""Batch-API runner for the survey-alignment method (≈50% cost, asynchronous).

The whole study is a large set of independent, latency-insensitive calls — the
ideal batch workload. This module:

1. ``build_requests`` — generate every (item × language × framing × template ×
   sample) request, each with a ``custom_id`` that encodes where it belongs.
2. provider adapters — submit the batch and collect the results:
   - **OpenAI-compatible** (OpenAI, Groq): the files + batches endpoints.
   - **Anthropic**: the Message Batches API.
   - **Gemini**: the google-genai batch API (inline requests).
3. ``results_to_eval`` — parse the returned texts into per-template
   distributions and reuse :func:`political_alignment.survey_alignment.aggregate`,
   producing the same ``evals/<id>.json`` as a live run.

Steps 1 and 3 are pure and unit-tested. The submit/collect adapters need real
API keys; verify each provider on its first real run — batch endpoints differ.
"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict

import numpy as np

from . import survey_alignment
from .framings import render_variants
from .providers import _letter_from_text

# custom_id separator. Anthropic only allows [a-zA-Z0-9_-], so "__" (no field
# contains it). The parser also accepts the legacy "|" so batches submitted
# before this change can still be collected.
SEP = "__"


def provider_of(litellm_model: str) -> tuple[str, str]:
    """Split a LiteLLM model id into (provider, raw model id for native SDK).

    ``groq/openai/gpt-oss-120b`` -> ``("groq", "openai/gpt-oss-120b")``;
    ``anthropic/claude-opus-4-8`` -> ``("anthropic", "claude-opus-4-8")``.
    """
    provider, _, raw = litellm_model.partition("/")
    return provider, raw


def _cid(item_id, lang, framing, tmpl, sample) -> str:
    return SEP.join([item_id, lang, framing, str(tmpl), str(sample)])


def _parse_cid(cid: str):
    sep = "|" if "|" in cid else "__"
    item_id, lang, framing, tmpl, sample = cid.split(sep)
    return item_id, lang, framing, int(tmpl), int(sample)


# markers of an explicit refusal ("as an AI I have no political opinions…")
_REFUSAL_MARKERS = (
    "no tinc opinions", "no tengo opiniones", "no puc posicionar", "no puedo posicionar",
    "no puc escollir", "no puedo elegir", "no puc triar", "com a model", "como modelo",
    "model de llenguatge", "modelo de lenguaje", "as an ai", "language model",
    "i cannot", "i can't", "i'm unable", "i am unable", "no puedo", "lo siento",
)


def classify_failure(text: str) -> str:
    """Why a sample produced no parseable option letter: refusal / empty / unparsed."""
    t = (text or "").strip().lower()
    if not t:
        return "empty"
    if any(m in t for m in _REFUSAL_MARKERS):
        return "refusal"
    return "unparsed"


# -- 1. request generation (pure) ------------------------------------------
def build_requests(items, config: dict, *, raw_model: str) -> list[dict]:
    sampling = config.get("sampling", {})
    n_samples = sampling.get("n_samples", 20)
    temperature = sampling.get("temperature", 1.0)
    max_tokens = sampling.get("max_tokens", 512)
    languages = config.get("languages", ["ca", "es"])
    framings = config.get("framings", ["neutral", "personalised", "observer"])

    reqs = []
    for item in items:
        for framing in framings:
            for lang in languages:
                for tmpl, prompt, _letters in render_variants(item, lang, framing):
                    for s in range(n_samples):
                        reqs.append({
                            "custom_id": _cid(item.item_id, lang, framing, tmpl, s),
                            "model": raw_model,
                            "prompt": prompt,
                            "max_tokens": max_tokens,
                            "temperature": temperature,
                        })
    return reqs


# -- 3. results -> eval (pure) ---------------------------------------------
def results_to_eval(model_id: str, items, config: dict, results: dict) -> dict:
    """``results``: {custom_id: completion_text}. Returns the eval dict."""
    texts = defaultdict(lambda: defaultdict(list))  # (item,lang,framing) -> tmpl -> [text]
    for cid, text in results.items():
        item_id, lang, framing, tmpl, _s = _parse_cid(cid)
        texts[(item_id, lang, framing)][tmpl].append(text)

    def template_probe(item, lang, framing):
        per_tmpl = texts.get((item.item_id, lang, framing), {})
        out = []
        for tmpl, _prompt, letters in render_variants(item, lang, framing):
            counts = {ltr: 0.0 for ltr in letters}
            n_fail = 0
            reasons = defaultdict(int)
            examples = []
            tl = per_tmpl.get(tmpl, [])
            for t in tl:
                ltr = _letter_from_text(t, letters)
                if ltr is not None:
                    counts[ltr] += 1.0
                else:
                    n_fail += 1
                    reasons[classify_failure(t)] += 1
                    if len(examples) < 2 and (t or "").strip():
                        examples.append((t or "").strip()[:200])
            n = len(tl)
            vec = np.array([counts[ltr] for ltr in letters], dtype=np.float64)
            if vec.sum() <= 1e-9:
                dist = np.full(len(letters), 1.0 / len(letters))
            else:
                vec = vec + 1.0 / max(n, 1)  # Laplace smoothing
                dist = vec / vec.sum()
            out.append((dist, {"method": "batch", "n": n, "n_fail": n_fail,
                               "fail_reasons": dict(reasons), "fail_examples": examples}))
        return out

    return survey_alignment.aggregate(model_id, items, config, template_probe)


# -- 2a. OpenAI-compatible (OpenAI, Groq) ----------------------------------
_OPENAI_COMPAT = {
    "openai": {"base_url": None, "key": "OPENAI_API_KEY"},
    "groq": {"base_url": "https://api.groq.com/openai/v1", "key": "GROQ_API_KEY"},
}


def _openai_client(provider: str):
    from openai import OpenAI
    cfg = _OPENAI_COMPAT[provider]
    return OpenAI(api_key=os.environ[cfg["key"]], base_url=cfg["base_url"])


def _chat_body(provider: str, r: dict) -> dict:
    body = {"model": r["model"], "messages": [{"role": "user", "content": r["prompt"]}]}
    if provider == "openai":
        # gpt-5 / reasoning models reject `max_tokens` (need `max_completion_tokens`)
        # and force the default temperature; give reasoning room.
        body["max_completion_tokens"] = max(r["max_tokens"], 2048)
    else:
        body["max_tokens"] = r["max_tokens"]
        body["temperature"] = r["temperature"]
    return body


def submit_openai_compatible(provider: str, requests: list[dict],
                             completion_window: str = "24h") -> str:
    client = _openai_client(provider)
    lines = [json.dumps({
        "custom_id": r["custom_id"], "method": "POST", "url": "/v1/chat/completions",
        "body": _chat_body(provider, r),
    }) for r in requests]
    data = "\n".join(lines).encode()
    fobj = client.files.create(file=("batch.jsonl", data), purpose="batch")
    batch = client.batches.create(input_file_id=fobj.id,
                                  endpoint="/v1/chat/completions",
                                  completion_window=completion_window)
    return batch.id


def collect_openai_compatible(provider: str, batch_id: str, poll: int = 30) -> dict:
    client = _openai_client(provider)
    while True:
        b = client.batches.retrieve(batch_id)
        if b.status in ("completed", "failed", "cancelled", "expired"):
            break
        time.sleep(poll)
    if b.status != "completed":
        raise RuntimeError(f"batch {batch_id} ended with status {b.status}")
    content = client.files.content(b.output_file_id).text
    results = {}
    for line in content.splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        body = (obj.get("response") or {}).get("body") or {}
        choices = body.get("choices") or [{}]
        results[obj["custom_id"]] = (choices[0].get("message") or {}).get("content") or ""
    return results


# -- 2b. Anthropic Message Batches -----------------------------------------
def submit_anthropic(requests: list[dict]) -> str:
    from anthropic import Anthropic
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    reqs = [{"custom_id": r["custom_id"],
             "params": {"model": r["model"], "max_tokens": r["max_tokens"],
                        "temperature": r["temperature"],
                        "messages": [{"role": "user", "content": r["prompt"]}]}}
            for r in requests]
    return client.messages.batches.create(requests=reqs).id


def collect_anthropic(batch_id: str, poll: int = 30) -> dict:
    from anthropic import Anthropic
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    while client.messages.batches.retrieve(batch_id).processing_status != "ended":
        time.sleep(poll)
    results = {}
    for r in client.messages.batches.results(batch_id):
        if r.result.type == "succeeded":
            txt = "".join(b.text for b in r.result.message.content
                          if getattr(b, "type", "") == "text")
        else:
            txt = ""
        results[r.custom_id] = txt
    return results


# -- 2c. Gemini batch (google-genai) — EXPERIMENTAL, verify on first run ----
def submit_gemini(raw_model: str, requests: list[dict]) -> tuple[str, list[str]]:
    """Returns (batch job name, custom_id order) — Gemini inline batches map by
    order, so we keep the custom_id sequence to re-key the responses."""
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    inline = [types.InlinedRequest(
        contents=[types.Content(role="user", parts=[types.Part(text=r["prompt"])])],
        config=types.GenerateContentConfig(max_output_tokens=r["max_tokens"],
                                           temperature=r["temperature"]),
    ) for r in requests]
    job = client.batches.create(model=raw_model, src=inline)
    return job.name, [r["custom_id"] for r in requests]


def collect_gemini(job_name: str, custom_ids: list[str], poll: int = 30) -> dict:
    from google import genai
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    while True:
        job = client.batches.get(name=job_name)
        if str(job.state).endswith(("SUCCEEDED", "FAILED", "CANCELLED")):
            break
        time.sleep(poll)
    results = {}
    for cid, resp in zip(custom_ids, job.dest.inlined_responses, strict=False):
        try:
            results[cid] = resp.response.text or ""
        except Exception:  # noqa: BLE001
            results[cid] = ""
    return results
