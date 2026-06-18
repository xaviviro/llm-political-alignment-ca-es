"""Model access via LiteLLM, with answer-distribution estimation.

A provider turns a multiple-choice prompt into a probability distribution over
the option letters. Two strategies:

- ``logprobs``  — one call with OpenAI-style ``logprobs``; read the probability
  mass on each option-letter token at the first generated position. Cheapest
  and most faithful, where the provider exposes it.
- ``sampling``  — sample ``n_samples`` completions at ``temperature`` and count
  the chosen letters. Works on any chat API, including local models via Ollama.

``MockProvider`` is a deterministic, network-free provider for tests.

Every hosted model is reached through ``litellm.completion`` so any LiteLLM
provider id works; API keys come from the environment, never from config.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np

EPS = 1e-9


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_MARKER_RE = re.compile(r"(?:resposta|respuesta|answer)\s*:?\s*\**\s*([A-Z])\b", re.IGNORECASE)


def _letter_from_text(text: str, letters: list[str]) -> str | None:
    """Parse the chosen option letter from a (possibly reasoning) completion.

    Reasoning models emit ``<think>...</think>`` blocks and explanations, so we
    strip the thinking, prefer an explicit "Resposta: X" marker, and otherwise
    fall back to the LAST standalone option letter (the final answer usually
    comes last). Returns None if no option letter is found.
    """
    cleaned = _THINK_RE.sub(" ", text)
    valid = set(letters)
    m = _MARKER_RE.search(cleaned)
    if m and m.group(1).upper() in valid:
        return m.group(1).upper()
    # last standalone letter that is a valid option
    found = re.findall(r"\b([A-Z])\b", cleaned.upper())
    for ltr in reversed(found):
        if ltr in valid:
            return ltr
    return None


@dataclass
class Provider:
    """Base provider. Subclasses implement ``answer_distribution``."""

    id: str

    def answer_distribution(self, prompt: str, letters: list[str]) -> tuple[np.ndarray, dict]:
        """Return (distribution over letters, info dict with parse diagnostics)."""
        raise NotImplementedError


class LiteLLMProvider(Provider):
    def __init__(self, id, model, method="logprobs", supports_logprobs=False,
                 api_base=None, n_samples=20, temperature=1.0, max_retries=2,
                 max_tokens=512, max_workers=1, n_per_category=0):
        self.id = id
        self.model = model
        self.method = method
        self.supports_logprobs = supports_logprobs
        self.api_base = api_base
        self.n_samples = n_samples
        self.temperature = temperature
        self.max_retries = max_retries
        # generation budget for sampling: reasoning models need room to finish
        # their hidden reasoning before emitting the answer letter.
        self.max_tokens = max_tokens
        # concurrent samples per call (independent draws); speeds up sync sampling.
        self.max_workers = max(1, int(max_workers))
        # if >0, scale the sample count with the number of options so items with
        # many categories get enough mass: n = max(n_samples, n_per_category * k).
        self.n_per_category = max(0, int(n_per_category))

    # -- low-level call ----------------------------------------------------
    def _complete(self, prompt, **kwargs):
        import litellm
        last_err = None
        for _ in range(self.max_retries + 1):
            try:
                return litellm.completion(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    api_base=self.api_base,
                    **kwargs,
                )
            except Exception as e:  # noqa: BLE001 — surface after retries
                last_err = e
        raise RuntimeError(f"{self.model}: completion failed after retries: {last_err}")

    # -- distribution strategies ------------------------------------------
    def _dist_logprobs(self, prompt, letters):
        resp = self._complete(prompt, max_tokens=1, temperature=0.0,
                              logprobs=True, top_logprobs=20)
        content = resp.choices[0].logprobs.content
        mass = {ltr: 0.0 for ltr in letters}
        if content:
            for cand in content[0].top_logprobs:
                tok = cand.token.strip().upper()
                if tok in mass:
                    mass[tok] += math.exp(cand.logprob)
        vec = np.array([mass[ltr] for ltr in letters], dtype=np.float64)
        if vec.sum() <= EPS:
            # no letter token surfaced — fall back to sampling
            return self._dist_sampling(prompt, letters)
        return vec / vec.sum(), {"method": "logprobs", "n": 1, "n_fail": 0}

    def _one_sample(self, prompt, letters):
        resp = self._complete(prompt, max_tokens=self.max_tokens,
                              temperature=self.temperature)
        text = resp.choices[0].message.content or ""
        return _letter_from_text(text, letters)

    def _dist_sampling(self, prompt, letters, n=None):
        n = n or self.n_samples
        if self.n_per_category:
            n = max(n, self.n_per_category * len(letters))
        counts = {ltr: 0.0 for ltr in letters}
        n_fail = 0
        if self.max_workers > 1 and n > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=min(self.max_workers, n)) as ex:
                drawn = list(ex.map(lambda _: self._one_sample(prompt, letters), range(n)))
        else:
            drawn = [self._one_sample(prompt, letters) for _ in range(n)]
        for ltr in drawn:
            if ltr is not None:
                counts[ltr] += 1.0
            else:
                n_fail += 1
        info = {"method": "sampling", "n": n, "n_fail": n_fail}
        vec = np.array([counts[ltr] for ltr in letters], dtype=np.float64)
        if vec.sum() <= EPS:
            # every sample was unparseable — return uniform but flag it so the
            # analysis can exclude this response rather than treat it as real.
            return np.full(len(letters), 1.0 / len(letters)), info
        # Laplace smoothing so unseen options keep tiny mass.
        vec = vec + 1.0 / n
        return vec / vec.sum(), info

    def answer_distribution(self, prompt, letters):
        if self.method == "logprobs" and self.supports_logprobs:
            return self._dist_logprobs(prompt, letters)
        return self._dist_sampling(prompt, letters)


class MockProvider(Provider):
    """Deterministic, network-free provider for tests and dry runs.

    Produces a stable distribution from a hash of the prompt so end-to-end runs
    are reproducible without any API. Not a real measurement.
    """

    def __init__(self, id="mock", seed=0):
        self.id = id
        self.seed = seed

    def answer_distribution(self, prompt, letters):
        h = abs(hash((self.seed, prompt))) % (2**32)
        rng = np.random.default_rng(h)
        vec = rng.dirichlet(np.ones(len(letters)))
        return vec, {"method": "mock", "n": 0, "n_fail": 0}


def build_provider(model_cfg: dict, global_cfg: dict) -> Provider:
    """Construct a provider from a models.yaml entry + config.yaml."""
    if model_cfg.get("mock"):
        return MockProvider(id=model_cfg["id"], seed=model_cfg.get("seed", 0))
    sampling = global_cfg.get("sampling", {})
    return LiteLLMProvider(
        id=model_cfg["id"],
        model=model_cfg["model"],
        method=global_cfg.get("distribution_method", "logprobs"),
        supports_logprobs=model_cfg.get("supports_logprobs", False),
        api_base=model_cfg.get("api_base"),
        n_samples=sampling.get("n_samples", 20),
        temperature=sampling.get("temperature", 1.0),
        max_tokens=model_cfg.get("max_tokens", sampling.get("max_tokens", 512)),
        max_workers=model_cfg.get("max_workers", sampling.get("max_workers", 1)),
        n_per_category=sampling.get("n_samples_per_category", 0),
    )
