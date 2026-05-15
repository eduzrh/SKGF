"""
Token-level probability estimation for entropy-driven validity (paper §Entropy-driven
Validity Estimation via Token Probability).

We task the LLM with predicting a binary verification token y in {"Yes", "No"}.
The validity probability P(f | G_local^s) is computed from the raw logits of the
first generated token:

    P(f | G_local^s) = exp(l("Yes")) / (exp(l("Yes")) + exp(l("No"))

This is the canonical softmax-normalized probability over the restricted binary
vocabulary {"Yes", "No"}, following the formulation in the paper.

If the underlying OpenAI endpoint does not return `top_logprobs`, the function
falls back to parsing the textual answer (default 1.0 for "Yes", 0.0 otherwise)
and emits a warning. The caller is expected to log which mode was used.
"""

from __future__ import annotations

import math
import os
from typing import List, Tuple, Dict, Any, Optional

import httpx
from openai import OpenAI


# Canonical verification tokens. We use the exact strings in the paper's Prompt 1.
POSITIVE_TOKEN = "Yes"
NEGATIVE_TOKEN = "No"

# Tokens we treat as semantically equivalent to the canonical ones (parses both
# upper- and lower-case responses emitted by chat-tuned models).
_ALIASES = {
    "yes": POSITIVE_TOKEN,
    "no": NEGATIVE_TOKEN,
    "y": POSITIVE_TOKEN,
    "n": NEGATIVE_TOKEN,
    "true": POSITIVE_TOKEN,
    "false": NEGATIVE_TOKEN,
    "1": POSITIVE_TOKEN,
    "0": NEGATIVE_TOKEN,
}


def _resolve_alias(token: str) -> Optional[str]:
    """Map a free-form token to {"Yes", "No"} if possible."""
    if token is None:
        return None
    t = token.strip().strip(".,;:!?\"' \t\n")
    if not t:
        return None
    if t in (POSITIVE_TOKEN, NEGATIVE_TOKEN):
        return t
    if t.lower() in _ALIASES:
        return _ALIASES[t.lower()]
    # Some models emit multi-token answers; use the first word heuristically.
    head = t.split()[0]
    if head in (POSITIVE_TOKEN, NEGATIVE_TOKEN):
        return head
    if head.lower() in _ALIASES:
        return _ALIASES[head.lower()]
    return None


def _client_from_env(api_base: Optional[str] = None, api_key: Optional[str] = None) -> OpenAI:
    base = api_base or os.environ.get("OPENAI_API_BASE", "yours")
    key = api_key or os.environ.get("OPENAI_API_KEY", "yours")
    return OpenAI(
        base_url=base,
        api_key=key,
        http_client=httpx.Client(base_url=base, follow_redirects=True),
    )


def _extract_binary_logits(top_logprobs: List[Dict[str, Any]]) -> Tuple[Optional[float], Optional[float], bool]:
    """
    Pull the logit values for "Yes" and "No" out of the OpenAI `top_logprobs`
    payload (list of dicts mapping token -> {logprob, ...}).

    Returns (l_yes, l_no, found_both).
    """
    if not top_logprobs:
        return None, None, False

    # We only inspect the first generated token (top_logprobs[0]).
    candidates = top_logprobs[0] or {}
    yes_logit = None
    no_logit = None

    # The OpenAI chat completion API normalizes `logprob` (already log(p)),
    # so to get a raw logit we would need `top_logprobs[i].logprob` (already a log-prob).
    # The paper's formula uses raw logits; here we use the log-probabilities and
    # exponentiate to obtain probabilities (mathematically equivalent up to a
    # constant shift, which is cancelled by the softmax over a binary vocabulary).
    for tok, info in candidates.items():
        canonical = _resolve_alias(tok)
        if canonical is None:
            continue
        # Some endpoints return {"token": ..., "logprob": ...}; others return
        # just the log-probability as a float.
        if isinstance(info, dict):
            lp = info.get("logprob")
        else:
            lp = info
        if lp is None:
            continue
        if canonical == POSITIVE_TOKEN and yes_logit is None:
            yes_logit = float(lp)
        elif canonical == NEGATIVE_TOKEN and no_logit is None:
            no_logit = float(lp)

    return yes_logit, no_logit, (yes_logit is not None and no_logit is not None)


def logprob_yes_no(
    prompt: str,
    client: Optional[OpenAI] = None,
    model: str = "gpt-4o-mini",
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    max_tokens: int = 1,
    use_logprob_fallback: bool = True,
) -> float:
    """
    Estimate P(f | G_local^s) using token log-probabilities (paper §token_entropy).

    The returned probability is the softmax-normalized mass on "Yes" over the
    binary vocabulary {"Yes", "No"}.

    If the underlying endpoint does not expose `top_logprobs`, the function
    degrades to a textual parser (defaulting to 1.0 on "Yes" and 0.0 otherwise)
    when `use_logprob_fallback` is True; otherwise it raises RuntimeError.
    """
    if client is None:
        client = _client_from_env(api_base, api_key)

    p_yes = None
    used_logprob = False
    raw_answer = ""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=max_tokens,
            logprobs=True,
            top_logprobs=20,
        )
        choice = response.choices[0]
        raw_answer = (choice.message.content or "").strip()
        top_lp = getattr(choice, "logprobs", None)
        if top_lp is not None and getattr(top_lp, "content", None):
            top_logprobs = [dict(item) for item in top_lp.content]
            yes_lp, no_lp, ok = _extract_binary_logits(top_logprobs)
            if ok:
                # Softmax over the two log-probabilities (log-probs are already log(p),
                # so this is the same distribution as softmax over raw logits up to a
                # shared constant that cancels in normalization).
                m = max(yes_lp, no_lp)
                exp_y = math.exp(yes_lp - m)
                exp_n = math.exp(no_lp - m)
                p_yes = exp_y / (exp_y + exp_n)
                used_logprob = True
    except Exception as e:
        # Endpoint may not support logprobs (e.g., some proxy gateways).
        # Fall through to textual fallback below.
        print(f"[logprob_yes_no] logprob extraction failed: {e}")

    if p_yes is None:
        if not use_logprob_fallback:
            raise RuntimeError("logprobs unavailable and fallback disabled")
        # Textual fallback (verbalized confidence is known to be miscalibrated,
        # see paper §token_entropy). We deliberately keep this as the default
        # binary reading to preserve the paper's boolean output contract.
        canonical = _resolve_alias(raw_answer)
        if canonical == POSITIVE_TOKEN:
            p_yes = 1.0
        elif canonical == NEGATIVE_TOKEN:
            p_yes = 0.0
        else:
            # Ambiguous response: conservative default 0.5 (max-entropy).
            p_yes = 0.5
        used_logprob = False

    return float(p_yes)


def filter_facts_by_tau(
    facts: List[Tuple[Any, ...]],
    client: OpenAI,
    build_prompt,
    taus: Dict[Any, float],
    model: str = "gpt-4o-mini",
    concurrency: int = 16,
) -> List[Tuple[Any, ...]]:
    """
    Apply Eq. F_new in the paper:

        F_new = { (v, r, v̂) in C^g_meta(u_i^s) | P(f | G_local^s) >= tau, r in R^s }

    Args:
        facts: list of candidate facts (typically 5-tuples (h, r, t, t1, t2)).
        client: OpenAI client.
        build_prompt: callable fact -> str returning the verification prompt
                      (paper Prompt 1).
        taus: per-iteration entropy bound tau (currently a scalar float, but
              kept dict-shaped so that the per-fact interface is extensible).
        concurrency: number of parallel threads.

    Returns the filtered F_new.
    """
    import queue
    import threading
    from tqdm import tqdm

    sys_path_hack = "/root/shared-nvme/SKGF/DKGF-main/Self-Fusion-main"
    import sys
    if sys_path_hack not in sys.path:
        sys.path.append(sys_path_hack)
    try:
        from ThreadPoolExecutor import ThreadPoolExecutor
    except ImportError:
        from concurrent.futures import ThreadPoolExecutor

    tau = float(taus.get("__default__", 0.5) if isinstance(taus, dict) else taus)

    in_queue: "queue.Queue[Tuple[int, Tuple[Any, ...]]]" = queue.Queue()
    out_queue: "queue.Queue[Tuple[int, Tuple[Any, ...], float]]" = queue.Queue()
    for i, f in enumerate(facts):
        in_queue.put((i, f))

    lock = threading.Lock()
    stop_flag = threading.Event()
    accepted: List[Tuple[Any, ...]] = []

    def worker():
        while not stop_flag.is_set():
            try:
                idx, fact = in_queue.get_nowait()
            except queue.Empty:
                return
            try:
                prompt = build_prompt(fact)
                p = logprob_yes_no(prompt, client=client, model=model)
                if p >= tau:
                    out_queue.put((idx, fact, p))
            except Exception as e:
                print(f"[filter_facts_by_tau] error on fact {idx}: {e}")
            finally:
                in_queue.task_done()

    pool = ThreadPoolExecutor(max_workers=concurrency)
    workers = [pool.submit(worker) for _ in range(concurrency)]
    total = len(facts)

    with tqdm(total=total, desc=f"logprob-filter (tau={tau:.2f})") as pbar:
        processed = 0
        while processed < total:
            try:
                idx, fact, p = out_queue.get(timeout=5.0)
            except queue.Empty:
                continue
            with lock:
                accepted.append(fact)
            pbar.update(1)
            processed += 1

    stop_flag.set()
    pool.shutdown(wait=True)
    for w in workers:
        w.cancel()

    return accepted