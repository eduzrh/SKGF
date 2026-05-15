"""
Entropy computations for the progressive self-feedback loop (paper
§Entropy-driven Validity Estimation via Token Probability and §Fusion Scene
Graph Reconstruction).

Two notions of entropy are defined:

1. Generation entropy H_gen(y | F_new):
   the mean per-token entropy of the generated scene description y. Lower
   H_gen indicates that the fused facts F_new induce a coherent, deterministic
   description.

       H_gen(y | F_new) = -(1/T) * sum_t sum_w p(w | y_{<t}, F_new) log p(w | ...)
       (see paper Eq. after token_entropy §)

2. Cycle-consistency entropy H_cycle(F_new):
   the empirical log-loss between F_new and the reconstructed F_recon after a
   graph-to-text-to-graph round trip. H_cycle -> 0 indicates lossless recovery.

       H_cycle(F_new) ~= - sum_{f in F_new} log P(f in Recon(S_desc))
       (paper Eq. eq:cycle_entropy)

"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# Numerical guards.
_EPS = 1e-12


def _safe_log(p: float) -> float:
    return math.log(max(p, _EPS))


def sequence_entropy_from_top_logprobs(top_logprobs_per_step: Sequence[Sequence[Dict[str, Any]]]) -> float:
    """
    Compute H_gen from the per-step top_logprobs returned by the OpenAI chat
    completion API when `logprobs=True` and `top_logprobs=K` are requested.

    Each step is a sequence of dicts; each dict maps a token string to a payload
    that exposes a `logprob` field. We reconstruct the probability distribution
    over the *restricted* top-K set at each step and renormalize it. This is
    the standard Monte-Carlo estimator for token-level entropy when only the
    top-K mass is available.

    Args:
        top_logprobs_per_step: list (length T) of per-step lists of dicts.

    Returns:
        The mean per-token entropy in nats. Returns 0.0 if the payload is empty.
    """
    if not top_logprobs_per_step:
        return 0.0

    entropies: List[float] = []
    for step in top_logprobs_per_step:
        if not step:
            continue
        probs: List[float] = []
        for entry in step:
            if isinstance(entry, dict):
                # Try a few common shapes for top_logprobs payloads:
                #   1) {token: {"logprob": lp, ...}}   (OpenAI Chat API, this module's writer)
                #   2) {token: lp}                     (test-friendly flat shape)
                #   3) {"token": ..., "logprob": lp}   (raw OpenAI shape)
                if "logprob" in entry:
                    probs.append(math.exp(float(entry["logprob"])))
                else:
                    for v in entry.values():
                        lp = v["logprob"] if isinstance(v, dict) else v
                        probs.append(math.exp(float(lp)))
            else:
                probs.append(math.exp(float(entry)))
        if not probs:
            continue
        z = sum(probs)
        if z <= 0:
            continue
        probs = [p / z for p in probs]
        h = -sum(p * _safe_log(p) for p in probs)
        entropies.append(h)

    if not entropies:
        return 0.0
    return sum(entropies) / len(entropies)


def sequence_entropy_from_message_logprobs(message_logprobs: Any) -> float:
    """
    Convenience wrapper that accepts an OpenAI `choice.logprobs` object whose
    `content` attribute is a list of per-token entries. Returns the mean
    per-token entropy in nats.
    """
    if message_logprobs is None:
        return 0.0
    content = getattr(message_logprobs, "content", None) or message_logprobs
    per_step: List[List[Dict[str, Any]]] = []
    for entry in content or []:
        # Each entry has a `top_logprobs` attribute that is a list of LogprobInfo
        # objects. We convert each to a dict {"token": logprob}.
        top = getattr(entry, "top_logprobs", None) or []
        per_step.append([{getattr(t, "token", ""): getattr(t, "logprob", 0.0)} for t in top])
    return sequence_entropy_from_top_logprobs(per_step)


def cycle_entropy(f_new: Iterable[Tuple[Any, ...]], f_recon: Iterable[Tuple[Any, ...]]) -> float:
    """
    Compute the cycle-consistency entropy (paper Eq. eq:cycle_entropy):

        H_cycle(F_new) ~= - sum_{f in F_new} log P(f in Recon(S_desc))

    We model P(f in Recon) as an empirical indicator smoothed by a small
    epsilon: P(f in Recon) = 1 if f is in F_recon, otherwise P(f in Recon) =
    epsilon. This is a strict upper bound that yields H_cycle -> 0 iff every
    fact in F_new is recovered by Recon. We use nats throughout for consistency
    with H_gen.

    Args:
        f_new: iterable of candidate facts accepted by the entropy filter.
        f_recon: iterable of facts reconstructed from the generated scene text.

    Returns:
        The total cycle-consistency entropy. Returns 0.0 on empty input.
    """
    f_new = list(f_new)
    if not f_new:
        return 0.0
    f_recon_set = set(f_recon)
    h = 0.0
    for f in f_new:
        # The canonical tuple form is (h, r, t) for the relation, or
        # (h, r, t, t1, t2) for time-aware quadruples. We compare on the first
        # three fields (the relation) which are the deterministic scope of
        # the graph-to-text reconstruction (Prompt 3 forbids time/relational
        # extrapolation beyond the literal text).
        head = (f[0], f[1], f[2]) if len(f) >= 3 else tuple(f)
        if head in {(g[0], g[1], g[2]) for g in f_recon_set}:
            h += 0.0
        else:
            h += -_safe_log(_EPS)
    return h


def is_high_entropy_text(h_gen: float, tau_ent: float) -> bool:
    """Convenience predicate: flag high-entropy noise per paper §token_entropy."""
    return h_gen > tau_ent