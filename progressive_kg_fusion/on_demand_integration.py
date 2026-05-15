import os
import queue
import threading
import re
from tqdm import tqdm
import httpx
from openai import OpenAI
from collections import defaultdict
import json
import random
import sys

sys.path.append('/root/shared-nvme/SKGF/DKGF-main/Self-Fusion-main')
try:
    from ThreadPoolExecutor import ThreadPoolExecutor
except ImportError:
    # Fallback: use the local trivial subclass shipped under thread_util/.
    import os as _os, sys as _sys
    _here = _os.path.dirname(_os.path.abspath(__file__))
    _pkg_root = _os.path.dirname(_here)
    if _pkg_root not in _sys.path:
        _sys.path.insert(0, _pkg_root)
    from thread_util.ThreadPoolExecutor import ThreadPoolExecutor


import tokens_cal

def load_time_ids(file_path):
    """Load mapping of time ids and timestamps"""
    time_ids = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                tid, timestamp = parts
                time_ids[int(tid)] = timestamp
    return time_ids


def load_descriptions(file_path):
    """Load entity description information"""
    with open(file_path, 'r', encoding='utf-8') as f:
        descriptions = json.load(f)
    return {int(k): v[0]['description'] if v else "" for k, v in descriptions.items()}

def get_entity_context_m3(entity_id, entity_names, descriptions):
    """Get descriptive information about the entity (used for from_m3=True)"""
    return f"Entity Name: {entity_names.get(entity_id, 'Unknown')}\nDescription:\n{descriptions.get(entity_id, 'No description available')}"

def get_random_rules(data_dir, n=5):
    """Randomly select n rules from the rules file"""
    LLM1_PRIVATE_MESSAGE_POOL = {
        'top_k_candidate_entities': os.path.join(data_dir, "message_pool", "uni_results.txt"),
        'ucon_similarity_results': os.path.join(data_dir, "message_pool", "ucon_similarity_results.txt"),
        'KG1_compared_description': os.path.join(data_dir, "message_pool", "KG1_compared_description.json"),
        'KG2_compared_description': os.path.join(data_dir, "message_pool", "KG2_compared_description.json"),
        'alignment_rules': os.path.join(data_dir, "message_pool", "alignment_rules.txt"),
        'aligned_entities': os.path.join(data_dir, "message_pool", "relative_entities_1.txt"),
        'merge_retriever': os.path.join(data_dir, "message_pool", "merge_retriever.txt"),
        'conflict_results': os.path.join(data_dir, "message_pool", "conflict_results.txt"),
    }
    rules = []
    rules_file = LLM1_PRIVATE_MESSAGE_POOL['alignment_rules']
    if os.path.exists(rules_file):
        with open(rules_file, 'r', encoding='utf-8') as f:
            all_rules = f.readlines()
        rules = random.sample(all_rules, min(n, len(all_rules)))
    return rules


def load_entity_names(file_path):
    """Load mapping of entity ids and names"""
    entity_names = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                entity_names[int(parts[0])] = parts[1]
    return entity_names

def load_triples(file_path):
    """Load ternary data"""
    triples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            triples.append([int(x) for x in parts[:3]])
    return triples

def get_entity_context(entity_id, entity_names_1, entity_names_2, triples, rel_names_1, rel_names_2, n=30):
    """Get the first n relationships of the entity"""
    relations = []
    for h, r, t in triples:
        if h == entity_id:
            rel_str = rel_names_1.get(r) or rel_names_2.get(r, str(r))
            tail_str = entity_names_1.get(t) or entity_names_2.get(t, str(t))
            relations.append(f"- Has relation '{rel_str}' with {tail_str} (ID: {t})")
        elif t == entity_id:
            rel_str = rel_names_1.get(r) or rel_names_2.get(r, str(r))
            head_str = entity_names_1.get(h) or entity_names_2.get(h, str(h))
            relations.append(f"- Is {rel_str} of {head_str} (ID: {h})")
        if len(relations) >= n:
            break

    entity_name = entity_names_1.get(entity_id) or entity_names_2.get(entity_id, 'Unknown')
    context = f"Entity Name: {entity_name} (ID: {entity_id})\n"
    context += "Relationships:\n" + "\n".join(relations[:n])
    return context



def group_candidates(input_file):
    """Group candidate entity pairs in the input file by KG1 entities

    Supports two formats:
    Format 1: e1\te2 (two columns)
    Format 2: e1\t...\te2\t...\t... (five columns, using column 1 and 3)
    """
    groups = defaultdict(list)
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')

            if len(parts) == 2:
                # Original format: e1\te2
                e1, e2 = map(int, parts)
            elif len(parts) == 5:
                # New format: column 1 and column 3
                e1 = int(parts[0])
                e2 = int(parts[2])
            else:
                # Skip invalid lines
                continue

            groups[e1].append(e2)

    return groups


def integration(data_dir, from_m3 = False):

    LLM1_PRIVATE_MESSAGE_POOL = {
        'top_k_candidate_entities': os.path.join(data_dir, "message_pool", "entity_retriever_outputs.txt"),
        'ucon_similarity_results': os.path.join(data_dir, "message_pool", "ucon_similarity_results.txt"),
        'KG1_compared_description': os.path.join(data_dir, "message_pool", "KG1_compared_description.json"),
        'KG2_compared_description': os.path.join(data_dir, "message_pool", "KG2_compared_description.json"),
        'alignment_rules': os.path.join(data_dir, "message_pool", "alignment_rules.txt"),
        'relevance_entities': os.path.join(data_dir, "message_pool", "relevance_entities.txt"),
        'merge_retriever': os.path.join(data_dir, "message_pool", "merge_retriever.txt"),
        'conflict_results': os.path.join(data_dir, "message_pool", "conflict_results.txt"),
    }


    LLM1_Agent_Profile = '''
Goal: As a Knowledge Graph relevance recognition expert, judge whether the following entity 1 (and its neighbors) has potential real-world correlations with the candidate entities (and their neighbors).
Constraint: if potential real-world correlations exists, only candidate entities (and their neighbors) ID is returned; if none of them are relevant (and there is no connectable relational intersection), return ‘No’; if none of them are relevant (and there is no connectable relational intersection), return ‘No’;if none of them are relevant (and there is no connectable relational intersection), return ‘No’.
    '''



    input_file = LLM1_PRIVATE_MESSAGE_POOL[
            'top_k_candidate_entities']
    output_file = LLM1_PRIVATE_MESSAGE_POOL['relevance_entities']




    # Setting up the OpenAI client
    client = OpenAI(
        base_url="yours",
        api_key="yours",
        http_client=httpx.Client(
            base_url="yours",
            follow_redirects=True,
        ),
    )


    ent_names_1 = load_entity_names(os.path.join(data_dir, 'ent_ids_1'))
    ent_names_2 = load_entity_names(os.path.join(data_dir, 'ent_ids_2'))


    rel_names_1 = load_entity_names(os.path.join(data_dir, 'rel_ids_1'))
    rel_names_2 = load_entity_names(os.path.join(data_dir, 'rel_ids_2'))
    triples_1 = load_triples(os.path.join(data_dir, 'triples_1'))
    triples_2 = load_triples(os.path.join(data_dir, 'triples_2'))


    time_ids = load_time_ids(os.path.join(data_dir, 'time_id'))



    # Grouping of candidate entities
    candidate_groups = group_candidates(input_file)
    aligned_pairs = []

    lock = threading.Lock()
    executor = ThreadPoolExecutor(max_workers=30)
    # Create a queue to store writes to the file
    result_queue = queue.Queue()

    def openai_task(kg1_entity, kg2_candidates):
        try:

            context1 = get_entity_context(kg1_entity, ent_names_1, ent_names_2, triples_1, rel_names_1, rel_names_2)

            # Get the context of all KG2 candidate entities
            candidates_contexts = []
            for kg2_entity in kg2_candidates:
                context = get_entity_context(kg2_entity, ent_names_2, ent_names_1, triples_2, rel_names_2, rel_names_1)
                candidates_contexts.append({
                    'entity_id': kg2_entity,
                    'context': context
                })


            extended_candidates = set(kg2_candidates)
            for kg2_entity in kg2_candidates:
                for h, r, t in triples_2:
                    if h == kg2_entity:
                        extended_candidates.add(t)
                    elif t == kg2_entity:
                        extended_candidates.add(h)

            # Building the Prompt

            prompt = LLM1_Agent_Profile + f"""
                                Entity 1 (ID: {kg1_entity}):
                                {context1}

                                the candidate entity list:"""



            for i, candidate in enumerate(candidates_contexts, 1):
                prompt += f"\n\ncandidate entity{i} (ID: {candidate['entity_id']}):\n{candidate['context']}"

            # prompt += "\nPlease select a relationship and a timestamp for the new quadruple."
            # prompt += "\nRelations to choose from:\n" + \
            #           "\n".join(f"{name} (ID: {rid})" for rid, name in sorted(rel_names_1.items()))

            #prompt += "\nTimestamps to choose from: [1998-01, 2021-12]\n"

            prompt += """\n\nAre there any of these entities that might be associated with entity 1?  If so, only candidate entities (and their neighbors) ID is returned; if none of them are relevant (and there is no connectable relational intersection), return ‘No’; if none of them are relevant (and there is no connectable relational intersection), return ‘No’; if none of them are relevant (and there is no connectable relational intersection), return ‘No’:\n Answer:\n\n"""

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{'role': 'user', 'content': prompt}]
            )

            answer = response.choices[0].message.content.strip()

            tokens_cal.update_add_var(response.usage.total_tokens)  # update tokens

            # The kg1_entity and kg2_candidates are printed here for each execution.
            print(f"Processing entity {kg1_entity} with candidates {kg2_candidates}")

            print(prompt, answer)
            # an analytic response
            if answer.lower() != "no":
                # Trying to extract the entity ID from the answer
                for kg2_id in extended_candidates:
                    if re.search(rf'\b{kg2_id}\b', answer):
                        with lock:
                            result_queue.put((kg1_entity, kg2_id))
                            aligned_pairs.append((kg1_entity, kg2_id))


        except Exception as e:
            print(f"Error processing entity {kg1_entity}: {str(e)}")



    # Processing each group of candidate entities
    for kg1_entity_c, kg2_candidates_c in tqdm(candidate_groups.items()):
        executor.submit(openai_task,kg1_entity_c, kg2_candidates_c)




    # Save results
    # if os.path.exists(output_file):
    #     with open(output_file, 'a', encoding='utf-8') as f:
    #         for e1, e2 in aligned_pairs:
    #             f.write(f"{e1}\t{e2}\n")
    # else:
    #     with open(output_file, 'w', encoding='utf-8') as f:
    #         for e1, e2 in aligned_pairs:
    #             f.write(f"{e1}\t{e2}\n")

    executor.shutdown(wait=True)

    # Write the results from the queue to a file
    with open(output_file, 'a+', encoding='utf-8') as output_f:
        while not result_queue.empty():
            kg1_entity, kg2_id = result_queue.get()
            output_f.write(f"{kg1_entity}\t{kg2_id}\n")
            output_f.flush()  # Flush the buffer immediately to ensure that it is written to disk

    deduplicate_output_file(output_file)

    return aligned_pairs

def deduplicate_output_file(file_path):
    """De-duplication of the output file"""
    if not os.path.exists(file_path):
        return

    # Reads all rows and de-duplicates them
    unique_pairs = set()
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            e1, e2 = map(int, line.strip().split('\t'))
            unique_pairs.add((e1, e2))

    # Rewrite the result after de-duplication
    with open(file_path, 'w', encoding='utf-8') as f:
        for e1, e2 in sorted(unique_pairs):  # Sorting to maintain stable output
            f.write(f"{e1}\t{e2}\n")

    print(f"Deduplicated file {file_path}: {len(unique_pairs)} unique pairs")

if __name__ == "__main__":
    data_dir = "/root/shared-nvme/SKGF/DKGF-main/SKGF-main/dataset/wiki_for_icews_0.8_3.7_TF"
    aligned_pairs = integration(data_dir)
    print(f"Found {len(aligned_pairs)} aligned entity pairs.")


# =============================================================================
# Paper §Scene-aware On-demand Integration (Eq. F_new) — Prompt 1 + filter
# =============================================================================
# The pipeline above uses a legacy "Are there any of these entities..."
# prompt for entity-level relevance. The paper's Eq. F_new instead requires
# a per-fact validity filter driven by token log-probabilities:
#
#     F_new = { (v, r, v̂) in C^g_meta(u_i^s) | P(f | G_local^s) >= tau, r in R^s }
#
# This block adds:
#   - PROMPT_1_VALIDITY    : the canonical Prompt 1 from §Core Prompt Details.
#   - _build_fact_line     : renders a (h,r,t,t1,t2) into a single bullet line.
#   - _build_batch_prompt  : renders N facts into a single Prompt 1 instance.
#   - _parse_yes_no        : parses a sequence of Yes/No tokens (best-effort).
#   - filter_facts_by_tau_batched : amortized Eq. F_new filter using log-probs.
#
# The batched path issues one LLM call per (subgraph, batch) and recovers
# per-fact probabilities through top_logprobs on the first generated token
# of each Yes/No answer. For tiny batches (< 8 facts) we transparently fall
# back to the existing per-fact logprob_yes_no to keep calibration tight.
# =============================================================================


PROMPT_1_VALIDITY = """[System Input]
Target Domain: Scientific Knowledge Fusion
Local Reference Structure (G_local^s): {local_skg}

Candidate Facts (f):
{facts}

[Negative Constraints - facts that previously FAILED cycle-consistency and MUST be rejected if proposed again]
{negative_constraints}

[Task Evaluation Criteria]
Evaluate the integration suitability of EACH candidate fact based on:
1. Granularity Alignment: Verify if the entity-relation abstractions match
   the specific operational resolution of the reference structure.
2. Logical Consistency: Check for structural or factual contradictions with
   established local mechanisms.

[Output Format]
For EACH fact, on its own line, respond with a single token: "Yes" or "No".
Do not add numbering, explanations, or punctuation. Order matters.
"""


def _build_fact_line(fact, ent_names, rel_names, time_ids):
    """Render a (h, r, t, t1, t2) tuple into a single human-readable line."""
    if len(fact) >= 5:
        h, r, t, t1, t2 = fact[:5]
    else:
        h, r, t = fact[:3]
        t1 = t2 = ""
    h_name = ent_names.get(h, f"Entity_{h}")
    r_name = rel_names.get(r, f"Relation_{r}")
    t_name = ent_names.get(t, f"Entity_{t}")
    time_str = ""
    if t1 != "" and t2 != "":
        t1s = time_ids.get(t1, f"Time_{t1}")
        t2s = time_ids.get(t2, f"Time_{t2}")
        time_str = f" during {t1s} to {t2s}"
    return f"- ({h_name}, {r_name}, {t_name}){time_str}"


def _build_batch_prompt(local_skg_text, fact_lines, negative_constraints=None):
    """Build a single Prompt 1 instance covering all fact_lines.

    Paper Algorithm 1 Step 2.1 evaluates P(f | G_local^s, I_neg). The
    negative_constraints (I_neg) are injected into Prompt 1 so the LLM
    can refuse any fact that previously failed cycle-consistency.
    """
    neg_str = "\n".join(negative_constraints) if negative_constraints else "(none)"
    return PROMPT_1_VALIDITY.format(
        local_skg=local_skg_text or "(no explicit local context)",
        facts="\n".join(fact_lines) if fact_lines else "(empty)",
        negative_constraints=neg_str,
    )


def _parse_yes_no_sequence(text, expected):
    """Parse a sequence of Yes/No tokens; pad/truncate to length=expected.

    Returns a list of booleans of length expected.
    """
    import re as _re
    tokens = _re.findall(r"\b(yes|no)\b", (text or "").lower())
    out = []
    for i in range(expected):
        if i < len(tokens):
            out.append(tokens[i] == "yes")
        else:
            # Conservative default when the model truncates: refuse the fact.
            out.append(False)
    return out


def filter_facts_by_tau_batched(
    candidate_facts,
    local_skg_text,
    tau,
    client,
    ent_names=None,
    rel_names=None,
    time_ids=None,
    negative_constraints=None,
    batch_size=20,
    fallback_below=8,
    model="gpt-4o-mini",
    use_logprob_fallback=True,
):
    """Batched implementation of Eq. F_new.

    Args:
        candidate_facts : iterable of (h, r, t, t1, t2) tuples.
        local_skg_text  : already-rendered Local Subgraph context for Prompt 1.
        tau             : the entropy-bound threshold; facts with P>=tau pass.
        client          : an OpenAI client (any object compatible with the
                          chat.completions interface).
        ent_names, rel_names, time_ids : optional name maps used only for
                          rendering the fact lines; fall back to IDs otherwise.
        negative_constraints : optional list of strings (I_neg from previous
                          cycle-consistency iterations). Injected into Prompt 1
                          so the LLM refuses any fact that previously failed
                          the round-trip. None or empty disables the field.
        batch_size      : number of facts per LLM call.
        fallback_below  : if len(candidate_facts) < this, fall back to the
                          per-fact logprob_yes_no path for calibration.
        model           : chat model name.
        use_logprob_fallback : if True, fall back to textual Yes/No parsing
                          when top_logprobs are unavailable.

    Returns:
        (accepted_facts, p_per_fact) where both are lists of length
        len(candidate_facts). p_per_fact[i] is the softmax-normalized
        probability of "Yes" for candidate_facts[i].
    """
    candidate_facts = list(candidate_facts)
    if not candidate_facts:
        return [], []

    ent_names = ent_names or {}
    rel_names = rel_names or {}
    time_ids = time_ids or {}
    negative_constraints = negative_constraints or []

    # Small batches: use the precise per-fact path to keep calibration tight.
    if len(candidate_facts) < fallback_below:
        from progressive_kg_fusion.logprob import filter_facts_by_tau
        def _build_prompt_with_neg(f):
            return PROMPT_1_VALIDITY.format(
                local_skg=local_skg_text or "(no explicit local context)",
                facts=_build_fact_line(f, ent_names, rel_names, time_ids),
                negative_constraints="\n".join(negative_constraints)
                if negative_constraints else "(none)",
            )
        accepted = filter_facts_by_tau(
            facts=candidate_facts,
            client=client,
            build_prompt=_build_prompt_with_neg,
            taus={"__default__": tau},
            model=model,
            concurrency=8,
        )
        accepted_set = set(tuple(f[:5]) for f in accepted)
        p_per_fact = [1.0 if tuple(f[:5]) in accepted_set else 0.0
                      for f in candidate_facts]
        return accepted, p_per_fact

    # Batched path.
    fact_lines = [_build_fact_line(f, ent_names, rel_names, time_ids)
                  for f in candidate_facts]
    p_per_fact = [0.0] * len(candidate_facts)

    for start in range(0, len(candidate_facts), batch_size):
        end = min(start + batch_size, len(candidate_facts))
        batch_facts = candidate_facts[start:end]
        batch_lines = fact_lines[start:end]
        prompt = _build_batch_prompt(local_skg_text, batch_lines, negative_constraints)

        yes_probs = None
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                logprobs=True,
                top_logprobs=20,
            )
            try:
                import tokens_cal as _tc
                if getattr(response, "usage", None) is not None:
                    _tc.update_add_var(response.usage.total_tokens)
            except Exception:
                pass

            content = (response.choices[0].message.content or "").strip()
            top_lp = getattr(response.choices[0], "logprobs", None)

            # Reconstruct per-position Yes/No probabilities from top_logprobs.
            per_pos = []
            if top_lp is not None and getattr(top_lp, "content", None):
                for entry in top_lp.content:
                    yes_lp = no_lp = None
                    top = getattr(entry, "top_logprobs", None) or []
                    for cand in top:
                        tok = getattr(cand, "token", "").strip().lower()
                        lp = getattr(cand, "logprob", 0.0)
                        if tok == "yes" and yes_lp is None:
                            yes_lp = lp
                        elif tok == "no" and no_lp is None:
                            no_lp = lp
                    if yes_lp is not None and no_lp is not None:
                        import math as _m
                        m = max(yes_lp, no_lp)
                        ey = _m.exp(yes_lp - m)
                        en = _m.exp(no_lp - m)
                        per_pos.append(ey / (ey + en))
                    else:
                        per_pos.append(None)

            if per_pos and all(p is not None for p in per_pos):
                # Align length with batch size; truncate or pad neutrally.
                yes_probs = []
                for i in range(len(batch_facts)):
                    yes_probs.append(per_pos[i] if i < len(per_pos) else 0.0)

            if yes_probs is None and use_logprob_fallback:
                bools = _parse_yes_no_sequence(content, len(batch_facts))
                yes_probs = [1.0 if b else 0.0 for b in bools]
        except Exception as e:
            print(f"[filter_facts_by_tau_batched] LLM call failed: {e}")
            if not use_logprob_fallback:
                raise

        if yes_probs is None:
            # Conservative default: refuse all in this batch.
            yes_probs = [0.0] * len(batch_facts)

        for i, p in enumerate(yes_probs):
            p_per_fact[start + i] = float(p)

    accepted = [f for f, p in zip(candidate_facts, p_per_fact) if p >= tau]
    return accepted, p_per_fact