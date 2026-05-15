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
import time
import pickle

sys.path.append('/root/shared-nvme/SKGF/DKGF-main/Self-Fusion-main')
try:
    from ThreadPoolExecutor import ThreadPoolExecutor
except ImportError:
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


def load_entity_names(file_path):
    """Load mapping of entity ids and names"""
    entity_names = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                entity_names[int(parts[0])] = parts[1]
    return entity_names


def load_relation_names(file_path):
    """Load mapping of relation ids and names"""
    relation_names = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                relation_names[int(parts[0])] = parts[1]
    return relation_names


def load_quadruples(file_path):
    """Load quadruple data"""
    quadruples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 5:
                h, r, t, t1, t2 = map(int, parts)
                quadruples.append((h, r, t, t1, t2))
    return quadruples


def group_quadruples(quadruples):
    """
    Group quadruples by relation_id, time_id1, time_id2, and either same head or tail entity
    Returns: dict with key (r, t1, t2, common_entity_id) and value list of quadruples
    """
    groups = defaultdict(list)

    # First group by (r, t1, t2)
    temp_groups = defaultdict(list)
    for quad in quadruples:
        h, r, t, t1, t2 = quad
        temp_groups[(r, t1, t2)].append(quad)

    # Then further group by common head or tail entity
    for (r, t1, t2), quads in temp_groups.items():
        # Group by head entity
        head_groups = defaultdict(list)
        tail_groups = defaultdict(list)

        for quad in quads:
            h, _, t, _, _ = quad
            head_groups[h].append(quad)
            tail_groups[t].append(quad)

        # Add groups with same head entity (size > 1)
        for h, head_quads in head_groups.items():
            if len(head_quads) > 1:
                groups[(r, t1, t2, f"head_{h}")].extend(head_quads)

        # Add groups with same tail entity (size > 1)
        for t, tail_quads in tail_groups.items():
            if len(tail_quads) > 1:
                groups[(r, t1, t2, f"tail_{t}")].extend(tail_quads)

    return groups


def load_progress(progress_file):
    """Load processing progress"""
    if os.path.exists(progress_file):
        with open(progress_file, 'rb') as f:
            return pickle.load(f)
    return {'processed_groups': set(), 'results': []}


def save_progress(progress_file, processed_groups, results):
    """Save processing progress"""
    progress_data = {
        'processed_groups': processed_groups,
        'results': results
    }
    with open(progress_file, 'wb') as f:
        pickle.dump(progress_data, f)


def save_incremental_result(result, json_file, tsv_file, lock):
    """Save single result incrementally"""
    with lock:
        # Append to JSON file (one result per line for easy parsing)
        with open(json_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

        # Append to TSV file if facts occurred
        if result['status'] == 'YES':
            with open(tsv_file, 'a', encoding='utf-8') as f:
                for quad in result['occurred_facts']:
                    h, r, t, t1, t2 = quad
                    # Escape description to handle newlines and tabs
                    description = result['description'].replace('\n', ' ').replace('\t', ' ')
                    f.write(f"{h}\t{r}\t{t}\t{t1}\t{t2}\t{description}\n")

def judge_quadruple_facts(data_dir, resume=True):
    """Deprecated thin wrapper. Kept for backward compatibility with the
    original main_DKGF.py loop. New code should call scene_generate() (paper
    §Fusion Scene Generation, Prompt 2)."""
    return scene_generate(data_dir, resume=resume, mode="legacy_judge")


# Prompt 2 of the paper: synthesize the input relations into a coherent,
# logically linked scientific description. The text must explicitly map
# topological dependencies to causal or conditional linguistic mechanisms.
# Ambiguous or disconnected graph components must be documented with explicit
# uncertainty markers in the text. See paper §Core Prompt Details / Prompt 2.
SCENE_GENERATION_PROMPT = """You are a scientific knowledge synthesizer. Translate the following set of fused facts into a coherent, logically linked scientific description.

Set of Fused Scientific Triples (F_new):
{facts}

Local Subgraph Context (G_local^s):
{context}

Negative Constraints (do NOT introduce these; they failed cycle consistency in prior iterations):
{negative_constraints}

Objective:
Synthesize the input relations into a coherent, logically linked scientific description. The text must explicitly map topological dependencies to causal or conditional linguistic mechanisms. Ambiguous or disconnected graph components must be documented with explicit uncertainty markers in the text.

Output: a single paragraph of natural-language scientific prose. Do not enumerate the facts; weave them into a coherent description.
"""


def _format_facts_for_scene(facts, ent_names, rel_names, time_ids):
    """Format (h, r, t, t1, t2) tuples into a textual list for Prompt 2."""
    lines = []
    for f in facts:
        if len(f) >= 5:
            h, r, t, t1, t2 = f[:5]
        else:
            h, r, t = f[:3]
            t1 = t2 = ""
        h_name = ent_names.get(h, f"Entity_{h}")
        r_name = rel_names.get(r, f"Relation_{r}")
        t_name = ent_names.get(t, f"Entity_{t}")
        time_str = ""
        if t1 and t2:
            t1s = time_ids.get(t1, f"Time_{t1}")
            t2s = time_ids.get(t2, f"Time_{t2}")
            time_str = f" during {t1s} to {t2s}"
        lines.append(f"- ({h_name}, {r_name}, {t_name}){time_str}")
    return "\n".join(lines) if lines else "(empty fact set)"


def scene_generate(
    data_dir,
    resume=True,
    f_new=None,
    g_local_context=None,
    negative_constraints=None,
    tau_ent=1.5,
    mode="scene",
):
    """
    Paper §Fusion Scene Generation (Prompt 2).

    Given the fused facts F_new and the local subgraph context G_local^s, the
    function asks the LLM to produce a coherent scientific description S_desc.
    Optionally also returns H_gen (generation entropy) computed from token
    log-probabilities.

    Args:
        data_dir: dataset directory (used to load entity/relation/time maps).
        resume:    when mode=='legacy_judge', reuses the old resume semantics.
        f_new:     optional list of accepted facts; when None, defaults to the
                   quadruples in `rank_temp_output_triples_fusion.txt` (the
                   legacy input contract).
        g_local_context: optional textual context for the local subgraph.
        negative_constraints: optional list of fact strings describing negative
                             constraints to inject into the prompt.
        tau_ent:   high-entropy threshold; facts whose generation entropy
                   exceeds this are flagged but not auto-discarded (we keep the
                   paper's open-world philosophy of returning F_accepted).
        mode:      'scene' invokes the paper Prompt 2; 'legacy_judge' falls
                   back to the original fact-level QC behaviour for backward
                   compatibility with the existing ablation flag.

    Returns:
        dict with keys:
            scene_text (str | None): generated scientific description.
            f_accepted (list): accepted facts (== f_new by default).
            h_gen (float): mean per-token generation entropy.
            h_gen_high (bool): whether H_gen > tau_ent.
            results: legacy per-group results when mode='legacy_judge'.
    """
    import httpx
    from openai import OpenAI

    if mode == "legacy_judge":
        return _legacy_judge_quadruple_facts(data_dir, resume=resume)

    ent_names_1 = load_entity_names(os.path.join(data_dir, 'ent_ids_1'))
    ent_names_2 = load_entity_names(os.path.join(data_dir, 'ent_ids_2'))
    rel_names_1 = load_relation_names(os.path.join(data_dir, 'rel_ids_1'))
    time_ids = load_time_ids(os.path.join(data_dir, 'time_id'))
    all_entity_names = {**ent_names_1, **ent_names_2}

    if f_new is None:
        # Default: load the legacy input file used by the original pipeline.
        rank_temp = os.path.join(data_dir, "message_pool", 'rank_temp_output_triples_fusion.txt')
        f_new = load_quadruples(rank_temp)

    if not f_new:
        return {"scene_text": "", "f_accepted": [], "h_gen": 0.0, "h_gen_high": False}

    facts_str = _format_facts_for_scene(f_new, all_entity_names, rel_names_1, time_ids)
    ctx_str = g_local_context or "(no explicit local context)"
    neg_str = "\n".join(negative_constraints) if negative_constraints else "(none)"

    prompt = SCENE_GENERATION_PROMPT.format(
        facts=facts_str,
        context=ctx_str,
        negative_constraints=neg_str,
    )

    client = OpenAI(
        base_url=os.environ.get("OPENAI_API_BASE", "yours"),
        api_key=os.environ.get("OPENAI_API_KEY", "yours"),
        http_client=httpx.Client(
            base_url=os.environ.get("OPENAI_API_BASE", "yours"),
            follow_redirects=True,
        ),
    )

    h_gen = 0.0
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            logprobs=True,
            top_logprobs=20,
        )
        scene_text = (response.choices[0].message.content or "").strip()
        try:
            import tokens_cal
            tokens_cal.update_add_var(response.usage.total_tokens)
        except Exception:
            pass
        # Compute generation entropy.
        try:
            from entropy import sequence_entropy_from_message_logprobs
            h_gen = sequence_entropy_from_message_logprobs(
                getattr(response.choices[0], "logprobs", None)
            )
        except Exception as e:
            print(f"[scene_generate] entropy computation failed: {e}")
    except Exception as e:
        print(f"[scene_generate] LLM call failed: {e}")
        scene_text = ""

    return {
        "scene_text": scene_text,
        "f_accepted": list(f_new),
        "h_gen": h_gen,
        "h_gen_high": h_gen > tau_ent,
    }


def _legacy_judge_quadruple_facts(data_dir, resume=True):
    """Original fact-level quality-control implementation preserved for the
    legacy `--wo-scene-generation` ablation path. New code should use
    `scene_generate` (paper §Fusion Scene Generation)."""

    # File paths for saving results
    json_output_file = os.path.join(data_dir, "message_pool", 'fact_verification_results.jsonl')
    tsv_output_file = os.path.join(data_dir, "message_pool", 'occurred_facts.txt')
    progress_file = os.path.join(data_dir, "message_pool", 'processing_progress.pkl')

    # Initialize TSV file with header if it doesn't exist
    os.makedirs(os.path.dirname(tsv_output_file), exist_ok=True)

    if not resume or not os.path.exists(tsv_output_file):
        with open(tsv_output_file, 'w', encoding='utf-8') as f:
            pass

    # Load progress if resuming
    processed_groups = set()
    all_results = []
    if resume:
        progress_data = load_progress(progress_file)
        processed_groups = progress_data['processed_groups']
        all_results = progress_data['results']
        print(f"Resuming: {len(processed_groups)} groups already processed")
    else:
        if os.path.exists(json_output_file):
            open(json_output_file, 'w').close()
            print("Cleared existing JSON output file")
        if os.path.exists(progress_file):
            os.remove(progress_file)
            print("Removed existing progress file")

    # Setting up the OpenAI client
    client = OpenAI(
        base_url="yours",
        api_key="yours",
        http_client=httpx.Client(
            base_url="yours",
            follow_redirects=True,
        ),
    )

    # Load all necessary data
    ent_names_1 = load_entity_names(os.path.join(data_dir, 'ent_ids_1'))
    ent_names_2 = load_entity_names(os.path.join(data_dir, 'ent_ids_2'))
    rel_names_1 = load_relation_names(os.path.join(data_dir, 'rel_ids_1'))
    time_ids = load_time_ids(os.path.join(data_dir, 'time_id'))

    # Combine entity names
    all_entity_names = {**ent_names_1, **ent_names_2}

    # Load quadruples - check for inconsistent file first
    inconsistent_file = os.path.join(data_dir, "message_pool", 'inconsistent_output_triples_fusion.txt')
    rank_temp_file = os.path.join(data_dir, "message_pool", 'rank_temp_output_triples_fusion.txt')

    # Determine which file to use
    if os.path.exists(inconsistent_file) and os.path.getsize(inconsistent_file) > 0:
        quadruples_file = inconsistent_file
        print("Using inconsistent_output_triples_fusion.txt (file exists and is not empty)")
    else:
        quadruples_file = rank_temp_file
        print("Using rank_temp_output_triples_fusion.txt (inconsistent file doesn't exist or is empty)")

    quadruples = load_quadruples(quadruples_file)

    # Group quadruples
    quad_groups = group_quadruples(quadruples)

    # Filter out already processed groups
    remaining_groups = {k: v for k, v in quad_groups.items() if k not in processed_groups}

    print(f"Total quadruples: {len(quadruples)}")
    print(f"Total groups: {len(quad_groups)}")
    print(f"Remaining groups to process: {len(remaining_groups)}")

    # Results storage
    lock = threading.Lock()
    result_queue = queue.Queue()
    save_counter = 0

    def process_group(group_key, group_quads):
        """Process a group of quadruples - KEEP MOST, REJECT ONLY OBVIOUSLY WRONG"""
        nonlocal save_counter
        try:
            r, t1, t2, common_entity = group_key

            # Get relation and time information
            relation_name = rel_names_1.get(r, f"Relation_{r}")
            time1 = time_ids.get(t1, f"Time_{t1}")
            time2 = time_ids.get(t2, f"Time_{t2}")

            # Build context for each quadruple in the group
            quad_contexts = []
            for i, (h, r_id, t, t1_id, t2_id) in enumerate(group_quads):
                head_name = all_entity_names.get(h, f"Entity_{h}")
                tail_name = all_entity_names.get(t, f"Entity_{t}")

                context = f"Fact {i + 1}: {head_name} --{relation_name}--> {tail_name} during {time1} to {time2}"
                quad_contexts.append({
                    'quadruple': (h, r_id, t, t1_id, t2_id),
                    'context': context,
                    'head_name': head_name,
                    'tail_name': tail_name
                })

            prompt = f"""You are a quality control expert reviewing knowledge graph facts. Your task is to identify ONLY the facts that are OBVIOUSLY WRONG or IMPOSSIBLE.

Relation Type: {relation_name}
Time Period: {time1} to {time2}
Common Entity: {common_entity.replace('head_', '').replace('tail_', '')}

Facts to review:
"""

            for quad_context in quad_contexts:
                prompt += f"- {quad_context['context']}\n"

            entity_names_list = ', '.join(
                set([qc['head_name'] for qc in quad_contexts] + [qc['tail_name'] for qc in quad_contexts]))

            prompt += f"""

REVIEW CRITERIA (BE LENIENT - ONLY REJECT CLEAR ERRORS):
Your job is NOT to verify accuracy, but to FILTER OUT obvious errors. Keep facts UNLESS they meet one of these criteria:

REJECT ONLY IF:
1. The relationship is logically impossible (e.g., "Moon visited Earth in 1800")
2. Entities are completely unrelated and couldn't possibly interact (e.g., "Shakespeare signed treaty with NASA")
3. Time period is impossibly wrong (e.g., "Ancient Rome used internet in 100 BC")
4. The fact contains obvious data corruption or nonsensical combinations

KEEP (Accept) IF:
- The fact is plausible, even if you can't verify exact details
- Entities could reasonably have this relationship
- Time period is approximately correct or unknown
- Minor inaccuracies in names or dates
- Uncertainty about exact historical details
- The relationship type makes sense for these entities

Response format:
If ALL facts are acceptable (keep them): "KEEP ALL - [reason why they're plausible]"
If SOME facts have obvious errors: "REJECT: [Entity names from {entity_names_list}] - [specific reason for rejection]"

Examples:
✓ "KEEP ALL - These political interactions are plausible for the given timeframe and entity types"
✓ "KEEP ALL - While I cannot verify exact details, the relationship type is reasonable for these entities"
✗ "REJECT: Entity_X and Entity_Y - These entities exist in different time periods and couldn't have interacted"
✗ "REJECT: Country_A and Organization_B - This relationship type is logically impossible for these entity types"

IMPORTANT: When in doubt, KEEP the facts. Only reject when you're confident they're obviously wrong.

Your response:"""

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{'role': 'user', 'content': prompt}],
                temperature=0.5
            )

            answer = response.choices[0].message.content.strip()
            print(prompt, answer)
            tokens_cal.update_add_var(response.usage.total_tokens)

            print(f"Processing group {group_key[:3]}... with {len(group_quads)} quadruples")
            print(f"Response: {answer[:100]}...")

            answer_upper = answer.upper()
            answer_lower = answer.lower()

            # Check if we should KEEP all facts (default behavior)
            should_keep_all = (
                    "KEEP ALL" in answer_upper or
                    "KEEP" in answer_upper[:50] or  # KEEP appears early in response
                    "ACCEPTABLE" in answer_upper or
                    "PLAUSIBLE" in answer_upper
            )

            if should_keep_all:
                # Keep all facts in this group
                result = {
                    'group_key': group_key,
                    'status': 'YES',
                    'occurred_facts': [quad_ctx['quadruple'] for quad_ctx in quad_contexts],
                    'description': answer,
                    'all_quadruples': [quad_ctx['quadruple'] for quad_ctx in quad_contexts],
                    'timestamp': time.time()
                }
            else:
                # Try to identify which facts to reject
                rejected_facts = []

                # Look for entity names mentioned with "REJECT"
                for quad_ctx in quad_contexts:
                    head_name = quad_ctx['head_name']
                    tail_name = quad_ctx['tail_name']

                    # Check if this fact is specifically mentioned with rejection
                    reject_found = False

                    # Look for entity names near "REJECT" keyword
                    if "REJECT" in answer_upper:
                        # Split into sections around REJECT
                        reject_sections = answer.split("REJECT")
                        for section in reject_sections[1:]:  # Skip first section (before REJECT)
                            section_lower = section[:200].lower()  # Check first 200 chars

                            head_parts = head_name.replace('_', ' ').lower().split()
                            tail_parts = tail_name.replace('_', ' ').lower().split()

                            # Check if either entity is mentioned in this reject section
                            head_match = any(part in section_lower for part in head_parts if len(part) > 3)
                            tail_match = any(part in section_lower for part in tail_parts if len(part) > 3)

                            if head_match or tail_match:
                                reject_found = True
                                break

                    if reject_found:
                        rejected_facts.append(quad_ctx['quadruple'])

                # Keep facts that weren't explicitly rejected
                occurred_facts = [
                    quad_ctx['quadruple'] for quad_ctx in quad_contexts
                    if quad_ctx['quadruple'] not in rejected_facts
                ]

                # If no specific rejections found but response says reject, keep all (conservative)
                if "REJECT" in answer_upper and not rejected_facts:
                    print(
                        f"WARNING: REJECT keyword found but couldn't identify specific facts for group {group_key[:3]}")
                    print(f"Answer: {answer[:200]}")
                    print("Keeping all facts by default (conservative approach)")
                    occurred_facts = [quad_ctx['quadruple'] for quad_ctx in quad_contexts]

                status = 'YES' if occurred_facts else 'NO'

                result = {
                    'group_key': group_key,
                    'status': status,
                    'occurred_facts': occurred_facts,
                    'description': answer,
                    'all_quadruples': [quad_ctx['quadruple'] for quad_ctx in quad_contexts],
                    'timestamp': time.time()
                }

            # Save result incrementally
            save_incremental_result(result, json_output_file, tsv_output_file, lock)

            # Update progress
            with lock:
                processed_groups.add(group_key)
                all_results.append(result)
                save_counter += 1

                # Save progress every 10 processed groups
                if save_counter % 10 == 0:
                    save_progress(progress_file, processed_groups, all_results)
                    print(f"Progress saved: {len(processed_groups)} groups completed")

        except Exception as e:
            print(f"Error processing group {group_key}: {str(e)}")
            # On error, keep all facts (conservative)
            result = {
                'group_key': group_key,
                'status': 'YES',
                'occurred_facts': [quad_ctx['quadruple'] for quad_ctx in quad_contexts],
                'description': f"Error occurred, keeping all facts: {str(e)}",
                'all_quadruples': [quad_ctx['quadruple'] for quad_ctx in quad_contexts],
                'timestamp': time.time()
            }
            save_incremental_result(result, json_output_file, tsv_output_file, lock)

            with open(os.path.join(data_dir, 'processing_errors.log'), 'a') as f:
                f.write(f"{time.time()}: Error processing {group_key}: {str(e)}\n")

    # Process groups using thread pool
    executor = ThreadPoolExecutor(max_workers=80)

    try:
        for group_key, group_quads in tqdm(remaining_groups.items()):
            executor.submit(process_group, group_key, group_quads)

        executor.shutdown(wait=True)

    except KeyboardInterrupt:
        print("\nInterrupted by user. Saving progress...")
        executor.shutdown(wait=False)

    # Final progress save
    save_progress(progress_file, processed_groups, all_results)

    # Convert JSONL to final JSON format
    final_json_file = os.path.join(data_dir, 'fact_verification_results.json')
    final_results = []
    if os.path.exists(json_output_file):
        with open(json_output_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    final_results.append(json.loads(line))

        with open(final_json_file, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, ensure_ascii=False, indent=2)

    # Print summary
    total_groups = len(processed_groups)
    occurred_groups = sum(1 for r in all_results if r['status'] == 'YES')
    total_occurred_facts = sum(len(r['occurred_facts']) for r in all_results if r['status'] == 'YES')

    print(f"\nFinal Summary:")
    print(f"Total groups processed: {total_groups}")
    print(f"Groups with occurred facts: {occurred_groups}")
    print(f"Total occurred facts: {total_occurred_facts}")
    print(f"Results saved to: {final_json_file}")
    print(f"Occurred facts saved to: {tsv_output_file}")
    print(f"Progress file: {progress_file}")

    return all_results



if __name__ == "__main__":
    data_dir = "/home/dex/Desktop/SKGF/TEA-RAG/dataset/wiki_for_icews_0.8_3.7_TF"
    results = judge_quadruple_facts(data_dir, resume=False)
