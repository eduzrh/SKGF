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
            if len(parts) >= 2:
                entity_names[int(parts[0])] = '\t'.join(parts[1:])
    return entity_names


def load_relation_names(file_path):
    """Load mapping of relation ids and names"""
    relation_names = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                relation_names[int(parts[0])] = '\t'.join(parts[1:])
    return relation_names


def load_occurred_facts(file_path):
    """Load occurred facts data"""
    facts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) >= 6:
                h, r, t, t1, t2 = map(int, parts[:5])
                description = '\t'.join(parts[5:])
                facts.append((h, r, t, t1, t2, description))
            elif len(parts) == 5:
                h, r, t, t1, t2 = map(int, parts)
                facts.append((h, r, t, t1, t2, ""))
    return facts


def batch_facts_for_verification(facts, batch_size=5):
    """Batch facts for efficient processing"""
    batches = []
    for i in range(0, len(facts), batch_size):
        batch = facts[i:i + batch_size]
        batches.append(batch)
    return batches


def save_results(consistent_facts, inconsistent_facts, data_dir):
    """Save consistent and inconsistent facts to separate files"""

    # Define file paths
    message_pool_dir = os.path.join(data_dir, "message_pool")
    enhanced_file = os.path.join(message_pool_dir, 'enhanced_structure_output.txt')
    consistent_file = os.path.join(message_pool_dir, 'output_triples_fusion.txt')

    # Create message_pool directory if it doesn't exist
    os.makedirs(message_pool_dir, exist_ok=True)

    # Save consistent facts to output_triples_fusion.txt
    if os.path.exists(enhanced_file):
        # If enhanced_structure_output.txt exists, use it as base
        print(f"Using {enhanced_file} as base for output_triples_fusion.txt")
        try:
            # Copy content from enhanced_file to consistent_file
            with open(enhanced_file, 'r', encoding='utf-8') as src:
                enhanced_content = src.read()

            with open(consistent_file, 'w', encoding='utf-8') as dst:
                dst.write(enhanced_content)

            # Append new consistent facts
            with open(consistent_file, 'a', encoding='utf-8') as f:
                for h, r, t, t1, t2 in consistent_facts:
                    f.write(f"{h}\t{r}\t{t}\t{t1}\t{t2}\n")

            print(f"Consistent facts saved to: {consistent_file} ({len(consistent_facts)} facts)")

        except Exception as e:
            print(f"Error processing enhanced file: {e}")
            # Fallback to original logic
            with open(consistent_file, 'a', encoding='utf-8') as f:
                for h, r, t, t1, t2 in consistent_facts:
                    f.write(f"{h}\t{r}\t{t}\t{t1}\t{t2}\n")

    else:
        # Original logic: append to existing file or create new one
        print(f"Enhanced file not found, using original logic")
        mode = 'a' if os.path.exists(consistent_file) else 'w'

        with open(consistent_file, mode, encoding='utf-8') as f:
            for h, r, t, t1, t2 in consistent_facts:
                f.write(f"{h}\t{r}\t{t}\t{t1}\t{t2}\n")

        print(f"Saved {len(consistent_facts)} facts to {consistent_file}")

    # Save inconsistent facts (optional, based on your original function parameters)
    if inconsistent_facts:
        inconsistent_file = os.path.join(message_pool_dir, 'inconsistent_facts.txt')
        with open(inconsistent_file, 'a', encoding='utf-8') as f:
            for fact in inconsistent_facts:
                f.write(f"{fact}\n")
        print(f"Saved {len(inconsistent_facts)} inconsistent facts to {inconsistent_file}")

def load_progress(progress_file):
    """Load processing progress"""
    if os.path.exists(progress_file):
        with open(progress_file, 'rb') as f:
            return pickle.load(f)
    return {
        'processed_batches': 0,
        'consistent_facts': [],
        'inconsistent_facts': []
    }


def save_progress(progress_file, processed_batches, consistent_facts, inconsistent_facts):
    """Save processing progress"""
    progress_data = {
        'processed_batches': processed_batches,
        'consistent_facts': consistent_facts,
        'inconsistent_facts': inconsistent_facts
    }
    with open(progress_file, 'wb') as f:
        pickle.dump(progress_data, f)

# Prompt 3 of the paper: extract all deterministic factual assertions from
# the generated scene text and emit relational triples. No external relational
# extrapolation or logical inference beyond the literal scope of the text is
# permitted. See paper §Core Prompt Details / Prompt 3.
SCENE_RECONSTRUCTION_PROMPT = """You are a precise scientific fact extractor. Extract all deterministic factual assertions from the following scientific scene text and emit them as relational triples.

Generated Target Scientific Text (S_desc):
{scene_text}

Objective:
Extract all deterministic factual assertions from the text. Outputs must be formatted exclusively as relational triples (Head, Relation, Tail). No external relational extrapolation or logical inference beyond the literal scope of the text is permitted. If a sentence expresses uncertainty or hedges, do NOT extract it as a fact.

Output format (one triple per line, NO numbering, NO explanations):
Head1|Relation1|Tail1
Head2|Relation2|Tail2
...
"""


def _parse_triples_from_text(answer: str, ent_names, rel_names):
    """Parse the LLM answer into a list of (head_name, rel_name, tail_name) tuples.

    Accepts the canonical `Head|Relation|Tail` format. Falls back to a relaxed
    parsing strategy (splitting on tab, ' -- ', '->', etc.) for robustness.
    Returns tuples of (h_id_or_name, r_id_or_name, t_id_or_name); we leave the
    caller to map names back to IDs (Prompt 3 only constrains textual scope).
    """
    import re

    triples = []
    for raw_line in answer.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        # Strip leading bullets/numbering.
        line = re.sub(r"^[\-\*\u2022]\s*", "", line)
        line = re.sub(r"^\d+[\.\)]\s*", "", line)
        # Try canonical pipe-separated form.
        if "|" in line:
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 3:
                triples.append((parts[0], parts[1], parts[2]))
                continue
        # Try arrow form: "Head -- Relation --> Tail" or "Head -> Relation -> Tail".
        for sep in [" -- ", "->", " → ", " --> "]:
            if sep in line:
                parts = [p.strip() for p in line.split(sep)]
                if len(parts) >= 3:
                    triples.append((parts[0], parts[1], parts[2]))
                    break
        else:
            # Last-resort: split on whitespace if we have at least three tokens.
            toks = line.split()
            if len(toks) >= 3:
                triples.append((toks[0], toks[1], " ".join(toks[2:])))
    return triples


def reconstruct_facts(scene_text, client, ent_names=None, rel_names=None, f_new=None, model="gpt-4o-mini"):
    """
    Paper §Fusion Scene Graph Reconstruction (Prompt 3).

    Implements Recon(S_desc): extract a validation set F_recon of relational
    triples from the generated scientific scene description. When `f_new` is
    supplied, we also compute H_cycle(F_new, F_recon) (paper Eq. eq:cycle_entropy)
    directly against the caller-supplied F_new.

    Args:
        scene_text: the S_desc produced by scene_generate().
        client    : an OpenAI-compatible chat client.
        ent_names : optional {id -> name} map; used to render H_cycle names.
        rel_names : optional {id -> name} map; used to render H_cycle names.
        f_new     : optional iterable of facts (h, r, t[, t1, t2]) accepted by
                    the entropy filter. When provided, H_cycle is computed
                    against F_recon. When None, h_cycle returns 0.0.
        model     : chat model name.
    """
    if not scene_text:
        return {
            "f_recon": [],
            "f_recon_names": [],
            "h_cycle": 0.0,
        }

    prompt = SCENE_RECONSTRUCTION_PROMPT.format(scene_text=scene_text)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        answer = (response.choices[0].message.content or "").strip()
        try:
            import tokens_cal
            tokens_cal.update_add_var(response.usage.total_tokens)
        except Exception:
            pass
    except Exception as e:
        print(f"[reconstruct_facts] LLM call failed: {e}")
        return {"f_recon": [], "f_recon_names": [], "h_cycle": 0.0}

    triples = _parse_triples_from_text(answer, ent_names, rel_names)

    # H_cycle against the caller-supplied F_new (paper Eq. eq:cycle_entropy).
    h_cycle = 0.0
    if f_new:
        try:
            from entropy import cycle_entropy
            f_new_names = []
            for f in f_new:
                if len(f) >= 3:
                    h_id, r_id, t_id = f[0], f[1], f[2]
                    h_name = (ent_names or {}).get(h_id, f"Entity_{h_id}")
                    r_name = (rel_names or {}).get(r_id, f"Relation_{r_id}")
                    t_name = (ent_names or {}).get(t_id, f"Entity_{t_id}")
                    f_new_names.append((h_name, r_name, t_name))
            h_cycle = cycle_entropy(f_new_names, triples)
        except Exception as e:
            print(f"[reconstruct_facts] cycle_entropy failed: {e}")

    return {
        "f_recon": triples,
        "f_recon_names": triples,
        "h_cycle": h_cycle,
    }


def verify_fact_consistency(data_dir, resume=True, batch_size=5):
    """Deprecated thin wrapper. Kept for backward compatibility with the
    original main_DKGF.py loop. New code should call reconstruct_facts()
    (paper §Fusion Scene Graph Reconstruction, Prompt 3)."""
    return _legacy_verify_fact_consistency(data_dir, resume=resume, batch_size=batch_size)


def _legacy_verify_fact_consistency(data_dir, resume=True, batch_size=5):
    """Main function to verify fact consistency with reality - Single Fact Evaluation.

    This is the original implementation preserved verbatim for backward
    compatibility with `--wo-scene-graph-reconstruction` and other ablation
    paths. New code should use `reconstruct_facts` (Paper §Fusion Scene Graph
    Reconstruction).
    """

    progress_file = os.path.join(data_dir, 'verification_progress.pkl')

    # Load progress if resuming
    if not resume:
        if os.path.exists(os.path.join(data_dir, "message_pool", 'inconsistent_output_triples_fusion.txt')):
            os.remove(os.path.join(data_dir, "message_pool", 'inconsistent_output_triples_fusion.txt'))
            print("Cleared existing txt output file")
        if os.path.exists(progress_file):
            os.remove(progress_file)
            print("Removed existing progress file")

    progress_data = load_progress(progress_file)
    processed_batches = progress_data['processed_batches']
    consistent_facts = progress_data['consistent_facts']
    inconsistent_facts = progress_data['inconsistent_facts']

    if resume:
        print(f"Resuming: {processed_batches} batches already processed")
        print(f"Current consistent facts: {len(consistent_facts)}")
        print(f"Current inconsistent facts: {len(inconsistent_facts)}")

    # Setting up the OpenAI client
    client = OpenAI(
        base_url="yours",
        api_key="sk-Kdsw4501MdLcWIkD4i6B6nVKJqTNk82QKzz8NjzeJDEP6lxY",
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

    # Load occurred facts
    occurred_facts_file = os.path.join(data_dir, "message_pool", 'occurred_facts.txt')
    if not os.path.exists(occurred_facts_file):
        print(f"Error: {occurred_facts_file} not found!")
        return

    all_facts = load_occurred_facts(occurred_facts_file)
    print(f"Total facts to verify: {len(all_facts)}")

    # Create batches
    fact_batches = batch_facts_for_verification(all_facts, batch_size)
    remaining_batches = fact_batches[processed_batches:]

    print(f"Total batches: {len(fact_batches)}")
    print(f"Remaining batches: {len(remaining_batches)}")

    lock = threading.Lock()

    def process_batch(batch_idx, batch_facts):
        """Process a batch of facts - KEEP MOST, REJECT ONLY OBVIOUS ERRORS"""
        try:
            # ========== 新策略：反向筛选 Prompt ==========
            prompt = """You are a quality control expert reviewing knowledge graph facts. Your task is to identify ONLY the facts that are CLEARLY INCONSISTENT with known reality or contain obvious errors.

For each fact, ask yourself: Is this OBVIOUSLY WRONG or IMPOSSIBLE?

Facts to review:
"""

            fact_descriptions = []
            for i, fact_data in enumerate(batch_facts):
                if len(fact_data) == 6:
                    h, r, t, t1, t2, description = fact_data
                else:
                    h, r, t, t1, t2 = fact_data[:5]
                    description = ""

                head_name = all_entity_names.get(h, f"Entity_{h}")
                tail_name = all_entity_names.get(t, f"Entity_{t}")
                relation_name = rel_names_1.get(r, f"Relation_{r}")
                time1 = time_ids.get(t1, f"Time_{t1}")
                time2 = time_ids.get(t2, f"Time_{t2}")

                fact_desc = f"Fact {i + 1}: {head_name} --{relation_name}--> {tail_name} during {time1} to {time2}"
                if description:
                    fact_desc += f"\nContext: {description}"

                fact_descriptions.append(fact_desc)
                prompt += f"\n{fact_desc}\n"

            prompt += """
REVIEW CRITERIA (BE LENIENT - ONLY FLAG CLEAR ERRORS):

Mark as INCONSISTENT ONLY if the fact:
1. Contains impossible relationships (e.g., "Ancient Rome used smartphones")
2. Has completely wrong time periods (off by centuries)
3. Involves entities that couldn't possibly interact
4. Contains obvious logical contradictions
5. Has clear data corruption or nonsensical combinations

Mark as CONSISTENT if the fact:
- Is plausible, even if you can't verify exact details
- Has approximately correct time periods
- Involves entities that could reasonably interact
- Contains minor name variations or date approximations
- Is simply unverifiable (assume correct when uncertain)

Response format - for EACH fact:
Fact 1: CONSISTENT/INCONSISTENT - [Brief reason ONLY if inconsistent]
Fact 2: CONSISTENT/INCONSISTENT - [Brief reason ONLY if inconsistent]
...

Examples:
✓ "Fact 1: CONSISTENT - Political interactions between these entities are plausible for this timeframe"
✓ "Fact 2: CONSISTENT - Cannot verify exact details but relationship type is reasonable"
✗ "Fact 3: INCONSISTENT - Time period predates the existence of one of the entities"

IMPORTANT: When uncertain, default to CONSISTENT. Only mark INCONSISTENT when you're confident it's wrong.

Your response:"""


            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{'role': 'user', 'content': prompt}],
                temperature=0.5
            )

            answer = response.choices[0].message.content.strip()
            tokens_cal.update_add_var(response.usage.total_tokens)

            print(f"Processing batch {batch_idx + processed_batches + 1}/{len(fact_batches)}")
            print(f"Response: {answer[:200]}...")


            batch_consistent = []
            batch_inconsistent = []

            lines = answer.split('\n')
            fact_results = {}

            for line in lines:
                line = line.strip()
                if 'Fact' in line and ':' in line:
                    try:
                        parts = line.split(':', 1)
                        fact_num = int(re.findall(r'\d+', parts[0])[0]) - 1
                        result_text = parts[1].strip().upper()

                        # Extract the actual result
                        if 'INCONSISTENT' in result_text:
                            result = "INCONSISTENT"
                        else:
                            result = "CONSISTENT"  # Default to CONSISTENT

                        if fact_num < len(batch_facts):
                            fact_results[fact_num] = result
                    except (ValueError, IndexError):
                        continue

            # Classify facts based on results - DEFAULT TO CONSISTENT
            for i, fact_data in enumerate(batch_facts):
                fact_tuple = fact_data[:5]  # (h, r, t, t1, t2)

                # Default to CONSISTENT if not explicitly marked as inconsistent
                result = fact_results.get(i, "CONSISTENT")

                if result == "CONSISTENT":
                    batch_consistent.append(fact_tuple)
                else:
                    batch_inconsistent.append(fact_tuple)

            # Update global results
            with lock:
                consistent_facts.extend(batch_consistent)
                inconsistent_facts.extend(batch_inconsistent)

                print(f"Batch {batch_idx + processed_batches + 1} results:")
                print(f"  Consistent: {len(batch_consistent)}")
                print(f"  Inconsistent: {len(batch_inconsistent)}")

        except Exception as e:
            print(f"Error processing batch {batch_idx}: {str(e)}")
            # In case of error, mark all facts as inconsistent to be safe
            with lock:
                for fact_data in batch_facts:
                    fact_tuple = fact_data[:5]
                    inconsistent_facts.append(fact_tuple)

    # Process batches using thread pool
    executor = ThreadPoolExecutor(max_workers=60)

    try:
        for batch_idx, batch_facts in enumerate(tqdm(remaining_batches)):
            executor.submit(process_batch, batch_idx, batch_facts)

            # Save progress periodically
            if (batch_idx + 1) % 5 == 0:
                with lock:
                    save_progress(progress_file, processed_batches + batch_idx + 1,
                                  consistent_facts, inconsistent_facts)
                    print(f"Progress saved: {processed_batches + batch_idx + 1} batches completed")

        executor.shutdown(wait=True)

    except KeyboardInterrupt:
        print("\nInterrupted by user. Saving progress...")
        executor.shutdown(wait=False)

    # Final save
    total_processed = processed_batches + len(remaining_batches)
    save_progress(progress_file, total_processed, consistent_facts, inconsistent_facts)

    # Save final results to files
    save_results(consistent_facts, inconsistent_facts, data_dir)

    # Print summary
    print(f"\nFinal Summary:")
    print(f"Total facts processed: {len(consistent_facts) + len(inconsistent_facts)}")
    print(f"Consistent facts: {len(consistent_facts)}")
    print(f"Inconsistent facts: {len(inconsistent_facts)}")
    if (len(consistent_facts) + len(inconsistent_facts)) != 0:
        print(
            f"Consistency rate: {len(consistent_facts) / (len(consistent_facts) + len(inconsistent_facts)) * 100:.2f}%")

    return consistent_facts, inconsistent_facts


def _legacy_verify_fact_consistency_duplicate(data_dir, resume=True, batch_size=5):
    """Main function to verify fact consistency - FILTER OUT OBVIOUSLY WRONG FACTS ONLY

    This is a historical duplicate of the original verify_fact_consistency
    implementation; preserved as `_legacy_verify_fact_consistency_duplicate`
    to avoid behavioural changes. The active legacy path is
    `_legacy_verify_fact_consistency`.
    """

    progress_file = os.path.join(data_dir, 'verification_progress.pkl')

    # Load progress if resuming
    if not resume:
        if os.path.exists(os.path.join(data_dir, "message_pool", 'inconsistent_output_triples_fusion.txt')):
            os.remove(os.path.join(data_dir, "message_pool", 'inconsistent_output_triples_fusion.txt'))
            print("Cleared existing txt output file")
        if os.path.exists(progress_file):
            os.remove(progress_file)
            print("Removed existing progress file")

    progress_data = load_progress(progress_file)
    processed_batches = progress_data['processed_batches']
    consistent_facts = progress_data['consistent_facts']
    inconsistent_facts = progress_data['inconsistent_facts']

    if resume:
        print(f"Resuming: {processed_batches} batches already processed")
        print(f"Current consistent facts: {len(consistent_facts)}")
        print(f"Current inconsistent facts: {len(inconsistent_facts)}")

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

    # Load occurred facts
    occurred_facts_file = os.path.join(data_dir, "message_pool", 'occurred_facts.txt')
    if not os.path.exists(occurred_facts_file):
        print(f"Error: {occurred_facts_file} not found!")
        return

    all_facts = load_occurred_facts(occurred_facts_file)
    print(f"Total facts to verify: {len(all_facts)}")

    # Create batches
    fact_batches = batch_facts_for_verification(all_facts, batch_size)
    remaining_batches = fact_batches[processed_batches:]

    print(f"Total batches: {len(fact_batches)}")
    print(f"Remaining batches: {len(remaining_batches)}")

    lock = threading.Lock()

    def process_batch(batch_idx, batch_facts):
        """Process a batch of facts - KEEP MOST, REJECT ONLY OBVIOUS ERRORS"""
        try:
            # ========== 新策略：反向筛选 Prompt ==========
            prompt = """You are a quality control expert reviewing knowledge graph facts. Your task is to identify ONLY the facts that are CLEARLY INCONSISTENT with known reality or contain obvious errors.

For each fact, ask yourself: Is this OBVIOUSLY WRONG or IMPOSSIBLE?

Facts to review:
"""

            fact_descriptions = []
            for i, fact_data in enumerate(batch_facts):
                if len(fact_data) == 6:
                    h, r, t, t1, t2, description = fact_data
                else:
                    h, r, t, t1, t2 = fact_data[:5]
                    description = ""

                head_name = all_entity_names.get(h, f"Entity_{h}")
                tail_name = all_entity_names.get(t, f"Entity_{t}")
                relation_name = rel_names_1.get(r, f"Relation_{r}")
                time1 = time_ids.get(t1, f"Time_{t1}")
                time2 = time_ids.get(t2, f"Time_{t2}")

                fact_desc = f"Fact {i + 1}: {head_name} --{relation_name}--> {tail_name} during {time1} to {time2}"
                if description:
                    fact_desc += f"\nContext: {description}"

                fact_descriptions.append(fact_desc)
                prompt += f"\n{fact_desc}\n"

            prompt += """
REVIEW CRITERIA (BE LENIENT - ONLY FLAG CLEAR ERRORS):

Mark as INCONSISTENT ONLY if the fact:
1. Contains impossible relationships (e.g., "Ancient Rome used smartphones")
2. Has completely wrong time periods (off by centuries)
3. Involves entities that couldn't possibly interact
4. Contains obvious logical contradictions
5. Has clear data corruption or nonsensical combinations

Mark as CONSISTENT if the fact:
- Is plausible, even if you can't verify exact details
- Has approximately correct time periods
- Involves entities that could reasonably interact
- Contains minor name variations or date approximations
- Is simply unverifiable (assume correct when uncertain)

Response format - for EACH fact:
Fact 1: CONSISTENT/INCONSISTENT - [Brief reason ONLY if inconsistent]
Fact 2: CONSISTENT/INCONSISTENT - [Brief reason ONLY if inconsistent]
...

Examples:
✓ "Fact 1: CONSISTENT - Political interactions between these entities are plausible for this timeframe"
✓ "Fact 2: CONSISTENT - Cannot verify exact details but relationship type is reasonable"
✗ "Fact 3: INCONSISTENT - Time period predates the existence of one of the entities"

IMPORTANT: When uncertain, default to CONSISTENT. Only mark INCONSISTENT when you're confident it's wrong.

Your response:"""


            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{'role': 'user', 'content': prompt}],
                temperature=0.5 
            )

            answer = response.choices[0].message.content.strip()
            tokens_cal.update_add_var(response.usage.total_tokens)

            print(f"Processing batch {batch_idx + processed_batches + 1}/{len(fact_batches)}")
            print(f"Response: {answer[:200]}...")


            batch_consistent = []
            batch_inconsistent = []

            lines = answer.split('\n')
            fact_results = {}

            for line in lines:
                line = line.strip()
                if 'Fact' in line and ':' in line:
                    try:
                        parts = line.split(':', 1)
                        fact_num = int(re.findall(r'\d+', parts[0])[0]) - 1
                        result_text = parts[1].strip().upper()

                        # Extract the actual result
                        if 'INCONSISTENT' in result_text:
                            result = "INCONSISTENT"
                        else:
                            result = "CONSISTENT"  # Default to CONSISTENT

                        if fact_num < len(batch_facts):
                            fact_results[fact_num] = result
                    except (ValueError, IndexError):
                        continue

            # Classify facts based on results - DEFAULT TO CONSISTENT
            for i, fact_data in enumerate(batch_facts):
                fact_tuple = fact_data[:5]  # (h, r, t, t1, t2)

                # Default to CONSISTENT if not explicitly marked as inconsistent
                result = fact_results.get(i, "CONSISTENT")

                if result == "CONSISTENT":
                    batch_consistent.append(fact_tuple)
                else:
                    batch_inconsistent.append(fact_tuple)

            # Update global results
            with lock:
                consistent_facts.extend(batch_consistent)
                inconsistent_facts.extend(batch_inconsistent)

                print(f"Batch {batch_idx + processed_batches + 1} results:")
                print(f"  Consistent: {len(batch_consistent)}")
                print(f"  Inconsistent: {len(batch_inconsistent)}")

        except Exception as e:
            print(f"Error processing batch {batch_idx}: {str(e)}")
            # On error, keep all facts as consistent (conservative)
            with lock:
                for fact_data in batch_facts:
                    fact_tuple = fact_data[:5]
                    consistent_facts.append(fact_tuple)

    # Process batches using thread pool
    executor = ThreadPoolExecutor(max_workers=60)

    try:
        for batch_idx, batch_facts in enumerate(tqdm(remaining_batches)):
            executor.submit(process_batch, batch_idx, batch_facts)

            # Save progress periodically
            if (batch_idx + 1) % 5 == 0:
                with lock:
                    save_progress(progress_file, processed_batches + batch_idx + 1,
                                  consistent_facts, inconsistent_facts)
                    print(f"Progress saved: {processed_batches + batch_idx + 1} batches completed")

        executor.shutdown(wait=True)

    except KeyboardInterrupt:
        print("\nInterrupted by user. Saving progress...")
        executor.shutdown(wait=False)

    # Final save
    total_processed = processed_batches + len(remaining_batches)
    save_progress(progress_file, total_processed, consistent_facts, inconsistent_facts)

    # Save final results to files
    save_results(consistent_facts, inconsistent_facts, data_dir)

    # Print summary
    print(f"\nFinal Summary:")
    print(f"Total facts processed: {len(consistent_facts) + len(inconsistent_facts)}")
    print(f"Consistent facts: {len(consistent_facts)}")
    print(f"Inconsistent facts: {len(inconsistent_facts)}")
    if (len(consistent_facts) + len(inconsistent_facts)) != 0:
        print(
            f"Consistency rate: {len(consistent_facts) / (len(consistent_facts) + len(inconsistent_facts)) * 100:.2f}%")

    return consistent_facts, inconsistent_facts


if __name__ == "__main__":
    data_dir = "/home/dex/Desktop/SKGF/TEA-RAG/dataset/wiki_for_icews_0.8_3.7_TF"
    consistent_facts, inconsistent_facts = verify_fact_consistency(data_dir, resume=True, batch_size=5)
