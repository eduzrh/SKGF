import os
import time
import argparse

from modules import extract_relevant_entities_and_triples
from fuzzy_retriever.semantic_rag import semantic_rag_all
from progressive_kg_fusion.on_demand_integration import integration
from progressive_kg_fusion.scene_generation import judge_quadruple_facts
from progressive_kg_fusion.scene_graph_reconstruction import verify_fact_consistency
from fuzzy_retriever.struc import structure_similarity_filter
import openai
import tokens_cal
from fuzzy_retriever.line_trans import (
    convert_triples_to_entities_2,
    generate_line_triples_and_names,
    filter_relevant_triples,
    rank_and_group_triples
)
from fusion_eval import calculate_entity_performance, calculate_fusion_performance

# OpenAI API configuration
os.environ["OPENAI_API_BASE"] = 'yours'
os.environ["OPENAI_API_KEY"] = "yours"
openai.api_key = os.getenv("OPENAI_API_KEY")


def run_full_process(data_dir, args):
    start_time = time.time()
    # Extract relevant entities and triples
    result = extract_relevant_entities_and_triples(
        false_ref_triples_file=os.path.join(data_dir,"false_ref_triples_1_fusion"),
        ref_triples_file=os.path.join(data_dir,"ref_triples_1_fusion"),
        ent_ids_1_file=os.path.join(data_dir,"ent_ids_1"),
        ent_ids_2_file=os.path.join(data_dir,"ent_ids_2"),
        triples_1_file=os.path.join(data_dir,"message_pool","ref_sum_triples_1_fusion"),
        triples_2_file=os.path.join(data_dir,"triples_2"),
        output_dir=os.path.join(data_dir, "message_pool")
    )
    # ========== If Fuzzy Retriever is removed, directly proceed to Scene-aware On-demand Integration ==========
    if not args.use_fuzzy_retriever:
        print("[Ablation] Skipping Fuzzy Retriever - directly copying ref_sum_triples_1_fusion")
        import shutil
        # Directly copy ref_sum_triples_1_fusion to rank_temp_output_triples_fusion.txt
        shutil.copy(
            os.path.join(data_dir, "message_pool", "ref_sum_triples_1_fusion"),
            os.path.join(data_dir, "message_pool", "entity_retriever_outputs.txt")
        )
        # Proceed to Scene-aware On-demand Integration
        # Note: Ensure that files required for subsequent steps exist or are properly handled
    else:
        # ========== Meta-Knowledge Line Graph Transformation ==========
        if args.use_line_graph_trans:
            print("[Ablation] Using Meta-Knowledge Line Graph Transformation")
            generate_line_triples_and_names(
                triples_path_1=os.path.join(data_dir, "message_pool", "ref_triples_1"),
                triples_path_2=os.path.join(data_dir, "message_pool", "ref_triples_2"),
                ent_ids_path_1=os.path.join(data_dir, "ent_ids_1"),
                ent_ids_path_2=os.path.join(data_dir, "ent_ids_2"),
                rel_ids_path_1=os.path.join(data_dir, "rel_ids_1"),
                rel_ids_path_2=os.path.join(data_dir, "rel_ids_2"),
                output_line_triples_1=os.path.join(data_dir, "message_pool", "line_triples_1"),
                output_line_triples_2=os.path.join(data_dir, "message_pool", "line_triples_2"),
                output_line_triples_name_1=os.path.join(data_dir, "message_pool", "line_triples_name_1"),
                output_line_triples_name_2=os.path.join(data_dir, "message_pool", "line_triples_name_2")
            )
        else:
            print("[Ablation] Skipping Meta-Knowledge Line Graph Transformation - using original triples")
            import shutil
            shutil.copy(os.path.join(data_dir, "message_pool", "ref_triples_1"),
                        os.path.join(data_dir, "message_pool", "line_triples_1"))
            shutil.copy(os.path.join(data_dir, "message_pool", "ref_triples_2"),
                        os.path.join(data_dir, "message_pool", "line_triples_2"))

            _create_simple_name_files(data_dir)

        # ========== Semantic Fuzzy Retrieval ==========
        if args.use_semantic_retrieval:
            print("[Ablation] Using Semantic Fuzzy Retrieval")
            semantic_rag_all(data_dir)
        else:
            print("[Ablation] Skipping Semantic Fuzzy Retrieval - using all entity pairs")
            _create_full_entity_pairs(data_dir)

        # ========== Structural Fuzzy Perception ==========
        if args.use_structural_perception:
            print("[Ablation] Using Structural Fuzzy Perception")
            structure_similarity_filter(
                retriever_output_file=os.path.join(data_dir, "message_pool", "retriever_outputs.txt"),
                line_triples_1_file=os.path.join(data_dir, "message_pool", "line_triples_1"),
                line_triples_2_file=os.path.join(data_dir, "message_pool", "line_triples_2"),
                stuc_retriever_output_file=os.path.join(data_dir, "message_pool", "stuc_retriever_output.txt"),
                alpha=args.struct_alpha,
                threshold=args.struct_threshold,
                use_edit_distance=False,
                use_enhanced_structure=True,
                enhanced_threshold=0.5
            )
            retriever_output_path = os.path.join(data_dir, "message_pool", "stuc_retriever_output.txt")
        else:
            print("[Ablation] Skipping Structural Fuzzy Perception")
            retriever_output_path = os.path.join(data_dir, "message_pool", "retriever_outputs.txt")

        # Convert triples to entities
        convert_triples_to_entities_2(
            retriever_output_path=retriever_output_path,
            line_triples_1_path=os.path.join(data_dir, "message_pool", "line_triples_1"),
            line_triples_2_path=os.path.join(data_dir, "message_pool", "line_triples_2"),
            entity_retriever_output_path=os.path.join(data_dir, "message_pool", "entity_retriever_outputs.txt"),
            ent_ids_1_file=os.path.join(data_dir, "ent_ids_1")
        )





    # ========== Progressive KG Fusion (Paper §Entropy-driven Progressive KG Fusion) ==========
    if args.use_progressive_fusion:
        print("[Ablation] Using Progressive KG Fusion")

        # ========== Scene-aware On-demand Integration (Paper §Scene-aware On-demand Integration) ==========
        if args.use_on_demand_integration:
            print("[Paper] Scene-aware On-demand Integration (entity-level relevance)")
            aligned_pairs = integration(data_dir)
            print(f"Found {len(aligned_pairs)} aligned entity pairs from integration.")
        else:
            print("[Ablation] Skipping Scene-aware On-demand Integration - using all entity pairs")
            _create_all_relevance_entities(data_dir)

        # Filter relevant triples (the GKG-side candidate pool ranked by structural / semantic similarity)
        count = filter_relevant_triples(
            os.path.join(data_dir, "false_ref_triples_1_fusion"),
            os.path.join(data_dir, "ref_triples_1_fusion"),
            os.path.join(data_dir, "message_pool", "relevance_entities.txt"),
            os.path.join(data_dir, "message_pool", "temp_output_triples_fusion.txt")
        )

        # Rank and group triples (organize F_new candidate order)
        rank_and_group_triples(
            false_ref_file=os.path.join(data_dir, "false_ref_triples_1_fusion"),
            ref_file=os.path.join(data_dir, "ref_triples_1_fusion"),
            temp_output_file=os.path.join(data_dir, "message_pool", "temp_output_triples_fusion.txt"),
            rank_output_file=os.path.join(data_dir, "message_pool", "rank_temp_output_triples_fusion.txt")
        )

        # ====================================================================================
        # Paper §Entropy-driven Progressive KG Fusion — Algorithm 1, Steps 2.1–2.4
        #   F_valid    = F_recon ∩ F_new              (facts surviving the round-trip)
        #   F_mismatch = F_recon \ (F_new ∪ F^s)      (extraneous reconstructions)
        # Convergence is achieved when F_mismatch == ∅ (entropy minimized); otherwise
        # we tighten the entropy bound τ and inject F_mismatch as negative constraints
        # I_neg into the next iteration's prompt.
        #
        # The four ablation switches preserve the on-disk contract:
        #   --wo-on-demand-integration     : skip Eq. F_new; accept all ranked candidates.
        #   --wo-scene-generation          : skip graph→text; F_recon := F_new (cycle collapses).
        #   --wo-scene-graph-reconstruction: skip text→graph; F_valid := F_new (no cycle check).
        # When the entire progressive_fusion block is disabled via --wo-progressive-fusion
        # we fall back to the legacy _direct_output_triples() path below.
        # ====================================================================================
        from progressive_kg_fusion.scene_generation import scene_generate
        from progressive_kg_fusion.scene_graph_reconstruction import reconstruct_facts
        from progressive_kg_fusion.on_demand_integration import (
            filter_facts_by_tau_batched,
            _build_fact_line,
        )

        # Pre-load the name maps once (used for both Eq. F_new filter and cycle entropy).
        _ent_names_all = _load_all_entity_names(data_dir)
        _rel_names_all = _load_all_relation_names(data_dir)
        _time_ids      = _load_time_id_map(data_dir)
        _skg_relations = _load_skg_relations(data_dir)

        tau_init   = float(getattr(args, "tau_init", 0.5))
        tau_delta  = float(getattr(args, "tau_delta", 0.05))
        max_iter   = int(getattr(args, "max_feedback_iterations", 3))
        h_cycle_eps = float(getattr(args, "h_cycle_eps", 1e-3))

        # Load the candidate pool C^g_meta exactly once. Step 2.1 (Eq. F_new) is
        # re-executed inside the cycle on the remaining subset so τ can be
        # tightened monotonically and I_neg can drive refusal of previously
        # mismatched facts, matching Paper Algorithm 1 line-by-line.
        rank_file = os.path.join(data_dir, "message_pool", 'rank_temp_output_triples_fusion.txt')
        candidate_pool = _load_candidates(rank_file)
        G_local_text = _load_local_subgraph_context(data_dir, candidate_pool)

        # State that the cycle owns: I_neg, accepted-so-far (drives the
        # remaining-candidate view), τ (monotonically increasing).
        F_final = []
        I_neg = []               # Paper I_neg (negative constraint set)
        F_settled = set()        # Facts already accepted on prior iterations
        tau = tau_init
        converged = False
        iteration = -1
        F_valid_named = set()    # pre-initialize so the post-loop guard is safe
        F_new = []               # current F_new, set inside the loop

        for iteration in range(max_iter):
            print(f"============================")
            print(f"[Cycle] iter={iteration} τ={tau:.3f} |I_neg|={len(I_neg)}")

            # ----- Step 2.1 — Eq. F_new (paper Algorithm 1) -----
            # F_new = { f in C^g_meta | P(f | G_local^s, I_neg) >= τ, r in R^s }
            # We filter the *remaining* candidate pool (excluding facts already
            # settled by prior iterations) so τ can be tightened each round.
            remaining_candidates = [
                c for c in candidate_pool
                if tuple(c[:5]) not in F_settled
            ]

            if remaining_candidates:
                if args.use_on_demand_integration:
                    openai_client = _openai_client()
                    accepted, p_per_fact = filter_facts_by_tau_batched(
                        remaining_candidates,
                        G_local_text,
                        tau=tau,
                        client=openai_client,
                        ent_names=_ent_names_all,
                        rel_names=_rel_names_all,
                        time_ids=_time_ids,
                        negative_constraints=I_neg,
                    )
                    F_new = list(accepted)
                else:
                    # Ablation: --wo-on-demand-integration accepts all remaining.
                    F_new = list(remaining_candidates)
            else:
                F_new = []

            # Paper Eq. F_new: enforce r ∈ R^s (paper §Scene-aware On-demand Integration).
            if F_new and _skg_relations:
                pre_count = len(F_new)
                F_new = [f for f in F_new if f[1] in _skg_relations]
                if pre_count != len(F_new):
                    print(f"[Eq. F_new] dropped {pre_count - len(F_new)} facts with r ∉ R^s")

            print(
                f"[Eq. F_new] iter={iteration} |C^g_meta|={len(remaining_candidates)} "
                f"|F_new|={len(F_new)} τ={tau:.3f}"
            )
            _write_f_new_to_disk(F_new, data_dir)

            if not F_new:
                print(f"[Cycle] iter={iteration} |F_new|==0, breaking")
                break

            print(f"[Cycle] iter={iteration} |F_new|={len(F_new)} |I_neg|={len(I_neg)}")

            # Step 2.2 — Fusion Scene Generation (Paper Prompt 2).
            if args.use_scene_generation:
                scene = scene_generate(
                    data_dir,
                    resume=False,
                    f_new=F_new,
                    g_local_context=G_local_text,
                    negative_constraints=I_neg,
                    tau_ent=args.tau_ent,
                )
                scene_text = scene.get("scene_text", "") or ""
                h_gen = scene.get("h_gen", 0.0)
                h_gen_high = scene.get("h_gen_high", False)
                print(f"[Cycle] H_gen={h_gen:.4f} high={h_gen_high}")

                # Paper §Entropy-driven Validity Estimation: facts whose scene
                # description has H_gen > τ_ent are flagged as high-entropy
                # noise and discarded. We push them into I_neg so they are
                # never re-accepted in subsequent rounds.
                if h_gen_high:
                    F_settled.update(tuple(f[:5]) for f in F_new)
                    I_neg.extend(
                        f"({h}, {r}, {t})" for (h, r, t, *_) in F_new
                    )
                    tau = min(1.0, tau + tau_delta)
                    print(
                        f"[Cycle] H_gen>{args.tau_ent}: discarded {len(F_new)} "
                        f"high-entropy facts and tightened τ to {tau:.3f}"
                    )
                    continue
            else:
                # Ablation: skip graph→text; F_recon := F_new (cycle collapses).
                scene_text = ""
                print("[Cycle] --wo-scene-generation: F_recon := F_new")

            # Step 2.3 — Fusion Scene Graph Reconstruction (Paper Prompt 3).
            if args.use_scene_graph_reconstruction and scene_text:
                openai_client = _openai_client()
                recon = reconstruct_facts(
                    scene_text,
                    openai_client,
                    ent_names=_ent_names_all,
                    rel_names=_rel_names_all,
                    f_new=F_new,
                )
                F_recon = set(recon["f_recon"])
                h_cycle = recon["h_cycle"]
                print(f"[Cycle] |F_recon|={len(F_recon)} H_cycle={h_cycle:.4f}")
            elif not args.use_scene_graph_reconstruction:
                # User ablated: skip text→graph; F_valid := F_new.
                F_recon = set(_named_facts(F_new, _ent_names_all, _rel_names_all))
                h_cycle = 0.0
                print(f"[Cycle] --wo-scene-graph-reconstruction: F_valid := F_new, |F_recon|={len(F_recon)}")
            else:
                # scene_text was empty (likely F_new was empty); degrade to F_valid := F_new.
                F_recon = set(_named_facts(F_new, _ent_names_all, _rel_names_all))
                h_cycle = 0.0
                print(f"[Cycle] empty scene_text; F_valid := F_new, |F_recon|={len(F_recon)}")

            # Step 2.4 — Cycle-consistency check.
            # The cycle compares NAMES (reconstructed text round-trips through
            # names), so we project F_new (IDs) to name form to match F_recon.
            # F_valid is then mapped back to the original (h, r, t, t1, t2)
            # tuples from F_new so the on-disk writer preserves timestamps.
            F_new_named = set(_named_facts(F_new, _ent_names_all, _rel_names_all))
            F_valid_named = F_new_named & F_recon
            F_mismatch = F_recon - F_new_named
            print(f"[Cycle] |F_valid|={len(F_valid_named)} |F_mismatch|={len(F_mismatch)}")

            if (not F_mismatch) or h_cycle < h_cycle_eps:
                # Project surviving names back to their source tuples (with timestamps).
                f_new_name_to_tuple = {
                    (ent_names.get(h, f"Entity_{h}"),
                     rel_names.get(r, f"Relation_{r}"),
                     ent_names.get(t, f"Entity_{t}")): (h, r, t, t1, t2)
                    for (h, r, t, t1, t2) in F_new
                    for ent_names in (_ent_names_all,)
                    for rel_names in (_rel_names_all,)
                }
                F_final.extend(
                    sorted({f_new_name_to_tuple[n] for n in F_valid_named
                            if n in f_new_name_to_tuple})
                )
                # Mark the round's valid facts as settled so they never enter
                # the remaining-candidate view again.
                F_settled.update(tuple(f[:5]) for f in F_new)
                converged = True
                print(f"[Cycle] Converged at iter={iteration} (entropy minimized).")
                break

            # Inject mismatch as negative constraints for the next round and
            # mark this iteration's F_new as settled (paper Algorithm 1:
            # τ = τ + δ happens at iteration boundary even on mismatch paths).
            I_neg.extend(sorted(F_mismatch))
            F_settled.update(tuple(f[:5]) for f in F_new)
            tau = min(1.0, tau + tau_delta)

        if not converged and F_valid_named:
            # Last-iteration fallback: accept whatever still passes the cycle.
            f_new_name_to_tuple = {
                (ent_names.get(h, f"Entity_{h}"),
                 rel_names.get(r, f"Relation_{r}"),
                 ent_names.get(t, f"Entity_{t}")): (h, r, t, t1, t2)
                for (h, r, t, t1, t2) in F_new
                for ent_names in (_ent_names_all,)
                for rel_names in (_rel_names_all,)
            }
            F_final.extend(
                sorted({f_new_name_to_tuple[n] for n in F_valid_named
                        if n in f_new_name_to_tuple})
            )

        # Step 2.4 (final) — Persist F_final as the canonical fused output.
        _write_output_fusion(F_final, data_dir)
        print(
            f"[Cycle] Final: |F_final|={len(F_final)}, τ_final={tau:.3f}, "
            f"converged={converged}, iterations={iteration + 1}"
        )
    else:
        print("[Ablation] Skipping Progressive KG Fusion - directly using filtered triples")
        _direct_output_triples(data_dir)

    # Deduplicate
    output_file = os.path.join(data_dir, "message_pool", "output_triples_fusion.txt")
    if os.path.exists(output_file):
        lines = open(output_file, 'r', encoding='utf-8').readlines()
        open(output_file, 'w', encoding='utf-8').writelines(dict.fromkeys(lines))

    # Calculate metrics
    end_time = time.time()
    total_seconds = end_time - start_time

    ref_path = os.path.join(data_dir, "ref_triples_1_fusion")
    with open(ref_path, 'r') as file:
        count = sum(1 for _ in file) * 2

    avg_time = total_seconds / count if count > 0 else 0
    avg_tokens = tokens_cal.global_tokens / count if count > 0 else 0

    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)

    print(f"Average time:  {avg_time:.4f} seconds")
    print(f"Average tokens: {avg_tokens:.2f} tokens")
    print(f"Final Process completed in: {end_time - start_time:.2f} seconds")
    print(f'Time Cost : {hours}hour, {minutes:02d}min, {seconds:02d}sec')
    print(f'Tokens Cost : {tokens_cal.global_tokens}')

    # Calculate entity performance
    entity_acc, entity_precision, entity_recall, entity_f1 = calculate_entity_performance(
        os.path.join(data_dir, "message_pool", "output_triples_fusion.txt"),
        os.path.join(data_dir, "ref_relative_ent_ids_2"),
        os.path.join(data_dir, "ent_ids_2")
    )

    # Calculate fusion performance
    fusion_acc, fusion_precision, fusion_recall, fusion_f1 = calculate_fusion_performance(
        os.path.join(data_dir, "message_pool", "output_triples_fusion.txt"),
        os.path.join(data_dir, "ref_triples_1_fusion"),
        os.path.join(data_dir, "false_ref_triples_1_fusion")
    )


# ========== 辅助函数：为消融实验提供默认行为 ==========
def _create_simple_name_files(data_dir):
    """Create simplified name files (without performing line graph transformation)"""

    def load_and_write_names(triple_file, ent_ids, rel_ids, output_name_file):
        ent_map = {}
        with open(ent_ids, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    ent_map[parts[0]] = parts[1]

        rel_map = {}
        with open(rel_ids, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    rel_map[parts[0]] = parts[1]


        with open(triple_file, 'r', encoding='utf-8') as f_in:
            triples = f_in.readlines()

        with open(output_name_file.replace('_name_', '_'), 'w', encoding='utf-8') as f_out:
            for idx, line in enumerate(triples):
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    h, r, t = parts[0], parts[1], parts[2]
                    rest = parts[3:] if len(parts) > 3 else []
                    # triple_id, head, rel, tail, [time1, time2]
                    f_out.write(f"{idx}\t{h}\t{r}\t{t}\t{' '.join(rest)}\n")


        with open(triple_file, 'r', encoding='utf-8') as f_in:
            with open(output_name_file, 'w', encoding='utf-8') as f_out:
                for idx, line in enumerate(f_in):
                    parts = line.strip().split('\t')
                    if len(parts) >= 3:
                        h, r, t = parts[0], parts[1], parts[2]
                        h_name = ent_map.get(h, h)
                        r_name = rel_map.get(r, r)
                        t_name = ent_map.get(t, t)
                        f_out.write(f"{idx}\t{h_name}|{r_name}|{t_name}\n")

    # KG1
    load_and_write_names(
        os.path.join(data_dir, "message_pool", "ref_triples_1"),
        os.path.join(data_dir, "ent_ids_1"),
        os.path.join(data_dir, "rel_ids_1"),
        os.path.join(data_dir, "message_pool", "line_triples_name_1")
    )

    # KG2
    load_and_write_names(
        os.path.join(data_dir, "message_pool", "ref_triples_2"),
        os.path.join(data_dir, "ent_ids_2"),
        os.path.join(data_dir, "rel_ids_2"),
        os.path.join(data_dir, "message_pool", "line_triples_name_2")
    )


def _create_full_entity_pairs(data_dir):
    """Create all possible entity pairs (without semantic retrieval) - corrected version"""

    triple_ids_1 = []
    with open(os.path.join(data_dir, "message_pool", "line_triples_name_1"), 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if parts:
                triple_ids_1.append(parts[0])


    triple_ids_2 = []
    with open(os.path.join(data_dir, "message_pool", "line_triples_name_2"), 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if parts:
                triple_ids_2.append(parts[0])


    k1 = min(100, len(triple_ids_1))
    k2 = min(10, len(triple_ids_2))

    with open(os.path.join(data_dir, "message_pool", "retriever_outputs.txt"), 'w', encoding='utf-8') as f:
        for t1 in triple_ids_1[:k1]:
            for t2 in triple_ids_2[:k2]:
                f.write(f"{t1}\t{t2}\n")


def _create_all_relevance_entities(data_dir):
    """Use all entity pairs as relevant entities (without on-demand integration)"""
    input_file = os.path.join(data_dir, "message_pool", "entity_retriever_outputs.txt")
    output_file = os.path.join(data_dir, "message_pool", "relevance_entities.txt")

    if os.path.exists(input_file):
        import shutil
        shutil.copy(input_file, output_file)
    else:
        open(output_file, 'w').close()
        print("Warning: entity_retriever_outputs.txt not found, created empty relevance_entities.txt")


def _accept_all_facts(data_dir):
    """Accept all facts (without scene generation judgment)"""
    rank_file = os.path.join(data_dir, "message_pool", "rank_temp_output_triples_fusion.txt")
    occurred_file = os.path.join(data_dir, "message_pool", "occurred_facts.txt")

    if not os.path.exists(rank_file):
        print(f"Warning: {rank_file} not found!")
        open(occurred_file, 'w').close()
        return

    with open(rank_file, 'r', encoding='utf-8') as f_in:
        with open(occurred_file, 'w', encoding='utf-8') as f_out:
            for line in f_in:
                parts = line.strip().split('\t')
                if len(parts) >= 5:
                    # h, r, t, t1, t2
                    f_out.write('\t'.join(parts[:5]) + '\n')


def _skip_consistency_check(data_dir):
    """Skip consistency check (directly output all facts)"""
    occurred_file = os.path.join(data_dir, "message_pool", "occurred_facts.txt")
    output_file = os.path.join(data_dir, "message_pool", "output_triples_fusion.txt")

    if not os.path.exists(occurred_file):
        print(f"Warning: {occurred_file} not found!")
        open(output_file, 'w').close()
        return

    with open(occurred_file, 'r', encoding='utf-8') as f_in:
        with open(output_file, 'w', encoding='utf-8') as f_out:
            for line in f_in:
                parts = line.strip().split('\t')
                if len(parts) >= 5:
                    f_out.write('\t'.join(parts[:5]) + '\n')


    inconsistent_file = os.path.join(data_dir, "message_pool", "inconsistent_output_triples_fusion.txt")
    open(inconsistent_file, 'w').close()


def _direct_output_triples(data_dir):
    """Directly output filtered triples (without progressive fusion)"""

    entity_file = os.path.join(data_dir, "message_pool", "entity_retriever_outputs.txt")

    if not os.path.exists(entity_file):
        print(f"Warning: {entity_file} not found! Creating empty output.")
        open(os.path.join(data_dir, "message_pool", "output_triples_fusion.txt"), 'w').close()
        return


    from fuzzy_retriever.line_trans import filter_relevant_triples

    count = filter_relevant_triples(
        os.path.join(data_dir, "false_ref_triples_1_fusion"),
        os.path.join(data_dir, "ref_triples_1_fusion"),
        entity_file,
        os.path.join(data_dir, "message_pool", "output_triples_fusion.txt")
    )

    print(f"Directly output {count} triples without progressive fusion")


# =============================================================================
# Paper §Entropy-driven Progressive KG Fusion — helper functions.
# These back the new Algorithm-1 loop in run_full_process() and keep the
# legacy file I/O contract intact (output_triples_fusion.txt remains a flat
# list of `(h, r, t, t1, t2)` tuples, one per line, as fusion_eval expects).
# =============================================================================
def _load_candidates(rank_file):
    """Load (h, r, t, t1, t2) candidates from rank_temp_output_triples_fusion.txt."""
    candidates = []
    if not os.path.exists(rank_file):
        return candidates
    with open(rank_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 5:
                try:
                    candidates.append(tuple(int(x) for x in parts[:5]))
                except ValueError:
                    continue
    return candidates


def _load_all_entity_names(data_dir):
    """Union of KG1 + KG2 entity name maps."""
    names = {}
    for fn in ("ent_ids_1", "ent_ids_2"):
        path = os.path.join(data_dir, fn)
        if not os.path.exists(path):
            continue
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    try:
                        names[int(parts[0])] = '\t'.join(parts[1:])
                    except ValueError:
                        continue
    return names


def _load_all_relation_names(data_dir):
    """Union of KG1 + KG2 relation name maps."""
    names = {}
    for fn in ("rel_ids_1", "rel_ids_2"):
        path = os.path.join(data_dir, fn)
        if not os.path.exists(path):
            continue
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    try:
                        names[int(parts[0])] = '\t'.join(parts[1:])
                    except ValueError:
                        continue
    return names


def _load_time_id_map(data_dir):
    """Load the time_id file as {tid -> timestamp}."""
    time_ids = {}
    path = os.path.join(data_dir, 'time_id')
    if not os.path.exists(path):
        return time_ids
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    time_ids[int(parts[0])] = parts[1]
                except ValueError:
                    continue
    return time_ids


def _load_skg_relations(data_dir):
    """Load the SKG relation set R^s (paper Eq. F_new constraint: r in R^s).

    Sources the relation ids from rel_ids_1 (the SKG side) and ref_triples_1
    (cross-checked). Returns a set[int] of relation ids observed in the SKG.
    """
    rels = set()
    rel_ids_path = os.path.join(data_dir, 'rel_ids_1')
    if os.path.exists(rel_ids_path):
        with open(rel_ids_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    try:
                        rels.add(int(parts[0]))
                    except ValueError:
                        continue
    # Cross-check against the SKG triples themselves.
    for fn in ('ref_triples_1', 'sup_triples_1', 'triples_1', 'ref_triples_1_fusion'):
        path = os.path.join(data_dir, fn)
        if not os.path.exists(path):
            continue
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        rels.add(int(parts[1]))
                    except (ValueError, IndexError):
                        continue
    return rels


def _load_local_subgraph_context(data_dir, candidates, max_facts=20):
    """Render a short textual G_local^s context for Prompt 1.

    We render the first few candidate facts so the model has a deterministic
    local reference structure without paying for a separate LLM call. This
    mirrors the spirit of the paper's "Local Subgraph Context" placeholder in
    PROMPT_1_VALIDITY without altering the candidate ranking.
    """
    if not candidates:
        return "(empty local subgraph)"
    sample = candidates[:max_facts]
    ent_names = _load_all_entity_names(data_dir)
    rel_names = _load_all_relation_names(data_dir)
    time_ids  = _load_time_id_map(data_dir)
    lines = []
    for fact in sample:
        try:
            h, r, t, t1, t2 = fact[:5]
        except Exception:
            continue
        h_name = ent_names.get(h, f"Entity_{h}")
        r_name = rel_names.get(r, f"Relation_{r}")
        t_name = ent_names.get(t, f"Entity_{t}")
        time_str = ""
        if t1 and t2:
            t1s = time_ids.get(t1, f"Time_{t1}")
            t2s = time_ids.get(t2, f"Time_{t2}")
            time_str = f" during {t1s} to {t2s}"
        lines.append(f"- ({h_name}, {r_name}, {t_name}){time_str}")
    return "\n".join(lines) if lines else "(empty local subgraph)"


def _openai_client():
    """Construct an OpenAI-compatible chat client from the module's env vars."""
    import httpx
    from openai import OpenAI
    return OpenAI(
        base_url=os.environ.get("OPENAI_API_BASE", "yours"),
        api_key=os.environ.get("OPENAI_API_KEY", "yours"),
        http_client=httpx.Client(
            base_url=os.environ.get("OPENAI_API_BASE", "yours"),
            follow_redirects=True,
        ),
    )


def _named_facts(facts, ent_names, rel_names):
    """Project a list of (h, r, t[, t1, t2]) tuples into (name_h, name_r, name_t)
    triples for cycle-consistency set arithmetic (Paper §Fusion Scene Graph
    Reconstruction)."""
    out = []
    for f in facts:
        if len(f) >= 3:
            h, r, t = f[0], f[1], f[2]
            h_name = ent_names.get(h, f"Entity_{h}")
            r_name = rel_names.get(r, f"Relation_{r}")
            t_name = ent_names.get(t, f"Entity_{t}")
            out.append((h_name, r_name, t_name))
    return out


def _write_f_new_to_disk(F_new, data_dir):
    """Persist Eq. F_new to message_pool/f_new.txt (one (h,r,t,t1,t2) per line).

    Useful for debugging and for downstream ablations that need to inspect
    the post-filter candidate pool.
    """
    path = os.path.join(data_dir, "message_pool", "f_new.txt")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for fact in F_new:
            if len(fact) >= 5:
                f.write('\t'.join(str(x) for x in fact[:5]) + '\n')


def _write_output_fusion(F_final, data_dir):
    """Persist the converged F_final as the canonical fused output.

    F_final is a list of (h, r, t, t1, t2) tuples — the source IDs from
    rank_temp_output_triples_fusion.txt whose names round-tripped through
    the cycle. We write `(h, r, t, t1, t2)` lines so that
    fusion_eval.calculate_fusion_performance continues to work unchanged
    and the timestamp IDs preserved by the source file are kept intact.
    """
    path = os.path.join(data_dir, "message_pool", "output_triples_fusion.txt")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    seen = set()
    with open(path, 'w', encoding='utf-8') as f:
        for fact in F_final:
            if len(fact) < 3:
                continue
            key = (fact[0], fact[1], fact[2])
            if key in seen:
                continue
            seen.add(key)
            h, r, t = fact[0], fact[1], fact[2]
            t1 = fact[3] if len(fact) > 3 else 0
            t2 = fact[4] if len(fact) > 4 else 0
            f.write(f"{h}\t{r}\t{t}\t{t1}\t{t2}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ablation Study Options")
    parser.add_argument("--data", type=str, default="W-I-S1", help="Dataset name")

    # 消融实验选项
    parser.add_argument("--wo-fuzzy-retriever", action="store_true",
                        help="Remove Fuzzy Retriever")
    parser.add_argument("--wo-progressive-fusion", action="store_true",
                        help="Remove Progressive KG Fusion")
    parser.add_argument("--wo-line-graph-trans", action="store_true",
                        help="Remove Meta-knowledge Line Graph Transformation")
    parser.add_argument("--wo-semantic-retrieval", action="store_true",
                        help="Remove Semantic Fuzzy Retrieval")
    parser.add_argument("--wo-structural-perception", action="store_true",
                        help="Remove Structural Fuzzy Perception")
    parser.add_argument("--wo-on-demand-integration", action="store_true",
                        help="Remove Scene-aware On-demand Integration")
    parser.add_argument("--wo-scene-generation", action="store_true",
                        help="Remove Scene Generation")
    parser.add_argument("--wo-scene-graph-reconstruction", action="store_true",
                        help="Remove Scene Graph Reconstruction")

    # Paper Algorithm 1 hyperparameters (entropy-driven progressive loop)
    parser.add_argument("--tau-init", type=float, default=0.5,
                        help="Initial entropy bound tau for F_new")
    parser.add_argument("--tau-delta", type=float, default=0.05,
                        help="Per-iteration increment of tau")
    parser.add_argument("--max-feedback-iterations", type=int, default=3,
                        help="Maximum number of cycle-consistency iterations")
    parser.add_argument("--h-cycle-eps", type=float, default=1e-3,
                        help="Cycle-consistency entropy convergence threshold")
    parser.add_argument("--tau-ent", type=float, default=1.5,
                        help="High-entropy threshold tau_ent for scene_generate "
                             "(marks H_gen > tau_ent as high-entropy scene)")
    parser.add_argument("--struct-alpha", type=float, default=0.5,
                        help="Structural-weight alpha in structure_similarity_filter "
                             "(blends structural similarity with neutral 0.5 prior)")
    parser.add_argument("--struct-threshold", type=float, default=0.26,
                        help="Final-score threshold for structure_similarity_filter "
                             "(pairs below this score are dropped)")

    args = parser.parse_args()

    args.use_fuzzy_retriever = not args.wo_fuzzy_retriever
    args.use_progressive_fusion = not args.wo_progressive_fusion
    args.use_line_graph_trans = not args.wo_line_graph_trans and args.use_fuzzy_retriever
    args.use_semantic_retrieval = not args.wo_semantic_retrieval and args.use_fuzzy_retriever
    args.use_structural_perception = not args.wo_structural_perception and args.use_fuzzy_retriever
    args.use_on_demand_integration = not args.wo_on_demand_integration and args.use_progressive_fusion
    args.use_scene_generation = not args.wo_scene_generation and args.use_progressive_fusion
    args.use_scene_graph_reconstruction = not args.wo_scene_graph_reconstruction and args.use_progressive_fusion


    print("=" * 60)
    print("Ablation Study Configuration:")
    print("=" * 60)
    print(f"Fuzzy Retriever: {'✓' if args.use_fuzzy_retriever else '✗'}")
    print(f"  - Meta-knowledge Line Graph Transformation: {'✓' if args.use_line_graph_trans else '✗'}")
    print(f"  - Semantic Fuzzy Retrieval: {'✓' if args.use_semantic_retrieval else '✗'}")
    print(f"  - Structural Fuzzy Perception: {'✓' if args.use_structural_perception else '✗'}")
    print(f"Progressive KG Fusion: {'✓' if args.use_progressive_fusion else '✗'}")
    print(f"  - Scene-aware On-demand Integration: {'✓' if args.use_on_demand_integration else '✗'}")
    print(f"  - Scene Generation: {'✓' if args.use_scene_generation else '✗'}")
    print(f"  - Scene Graph Reconstruction: {'✓' if args.use_scene_graph_reconstruction else '✗'}")

    print("=" * 60)

    data_dir = os.path.join("./SKGF-main/dataset", args.data)
    run_full_process(data_dir, args)
