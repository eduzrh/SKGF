import numpy as np
from collections import defaultdict, deque
from typing import Dict, List, Set, Tuple
import os
import sys
import subprocess
import networkx as nx
from tqdm import tqdm


def structure_similarity_filter(retriever_output_file: str,
                                line_triples_1_file: str,
                                line_triples_2_file: str,
                                stuc_retriever_output_file: str,
                                alpha: float = 0.5,
                                threshold: float = 0.4,
                                use_edit_distance: bool = False,
                                use_enhanced_structure: bool = True,
                                enhanced_threshold: float = 0.5):


    # 1. Read equivalent line entity pairs
    print("Reading equivalent line entity pairs...")
    equivalent_pairs = []
    with open(retriever_output_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                kg1_triple_id = int(parts[0])
                kg2_triple_id = int(parts[1])
                equivalent_pairs.append((kg1_triple_id, kg2_triple_id))

    # 2. Read triple data
    print("Reading KG1 triple data...")
    kg1_triples = {}  # triple_id -> (head, relation, tail)
    kg2_triples = {}

    # Read KG1 triples
    with open(line_triples_1_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Reading KG1 triples"):
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                triple_id = int(parts[0])
                head = int(parts[1])
                relation = int(parts[2])
                tail = int(parts[3])
                kg1_triples[triple_id] = (head, relation, tail)

    # Read KG2 triples
    print("正在读取KG2三元组数据...")
    with open(line_triples_2_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Reading KG2 triples"):
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                triple_id = int(parts[0])
                head = int(parts[1])
                relation = int(parts[2])
                tail = int(parts[3])
                kg2_triples[triple_id] = (head, relation, tail)

    # 3. Build entity-to-triples mapping (for finding neighbors)
    print("Building entity-to-triples mapping...")

    def build_entity_to_triples_map(triples_dict):
        entity_to_triples = defaultdict(set)
        for triple_id, (head, relation, tail) in triples_dict.items():
            entity_to_triples[head].add(triple_id)
            entity_to_triples[tail].add(triple_id)
        return entity_to_triples

    kg1_entity_to_triples = build_entity_to_triples_map(kg1_triples)
    kg2_entity_to_triples = build_entity_to_triples_map(kg2_triples)

    # 4. Get neighbors of line entities
    def get_neighbors(triple_id, triples_dict, entity_to_triples):
        if triple_id not in triples_dict:
            return set()

        head, relation, tail = triples_dict[triple_id]
        neighbors = set()


        for neighbor_id in entity_to_triples[head]:
            if neighbor_id != triple_id:
                neighbors.add(neighbor_id)


        for neighbor_id in entity_to_triples[tail]:
            if neighbor_id != triple_id:
                neighbors.add(neighbor_id)

        return neighbors

    # 5. Build equivalence relation mapping
    print("Building equivalence relation mapping...")
    equivalent_map = {}
    for kg1_id, kg2_id in tqdm(equivalent_pairs, desc="Building equivalence mapping"):
        equivalent_map[(kg1_id, kg2_id)] = True
        equivalent_map[(kg2_id, kg1_id)] = True

    # 6. Calculate structural similarity
    def calculate_structure_similarity(kg1_triple_id, kg2_triple_id):
        # Get neighbors
        kg1_neighbors = get_neighbors(kg1_triple_id, kg1_triples, kg1_entity_to_triples)
        kg2_neighbors = get_neighbors(kg2_triple_id, kg2_triples, kg2_entity_to_triples)

        if use_edit_distance:
            return calculate_edit_distance_similarity(
                kg1_triple_id, kg2_triple_id,
                kg1_neighbors, kg2_neighbors,
                equivalent_map
            )
        else:
            return calculate_max_common_subgraph_similarity(
                kg1_triple_id, kg2_triple_id,
                kg1_neighbors, kg2_neighbors,
                equivalent_map
            )

    def calculate_max_common_subgraph_similarity(kg1_triple_id, kg2_triple_id,
                                                 kg1_neighbors, kg2_neighbors,
                                                 equiv_map):
        # Convert to sorted lists to ensure consistent order
        kg1_neighbors_sorted = sorted(list(kg1_neighbors))
        kg2_neighbors_sorted = sorted(list(kg2_neighbors))


        g1 = nx.Graph()
        g2 = nx.Graph()


        g1.add_node(kg1_triple_id)
        g2.add_node(kg2_triple_id)


        for neighbor in kg1_neighbors_sorted:
            g1.add_node(neighbor)
            g1.add_edge(kg1_triple_id, neighbor)

        for neighbor in kg2_neighbors_sorted:
            g2.add_node(neighbor)
            g2.add_edge(kg2_triple_id, neighbor)

        # Calculate maximum common subgraph size
        common_edges = 0
        matched_kg1_neighbors = set()

        # Center node matching
        if (kg1_triple_id, kg2_triple_id) in equiv_map:
            common_edges += 1

        # Neighbor node matching (using sorted lists)
        for kg1_neighbor in kg1_neighbors_sorted:
            for kg2_neighbor in kg2_neighbors_sorted:
                if ((kg1_neighbor, kg2_neighbor) in equiv_map and
                        kg1_neighbor not in matched_kg1_neighbors):
                    common_edges += 1
                    matched_kg1_neighbors.add(kg1_neighbor)
                    break

        # Calculate similarity
        total_edges = max(len(kg1_neighbors_sorted), len(kg2_neighbors_sorted), 1)
        similarity = common_edges / total_edges

        print(common_edges, total_edges)
        return min(similarity, 1.0)

    def calculate_edit_distance_similarity(kg1_triple_id, kg2_triple_id,
                                           kg1_neighbors, kg2_neighbors,
                                           equiv_map):
        # Convert neighbors to lists for edit distance calculation
        kg1_list = [kg1_triple_id] + list(kg1_neighbors)
        kg2_list = [kg2_triple_id] + list(kg2_neighbors)

        # Calculate edit distance
        m, n = len(kg1_list), len(kg2_list)
        dp = [[0] * (n + 1) for _ in range(m + 1)]


        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j


        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if (kg1_list[i - 1], kg2_list[j - 1]) in equiv_map:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = min(
                        dp[i - 1][j] + 1,
                        dp[i][j - 1] + 1,
                        dp[i - 1][j - 1] + 1
                    )

        edit_distance = dp[m][n]
        max_len = max(m, n)
        similarity = 1.0 - (edit_distance / max_len) if max_len > 0 else 0.0

        return max(similarity, 0.0)

    # 7. Process each equivalent pair and calculate final score
    print("Processing equivalent pairs and calculating similarity...")
    filtered_pairs = []

    for kg1_triple_id, kg2_triple_id in tqdm(equivalent_pairs, desc="Processing equivalent pairs"):

        struct_similarity = calculate_structure_similarity(kg1_triple_id, kg2_triple_id)


        final_score = struct_similarity * (1 - alpha) + 0.5 * alpha


        if final_score > threshold:
            filtered_pairs.append((kg1_triple_id, kg2_triple_id))

        print(f"Triple pair ({kg1_triple_id}, {kg2_triple_id}): "
              f"struct_sim={struct_similarity:.4f}, final_score={final_score:.4f}")

    # 8. Save local structural similarity filtering results
    print("Saving local structural similarity filtering results...")
    with open(stuc_retriever_output_file, 'w', encoding='utf-8') as f:
        for kg1_id, kg2_id in tqdm(filtered_pairs, desc="Saving results"):
            f.write(f"{kg1_id}\t{kg2_id}\n")


    print(f"Original pair count: {len(equivalent_pairs)}")
    print(f"After local structure filtering: {len(filtered_pairs)}")
    print(f"Results saved to: {stuc_retriever_output_file}")

    # 9. Enhanced structural embedding calculation (global structural feature learning)
    if use_enhanced_structure:
        print("\n" + "=" * 50)
        print("Starting enhanced structural embedding calculation (deep structural feature learning)...")
        print("=" * 50)

        # Get current script directory (fuzzy_retriever directory)
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Build absolute path for struct_enhance_1.py
        script_path = os.path.join(script_dir, 'struct_enhance_1.py')
        # Check if file exists
        if not os.path.exists(script_path):
            raise FileNotFoundError(f"File not found: {script_path}")

        data_dir = os.path.dirname(os.path.dirname(retriever_output_file))
        enhanced_output = os.path.join(os.path.dirname(stuc_retriever_output_file),
                                       "enhanced_structure_output.txt")

        # Build enhanced structure calculation command
        structure_enhance_cmd = [
            sys.executable,
             script_path,  # Ensure struct_enhance_1.py is in same directory or provide full path
            "--dim", "100",
            "--epochs", "50",
            "--batch_size", "512",
            "--lr", "0.01",
            "--neg_ratio", "1.0",
            "--threshold", str(enhanced_threshold),
            "--device", "cuda",
            "--data_dir", data_dir,
            "--output_path", enhanced_output
        ]

        try:
            # Run enhanced structure calculation
            result = subprocess.run(structure_enhance_cmd,
                                    check=True,
                                    capture_output=True,
                                    text=True)
            print(result.stdout)
            print(f"\nEnhanced structure calculation results saved to: {enhanced_output}")


            fuse_structure_features(
                local_structure_file=stuc_retriever_output_file,
                global_structure_file=enhanced_output,
                fused_output_file=os.path.join(
                    os.path.dirname(stuc_retriever_output_file),
                    "fused_structure_output.txt"
                ),
                fusion_strategy="intersection"
            )

        except subprocess.CalledProcessError as e:
            print(f"Enhanced structure calculation failed: {e}")
            print(f"Error output: {e.stderr}")
        except FileNotFoundError:
            print("Error: Structure enhancement calculation module not found, ensure it's in correct location")


def fuse_structure_features(local_structure_file: str,
                            global_structure_file: str,
                            fused_output_file: str,
                            fusion_strategy: str = "intersection"):
    """
    Fuse local and global structural features

    Args:
        local_structure_file: Local structural similarity results
        global_structure_file: Global structural embedding calculation results
        fused_output_file: Fused output file
        fusion_strategy: Fusion strategy ("intersection" or "union")
    """
    print("\nFusing local and global structural features...")

    # Read local structure results
    local_pairs = set()
    with open(local_structure_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                local_pairs.add((int(parts[0]), int(parts[1])))


    global_pairs = set()
    with open(global_structure_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                global_pairs.add((int(parts[0]), int(parts[1])))


    if fusion_strategy == "intersection":
        fused_pairs = local_pairs & global_pairs
        strategy_desc = "Intersection (dual structure verification)"
    else:
        fused_pairs = local_pairs | global_pairs
        strategy_desc = "Union (diverse structure coverage)"


    with open(fused_output_file, 'w', encoding='utf-8') as f:
        for kg1_id, kg2_id in sorted(fused_pairs):
            f.write(f"{kg1_id}\t{kg2_id}\n")

    print(f"\n[Structural Feature Fusion Statistics]")
    print(f"  Local structural feature results: {len(local_pairs)}")
    print(f"  Global structural embedding results: {len(global_pairs)}")
    print(f"  Fusion strategy: {strategy_desc}")
    print(f"  Fused results: {len(fused_pairs)}")
    print(f"  Fused results saved to: {fused_output_file}")


# 使用示例
if __name__ == "__main__":

    data_dir = "/TEA-RAG/dataset/wiki_for_icews_0.8_3.7_TF/"

    structure_similarity_filter(
        retriever_output_file=os.path.join(data_dir, "message_pool", "retriever_outputs.txt"),
        line_triples_1_file=os.path.join(data_dir, "message_pool", "line_triples_1"),
        line_triples_2_file=os.path.join(data_dir, "message_pool", "line_triples_2"),
        stuc_retriever_output_file=os.path.join(data_dir, "message_pool", "stuc_retriever_output.txt"),
        alpha=0.5,
        threshold=0.26,
        use_edit_distance=False,
        use_enhanced_structure=True,
        enhanced_threshold=0.5
    )
