import os

from collections import defaultdict


def generate_line_triples_and_names(
        triples_path_1, triples_path_2,
        ent_ids_path_1, ent_ids_path_2,
        rel_ids_path_1, rel_ids_path_2,
        output_line_triples_1, output_line_triples_2,
        output_line_triples_name_1, output_line_triples_name_2
):
    """
    Convert two KG triple files into continuously numbered line_triples files,
    and generate line_triples_name files based on entity and relation names.
    KG2 triple IDs continue numbering from KG1 IDs.
    Entity/relation lookup performs cross-KG mapping queries.
    """

    def load_mapping(path):
        mapping = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    mapping[parts[0]] = parts[1]
        return mapping

    # Read entity and relation mappings
    ent_map_1 = load_mapping(ent_ids_path_1)
    ent_map_2 = load_mapping(ent_ids_path_2)
    rel_map_1 = load_mapping(rel_ids_path_1)
    rel_map_2 = load_mapping(rel_ids_path_2)

    def get_entity_name(entity_id, primary_map, secondary_map):
        """First search in primary mapping, if not found then search in secondary mapping"""
        if entity_id in primary_map:
            return primary_map[entity_id]
        elif entity_id in secondary_map:
            return secondary_map[entity_id]
        else:
            return f"UNK_{entity_id}"

    def get_relation_name(relation_id, primary_map, secondary_map):
        """First search in primary mapping, if not found then search in secondary mapping"""
        if relation_id in primary_map:
            return primary_map[relation_id]
        elif relation_id in secondary_map:
            return secondary_map[relation_id]
        else:
            return f"UNK_{relation_id}"

    def process_triples(triples_path, ent_map_primary, ent_map_secondary,
                        rel_map_primary, rel_map_secondary,
                        start_idx, out_line_path, out_name_path):
        line_triples = []
        line_triples_names = []
        with open(triples_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                head_id, rel_id, tail_id = parts[0], parts[1], parts[2]

                rest = parts[3:] if len(parts) > 3 else []
                triple_id = start_idx + i

                line_triples.append(f"{triple_id}\t{head_id}\t{rel_id}\t{tail_id}\t{' '.join(rest)}\n")


                head_name = get_entity_name(head_id, ent_map_primary, ent_map_secondary)
                rel_name = get_relation_name(rel_id, rel_map_primary, rel_map_secondary)
                tail_name = get_entity_name(tail_id, ent_map_primary, ent_map_secondary)

                line_triples_names.append(f"{triple_id}\t{head_name}|{rel_name}|{tail_name}\n")


        with open(out_line_path, "w", encoding="utf-8") as f_out:
            f_out.writelines(line_triples)
        with open(out_name_path, "w", encoding="utf-8") as f_out:
            f_out.writelines(line_triples_names)

        print(f"Generated {out_line_path} with {len(line_triples)} entries")
        print(f"Generated {out_name_path} with {len(line_triples_names)} entries")

        return start_idx + len(line_triples)

    print(f"Line Graph Trans...")
    next_id = process_triples(
        triples_path_1, ent_map_1, ent_map_2, rel_map_1, rel_map_2,
        0, output_line_triples_1, output_line_triples_name_1
    )

    _ = process_triples(
        triples_path_2, ent_map_2, ent_map_1, rel_map_2, rel_map_1,
        next_id, output_line_triples_2, output_line_triples_name_2
    )

    print(f"Two KG triple IDs have been continuously numbered, KG1: 0 ~ {next_id - 1}, KG2 starts from {next_id}")


def convert_triples_to_entities_2(
        retriever_output_path,
        line_triples_1_path,
        line_triples_2_path,
        entity_retriever_output_path,
        ent_ids_1_file
):

    kg1_whitelist = set()
    with open(ent_ids_1_file, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            ent_id = s.split()[0]
            kg1_whitelist.add(ent_id)

    # 1)  line_triples_1 => {tid: (head, tail)}
    triples1 = {}
    with open(line_triples_1_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 4:
                continue
            tid, head, rel, tail = parts[0], parts[1], parts[2], parts[3]
            triples1[tid] = (head, tail)

    # 2) 读取 line_triples_2 => {tid: (head, tail)}
    triples2 = {}
    with open(line_triples_2_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 4:
                continue
            tid, head, rel, tail = parts[0], parts[1], parts[2], parts[3]
            triples2[tid] = (head, tail)


    pair_set = set()
    bad_cols = miss_kg1 = miss_kg2 = kept = only_nonKG1 = 0

    with open(retriever_output_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                bad_cols += 1
                continue
            kg1_tid, kg2_tid = parts


            if kept < 5:
                print(f"P: kg1_tid={kg1_tid}, kg2_tid={kg2_tid}")
                print(f"  kg1_tid in triples1 : {kg1_tid in triples1}")
                print(f"  kg2_tid in triples2 : {kg2_tid in triples2}")

            if kg1_tid not in triples1:
                miss_kg1 += 1
                continue
            if kg2_tid not in triples2:
                miss_kg2 += 1
                continue

            kg1_head, kg1_tail = triples1[kg1_tid]
            kg2_head, kg2_tail = triples2[kg2_tid]


            candidates_from_kg1 = []
            if kg1_head in kg1_whitelist:
                candidates_from_kg1.append(kg1_head)
            if kg1_tail in kg1_whitelist:
                candidates_from_kg1.append(kg1_tail)

            if not candidates_from_kg1:
                only_nonKG1 += 1
                continue

            for e1 in candidates_from_kg1:
                pair_set.add((e1, kg2_head))
                pair_set.add((e1, kg2_tail))

            kept += 1


    def try_int(x):
        try:
            return int(x)
        except:
            return x

    sorted_pairs = sorted(pair_set, key=lambda x: (try_int(x[0]), try_int(x[1])))
    with open(entity_retriever_output_path, "w", encoding="utf-8") as out:
        for e1, e2 in sorted_pairs:
            out.write(f"{e1}\t{e2}\n")

    print(f"[DEBUG] bad_cols={bad_cols}, miss_kg1={miss_kg1}, miss_kg2={miss_kg2}, "
          f"kept_pairs_from_retriever={kept}, skipped_nonKG1_in_line_triples_1={only_nonKG1}, "
          f"unique_entity_pairs={len(pair_set)}")

    return len(pair_set)

def convert_triples_to_entities(
    retriever_output_path,
    line_triples_1_path,
    line_triples_2_path,
    entity_retriever_output_path
):
    # 1. read line_triples_1
    triples1 = {}
    with open(line_triples_1_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            tid, head, rel, tail = parts[0], parts[1], parts[2], parts[3]
            triples1[tid] = (head, tail)

    # 2. read line_triples_2
    triples2 = {}
    with open(line_triples_2_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            tid, head, rel, tail = parts[0], parts[1], parts[2], parts[3]
            triples2[tid] = (head, tail)

    # 3. read retriever_output and transfer
    pair_set = set()
    with open(retriever_output_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            kg1_tid, kg2_tid = parts
            if kg1_tid not in triples1 or kg2_tid not in triples2:
                continue
            kg1_head, kg1_tail = triples1[kg1_tid]
            kg2_head, kg2_tail = triples2[kg2_tid]


            pair_set.add((kg1_head, kg2_head))
            pair_set.add((kg1_head, kg2_tail))
            pair_set.add((kg1_tail, kg2_head))
            pair_set.add((kg1_tail, kg2_tail))


    sorted_pairs = sorted(pair_set, key=lambda x: (int(x[0]), int(x[1])))


    with open(entity_retriever_output_path, "w", encoding="utf-8") as out:
        for e1, e2 in sorted_pairs:
            out.write(f"{e1}\t{e2}\n")

def generate_3triples(relevance_file, rel_ids_file, output_file):

    with open(rel_ids_file, 'r', encoding='utf-8') as f:
        rel_ids = [line.strip().split()[0] for line in f if line.strip()]

    with open(relevance_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:

        for line in f_in:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            kg1, kg2 = parts

            for rel_id in rel_ids:
                f_out.write(f"{kg1}\t{rel_id}\t{kg2}\n")
                f_out.write(f"{kg2}\t{rel_id}\t{kg1}\n")


def filter_relevant_triples(false_ref_file, ref_file, relevance_file, output_file):
    """
    Filter qualified quadruples

    Args:
        false_ref_file: false_ref_triples_1_fusion file path
        ref_file: ref_triples_1_fusion file path
        relevance_file: relevance_entities.txt file path
        output_file: temp_output_triples_fusion.txt output file path
    """

    # Read relevance_entities.txt, build entity relation set
    relevance_pairs = set()
    with open(relevance_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split('\t')
                if len(parts) == 2:
                    entity1, entity2 = parts[0], parts[1]
                    print((entity1, entity2))
                    # 添加双向关系
                    relevance_pairs.add((entity1, entity2))
                    relevance_pairs.add((entity2, entity1))

    print(f"读取到 {len(relevance_pairs) // 2} 对关联实体")

    # Filter qualified triples
    filtered_triples = set()

    # Process false_ref_triples_1_fusion
    with open(false_ref_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split('\t')
                if len(parts) == 5:
                    head_entity, relation, tail_entity, time1, time2 = parts
                    # Check if head and tail entities are in relevance_pairs
                    if (head_entity, tail_entity) in relevance_pairs or (tail_entity, head_entity) in relevance_pairs:
                        filtered_triples.add(line)

    # Process ref_triples_1_fusion
    with open(ref_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split('\t')
                if len(parts) == 5:
                    head_entity, relation, tail_entity, time1, time2 = parts
                    # Check if head and tail entities are in relevance_pairs
                    if (head_entity, tail_entity) in relevance_pairs or (tail_entity, head_entity) in relevance_pairs:
                        print(line)
                        filtered_triples.add(line)

    # Write to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        for triple in filtered_triples:
            f.write(triple + '\n')

    print(f"Filtering completed, saved {len(filtered_triples)} qualified triples to {output_file}")

    return len(filtered_triples)

def rank_and_group_triples(false_ref_file, ref_file, temp_output_file, rank_output_file):
    """
    Sort quadruples by group and save

    Args:
        false_ref_file: false_ref_triples_1_fusion file path
        ref_file: ref_triples_1_fusion file path
        temp_output_file: temp_output_triples_fusion.txt file path
        rank_output_file: rank_temp_output_triples_fusion.txt output file path
    """

    # 1. Read and merge false_ref and ref files
    sum_ref_triples = set()

    # Read false_ref_triples_1_fusion
    with open(false_ref_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                sum_ref_triples.add(line)

    # Read ref_triples_1_fusion
    with open(ref_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                sum_ref_triples.add(line)

    print(f"Merged sum_ref_triples_1_fusion has {len(sum_ref_triples)} quadruples")

    # 2. Convert sum_ref_triples to search-friendly data structure
    # Use (relation_id, timestamp_id1, timestamp_id2, head_entity_id) and (relation_id, timestamp_id1, timestamp_id2, tail_entity_id) as keys
    ref_dict_by_head = {}  # key: (rel, time1, time2, head), value: set of full triples
    ref_dict_by_tail = {}  # key: (rel, time1, time2, tail), value: set of full triples

    for triple_line in sum_ref_triples:
        parts = triple_line.split('\t')
        if len(parts) == 5:
            head, rel, tail, time1, time2 = parts


            key_head = (rel, time1, time2, head)
            if key_head not in ref_dict_by_head:
                ref_dict_by_head[key_head] = set()
            ref_dict_by_head[key_head].add(triple_line)


            key_tail = (rel, time1, time2, tail)
            if key_tail not in ref_dict_by_tail:
                ref_dict_by_tail[key_tail] = set()
            ref_dict_by_tail[key_tail].add(triple_line)

    # 3. Read temp_output_triples_fusion.txt and group
    temp_triples = []
    with open(temp_output_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                temp_triples.append(line)

    print(f"temp_output_triples_fusion.txt has {len(temp_triples)} quadruples")

    # 4. Find matching groups for each temp quadruple
    groups = []
    processed_triples = set()

    for temp_triple in temp_triples:
        if temp_triple in processed_triples:
            continue

        parts = temp_triple.split('\t')
        if len(parts) != 5:
            continue

        head, rel, tail, time1, time2 = parts


        group_triples = set()
        group_triples.add(temp_triple)


        key_head = (rel, time1, time2, head)
        if key_head in ref_dict_by_head:
            group_triples.update(ref_dict_by_head[key_head])


        key_tail = (rel, time1, time2, tail)
        if key_tail in ref_dict_by_tail:
            group_triples.update(ref_dict_by_tail[key_tail])


        for triple in group_triples:
            processed_triples.add(triple)


        sorted_group = sorted(list(group_triples))
        groups.append(sorted_group)


    groups.sort(key=lambda x: (len(x), x[0] if x else ""))


    all_output_triples = []
    for group in groups:
        all_output_triples.extend(group)

    with open(rank_output_file, 'w', encoding='utf-8') as f:
        for triple in all_output_triples:
            f.write(triple + '\n')


    group_count = len(groups)
    groups_with_1 = len([g for g in groups if len(g) == 1])
    groups_with_2 = len([g for g in groups if len(g) == 2])
    groups_with_3_plus = len([g for g in groups if len(g) >= 3])

    print(f"\n=== Group Statistics ===")
    print(f"Total groups: {group_count}")
    print(f"Groups with 1 quadruple: {groups_with_1}")
    print(f"Groups with 2 quadruples: {groups_with_2}")
    print(f"Groups with 3 or more quadruples: {groups_with_3_plus}")
    print(f"Total quadruples output to {rank_output_file}: {len(all_output_triples)}")

    return group_count, groups_with_1, groups_with_2, groups_with_3_plus



if __name__ == "__main__":
    data_dir = "/TEA-RAG/dataset/wiki_for_icews_0.8_3.7_TF"



    count = filter_relevant_triples(
        os.path.join(data_dir, "false_ref_triples_1_fusion"),
        os.path.join(data_dir, "ref_triples_1_fusion"),
        os.path.join(data_dir, "message_pool", "relevance_entities.txt"),
        os.path.join(data_dir, "message_pool", "temp_output_triples_fusion.txt")
    )



    # generate_3triples(
    #     relevance_file=os.path.join(data_dir, "message_pool", "relevance_entities.txt"),
    #     rel_ids_file=os.path.join(data_dir, "rel_ids_1"),
    #     output_file=os.path.join(data_dir, "message_pool", "temp_zuhe_triples.txt")
    # )