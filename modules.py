import os


def extract_relevant_entities_and_triples(
        false_ref_triples_file,
        ref_triples_file,
        ent_ids_1_file,
        ent_ids_2_file,
        triples_1_file,
        triples_2_file,
        output_dir="output"
):
    """
    Extract relevant entities and partition test set entities and triples

    Args:
        false_ref_triples_file: Path to false_ref_triples_1_fusion file
        ref_triples_file: Path to ref_triples_1_fusion file
        ent_ids_1_file: Path to ent_ids_1 file
        ent_ids_2_file: Path to ent_ids_2 file
        triples_1_file: Path to triples_1 file
        triples_2_file: Path to triples_2 file
        output_dir: Output directory
    """

    os.makedirs(output_dir, exist_ok=True)


    relative_ent = set()

    print("Reading false_ref_triples_1_fusion...")
    with open(false_ref_triples_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split('\t')
                if len(parts) >= 3:
                    head_ent = parts[0]
                    tail_ent = parts[2]
                    relative_ent.add(head_ent)
                    relative_ent.add(tail_ent)

    print("Reading ref_triples_1_fusion...")
    with open(ref_triples_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split('\t')
                if len(parts) >= 3:
                    head_ent = parts[0]
                    tail_ent = parts[2]
                    relative_ent.add(head_ent)
                    relative_ent.add(tail_ent)

    print(f"Extracted {len(relative_ent)} relevant entities")

    # 2. Read KG1 entities and filter ref_ent_1
    print("Filtering ref_ent_1...")
    ref_ent_1 = set()
    ent_1_data = []

    with open(ent_ids_1_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split('\t', 1)
                if len(parts) >= 2:
                    ent_id = parts[0]
                    ent_name = parts[1]
                    if ent_id in relative_ent:
                        ref_ent_1.add(ent_id)
                        ent_1_data.append((ent_id, ent_name))

    print(f"Found {len(ref_ent_1)} relevant entities in KG1")

    # 3. Read KG2 entities and filter ref_ent_2
    print("Filtering ref_ent_2...")
    ref_ent_2 = set()
    ent_2_data = []

    with open(ent_ids_2_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split('\t', 1)
                if len(parts) >= 2:
                    ent_id = parts[0]
                    ent_name = parts[1]
                    if ent_id in relative_ent:
                        ref_ent_2.add(ent_id)
                        ent_2_data.append((ent_id, ent_name))

    print(f"Found {len(ref_ent_2)} relevant entities in KG2")

    # 4. Filter triples_1 to get ref_triples_1
    print("Filtering ref_triples_1...")
    ref_triples_1 = []

    with open(triples_1_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split('\t')
                if len(parts) >= 5:
                    head_ent = parts[0]
                    rel = parts[1]
                    tail_ent = parts[2]
                    time1 = parts[3]
                    time2 = parts[4]

                    # 如果头实体或尾实体属于ref_ent_1，则保存
                    if head_ent in ref_ent_1 or tail_ent in ref_ent_1:
                        ref_triples_1.append((head_ent, rel, tail_ent, time1, time2))

    print(f"Filtered {len(ref_triples_1)} relevant triples from KG1")

    # 5. Filter triples_2 to get ref_triples_2
    print("Filtering ref_triples_2...")
    ref_triples_2 = []

    with open(triples_2_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split('\t')
                if len(parts) >= 5:
                    head_ent = parts[0]
                    rel = parts[1]
                    tail_ent = parts[2]
                    time1 = parts[3]
                    time2 = parts[4]


                    if head_ent in ref_ent_2 or tail_ent in ref_ent_2:
                        ref_triples_2.append((head_ent, rel, tail_ent, time1, time2))

    print(f"Filtered {len(ref_triples_2)} relevant triples from KG2")


    with open(os.path.join(output_dir, 'ref_ent_1'), 'w', encoding='utf-8') as f:
        for ent_id, ent_name in ent_1_data:
            f.write(f"{ent_id}\t{ent_name}\n")

    with open(os.path.join(output_dir, 'ref_ent_2'), 'w', encoding='utf-8') as f:
        for ent_id, ent_name in ent_2_data:
            f.write(f"{ent_id}\t{ent_name}\n")

    with open(os.path.join(output_dir, 'ref_triples_1'), 'w', encoding='utf-8') as f:
        for head, rel, tail, time1, time2 in ref_triples_1:
            f.write(f"{head}\t{rel}\t{tail}\t{time1}\t{time2}\n")

    with open(os.path.join(output_dir, 'ref_triples_2'), 'w', encoding='utf-8') as f:
        for head, rel, tail, time1, time2 in ref_triples_2:
            f.write(f"{head}\t{rel}\t{tail}\t{time1}\t{time2}\n")


    print("\n=== Processing completed ===")
    print(f"Total relevant entities: {len(relative_ent)}")
    print(f"KG1 relevant entities: {len(ref_ent_1)}")
    print(f"KG2 relevant entities: {len(ref_ent_2)}")
    print(f"KG1 relevant triples: {len(ref_triples_1)}")
    print(f"KG2 relevant triples: {len(ref_triples_2)}")
    print(f"Results saved to {output_dir} directory")


    return {
        'relative_ent': relative_ent,
        'ref_ent_1': ref_ent_1,
        'ref_ent_2': ref_ent_2,
        'ref_triples_1': ref_triples_1,
        'ref_triples_2': ref_triples_2
    }
