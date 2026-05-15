from collections import defaultdict


def calculate_entity_performance(output_triples_path, relative_ent_ids_path, ent_ids_2_path):
    """
    Calculate ACC, P, R, F1 for relevant entity discovery
    output_triples_path: Model output fused triples file
    relative_ent_ids_path: Ground truth sum_relative_ent_ids_2
    ent_ids_2_path: KG2 entity ID file
    """
    # Read KG2 entity ID set
    with open(ent_ids_2_path, 'r', encoding='utf-8') as f:
        kg2_ent_ids = {line.strip().split('\t')[0] for line in f if line.strip()}

    # Read KG2 relevant entities from output triples
    predicted_entities = set()
    with open(output_triples_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                head, _, tail = parts[:3]
                if head in kg2_ent_ids:
                    predicted_entities.add(head)
                if tail in kg2_ent_ids:
                    predicted_entities.add(tail)

    # Read ground truth set (positive examples)
    with open(relative_ent_ids_path, 'r', encoding='utf-8') as f:
        gold_entities = {line.strip() for line in f if line.strip()}

    # Negative example set: entities in KG2 that do not belong to gold_entities
    negative_entities = kg2_ent_ids - gold_entities

    # Calculate TP, FP, TN, FN
    tp = len(predicted_entities & gold_entities)  # Predicted as positive, actually positive
    fp = len(predicted_entities & negative_entities)  # Predicted as positive, actually negative
    fn = len(gold_entities - predicted_entities)  # Predicted as negative, actually positive
    tn = len(negative_entities - predicted_entities)  # Predicted as negative, actually negative

    # Calculate ACC, P, R, F1
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    print(f"Entity Discovery Performance:")
    print(f"  ACC={accuracy:.4f}, P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}")
    print(f"  TP={tp}, FP={fp}, TN={tn}, FN={fn}")

    return accuracy, precision, recall, f1


def calculate_fusion_performance(output_triples_path, gold_triples_path, false_triples_path):
    """
    Calculate ACC, P, R, F1 for fused triples
    output_triples_path: Model output fused triples file
    gold_triples_path: Ground truth sum_triples_1_fusion
    false_triples_path: Negative examples false_sum_triples_fusion
    """
    # Read prediction results
    with open(output_triples_path, 'r', encoding='utf-8') as f:
        predicted_triples = {line.strip() for line in f if line.strip()}

    # Read positive examples (ground truth)
    with open(gold_triples_path, 'r', encoding='utf-8') as f:
        gold_triples = {line.strip() for line in f if line.strip()}

    # Read negative examples
    with open(false_triples_path, 'r', encoding='utf-8') as f:
        false_triples = {line.strip() for line in f if line.strip()}

    # Calculate TP, FP, TN, FN
    tp = len(predicted_triples & gold_triples)
    fp = len(predicted_triples & false_triples)
    fn = len(gold_triples - predicted_triples)
    tn = len(false_triples - predicted_triples)

    # ACC、P、R、F1
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    print(f"Fusion Triple Performance:")
    print(f"  ACC={accuracy:.4f}, P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}")
    print(f"  TP={tp}, FP={fp}, TN={tn}, FN={fn}")

    return accuracy, precision, recall, f1


if __name__ == "__main__":
    output_triples_fusion = '/home/dex/Desktop/SKGF/TEA-RAG/dataset/output_0902/graphormer_W_I_S1_v2.txt'#预测的结果文件
    sum_relative_ent_ids_2 = '/home/dex/Desktop/SKGF/TEA-RAG/dataset/fusion_icews_wiki_S1_del0.95_8_2/ref_relative_ent_ids_2'
    sum_triples_1_fusion = '/home/dex/Desktop/SKGF/TEA-RAG/dataset/fusion_icews_wiki_S1_del0.95_8_2/ref_triples_1_fusion'
    false_sum_triples_fusion = '/home/dex/Desktop/SKGF/TEA-RAG/dataset/fusion_icews_wiki_S1_del0.95_8_2/false_ref_triples_1_fusion'
    ent_ids_2 = '/home/dex/Desktop/SKGF/TEA-RAG/dataset/fusion_icews_wiki_S1_del0.95_8_2/ent_ids_2'


    # Calculate relevant entity discovery performance
    entity_acc, entity_precision, entity_recall, entity_f1 = calculate_entity_performance(
        output_triples_fusion, sum_relative_ent_ids_2, ent_ids_2
    )

    # Calculate fused triple performance
    fusion_acc, fusion_precision, fusion_recall, fusion_f1 = calculate_fusion_performance(
        output_triples_fusion, sum_triples_1_fusion, false_sum_triples_fusion
    )
