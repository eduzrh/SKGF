# -*- coding: utf-8 -*-
"""
Improvements:
1. Dynamic Negative Sample Generation
2. Model Output Standardization
3. Adaptive Threshold
4. Balanced Training Strategy
"""
import os
import random
import argparse
from typing import List, Tuple, Dict
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm



class StructuralDistanceAnalyzer(nn.Module):
    """Structural Distance Analyzer"""

    def __init__(self, n_ent: int, n_rel: int, dim: int):
        super().__init__()
        self.ent = nn.Embedding(n_ent, dim)
        self.rel = nn.Embedding(n_rel, dim)
        nn.init.xavier_uniform_(self.ent.weight)
        nn.init.xavier_uniform_(self.rel.weight)

    def forward(self, h, r, t):
        # Convert to similarity score: smaller distance, higher score
        dist = torch.norm(self.ent(h) + self.rel(r) - self.ent(t), p=1, dim=1)
        # Use negative exponential function to convert distance to similarity between 0-1
        return torch.exp(-dist / 10.0)  # Adjustable scaling factor


class HyperplaneProjectionAnalyzer(nn.Module):
    """Hyperplane Projection Analyzer"""

    def __init__(self, n_ent: int, n_rel: int, dim: int):
        super().__init__()
        self.ent = nn.Embedding(n_ent, dim)
        self.rel = nn.Embedding(n_rel, dim)
        self.norm = nn.Embedding(n_rel, dim)
        nn.init.xavier_uniform_(self.ent.weight)
        nn.init.xavier_uniform_(self.rel.weight)
        nn.init.xavier_uniform_(self.norm.weight)

    def _proj(self, e, norm):
        return e - torch.sum(e * norm, dim=1, keepdim=True) * norm

    def forward(self, h, r, t):
        n = F.normalize(self.norm(r), p=2, dim=1)
        h_e = self._proj(self.ent(h), n)
        t_e = self._proj(self.ent(t), n)
        r_e = self.rel(r)
        dist = torch.norm(h_e + r_e - t_e, p=1, dim=1)
        # Convert to similarity score
        return torch.exp(-dist / 10.0)


class BilinearInteractionAnalyzer(nn.Module):
    """Bilinear Interaction Analyzer"""

    def __init__(self, n_ent: int, n_rel: int, dim: int):
        super().__init__()
        self.ent = nn.Embedding(n_ent, dim)
        self.rel = nn.Embedding(n_rel, dim)
        nn.init.xavier_uniform_(self.ent.weight)
        nn.init.xavier_uniform_(self.rel.weight)

    def forward(self, h, r, t):
        # Original score
        score = torch.sum(self.ent(h) * self.rel(r) * self.ent(t), dim=1)
        # Use sigmoid to map score to 0-1
        return torch.sigmoid(score)


class ComplexValuedAnalyzer(nn.Module):
    """Complex-Valued Vector Analyzer"""

    def __init__(self, n_ent: int, n_rel: int, dim: int):
        super().__init__()
        self.ent_re = nn.Embedding(n_ent, dim)
        self.ent_im = nn.Embedding(n_ent, dim)
        self.rel_re = nn.Embedding(n_rel, dim)
        self.rel_im = nn.Embedding(n_rel, dim)
        for emb in [self.ent_re, self.ent_im, self.rel_re, self.rel_im]:
            nn.init.xavier_uniform_(emb.weight)

    def forward(self, h, r, t):
        h_re, h_im = self.ent_re(h), self.ent_im(h)
        r_re, r_im = self.rel_re(r), self.rel_im(r)
        t_re, t_im = self.ent_re(t), self.ent_im(t)
        score = torch.sum(
            h_re * r_re * t_re +
            h_im * r_re * t_im +
            h_re * r_im * t_im -
            h_im * r_im * t_re,
            dim=1
        )
        return torch.sigmoid(score)




#         Utility Functions


Triple5 = Tuple[int, int, int, int, int]


def set_seed(seed: int = 2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def read_triples(path: str) -> List[Triple5]:
    rows: List[Triple5] = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) != 5:
                parts = s.split('\t')
            if len(parts) != 5:
                raise ValueError(f"Line format error (requires 5 columns): {s}")
            h, r, t, ts1, ts2 = map(int, parts)
            rows.append((h, r, t, ts1, ts2))
    return rows


def build_ent_rel_mappings(data_dir: str) -> Tuple[Dict[int, int], Dict[int, int]]:
    ent_set, rel_set = set(), set()

    for fn in [
        'triples_1', 'triples_2',
        'sup_triples_1_fusion',
        'ref_fusion_triples_1', 'false_ref_fusion_triples_1_fusion',
        'ent_ids_1', 'ent_ids_2',
        'rel_ids_1', 'rel_ids_2'
    ]:
        path = os.path.join(data_dir, fn)
        if not os.path.exists(path):
            continue
        with open(path, encoding='utf-8') as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                parts = s.split()
                if fn.startswith("ent_ids"):
                    try:
                        ent_set.add(int(parts[0]))
                    except:
                        continue
                elif fn.startswith("rel_ids"):
                    try:
                        rel_set.add(int(parts[0]))
                    except:
                        continue
                elif len(parts) == 5:
                    try:
                        h, r, t, _, _ = map(int, parts)
                        ent_set.update([h, t])
                        rel_set.add(r)
                    except:
                        continue

    ent_list = sorted(ent_set)
    rel_list = sorted(rel_set)
    ent2dense = {ent: i for i, ent in enumerate(ent_list)}
    rel2dense = {rel: i for i, rel in enumerate(rel_list)}
    return ent2dense, rel2dense


def generate_negative_samples(pos_triples: List[Triple5],
                              ent2dense: Dict[int, int],
                              rel2dense: Dict[int, int],
                              neg_ratio: float = 1.0) -> List[Triple5]:

    entities = list(ent2dense.keys())
    relations = list(rel2dense.keys())
    pos_set = {(h, r, t) for h, r, t, _, _ in pos_triples}

    neg_triples = []
    target_count = int(len(pos_triples) * neg_ratio)

    print(f"[Negative Sample Generation] Target count: {target_count}")

    attempts = 0
    max_attempts = target_count * 10 # Avoid infinite loop

    while len(neg_triples) < target_count and attempts < max_attempts:
        attempts += 1

        # Randomly select a positive sample as base
        base_triple = random.choice(pos_triples)
        h, r, t, ts1, ts2 = base_triple

        # Randomly select corruption strategy: head entity, relation, or tail entity
        strategy = random.choice(['head', 'relation', 'tail'])

        if strategy == 'head':
            new_h = random.choice(entities)
            new_triple = (new_h, r, t)
        elif strategy == 'relation':
            new_r = random.choice(relations)
            new_triple = (h, new_r, t)
        else:  # tail
            new_t = random.choice(entities)
            new_triple = (h, r, new_t)

        # Ensure generated sample is truly negative
        if new_triple not in pos_set:
            neg_triples.append((new_triple[0], new_triple[1], new_triple[2], ts1, ts2))

    print(f"[Negative Sample Generation] Actually generated: {len(neg_triples)}")
    return neg_triples


def batchify(lst: List[int], bs: int) -> List[List[int]]:
    return [lst[i:i + bs] for i in range(0, len(lst), bs)]


# =========================
#       Training / Prediction
# =========================
def train_one_model(model: nn.Module,
                    pos_data: List[Triple5],
                    ent2dense: Dict[int, int],
                    rel2dense: Dict[int, int],
                    epochs: int = 50,
                    batch_size: int = 1024,
                    lr: float = 1e-3,
                    neg_ratio: float = 1.0,
                    device: str = "cpu"):
    model.to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"[Training] Positive samples: {len(pos_data)}")

    for ep in range(1, epochs + 1):
        neg_data = generate_negative_samples(pos_data, ent2dense, rel2dense, neg_ratio)

        train_triples = pos_data + neg_data
        labels = [1] * len(pos_data) + [0] * len(neg_data)
        idxs = list(range(len(train_triples)))

        random.shuffle(idxs)
        total_loss = 0.0
        correct = 0
        total = 0

        for chunk in tqdm(batchify(idxs, batch_size), desc=f"Epoch {ep}/{epochs}", ncols=100):
            batch = [train_triples[i] for i in chunk]
            y_raw = [labels[i] for i in chunk]

            kept = [(h, r, t, lab) for (h, r, t, _, _), lab in zip(batch, y_raw)
                    if (h in ent2dense and t in ent2dense and r in rel2dense)]

            if not kept:
                continue

            h = torch.tensor([ent2dense[x[0]] for x in kept], device=device)
            r = torch.tensor([rel2dense[x[1]] for x in kept], device=device)
            t = torch.tensor([ent2dense[x[2]] for x in kept], device=device)
            y = torch.tensor([x[3] for x in kept], dtype=torch.float32, device=device)


            probs = model(h, r, t)


            loss = F.binary_cross_entropy(probs, y)


            pred = (probs > 0.5).float()
            correct += (pred == y).sum().item()
            total += len(y)

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

        acc = correct / total if total > 0 else 0
        print(f"[Epoch {ep}] loss={total_loss:.4f}, acc={acc:.4f}")


def calculate_adaptive_threshold(model: nn.Module,
                                 validation_data: List[Triple5],
                                 ent2dense: Dict[int, int],
                                 rel2dense: Dict[int, int],
                                 device: str = "cpu") -> float:
    """Calculate adaptive threshold based on validation set"""
    model.eval()
    scores = []

    with torch.no_grad():
        for h, r, t, _, _ in validation_data:
            if (h in ent2dense) and (t in ent2dense) and (r in rel2dense):
                hh = torch.tensor([ent2dense[h]], device=device)
                rr = torch.tensor([rel2dense[r]], device=device)
                tt = torch.tensor([ent2dense[t]], device=device)
                prob = model(hh, rr, tt).item()
                scores.append(prob)

    if not scores:
        return 0.5

    # Use quantile as threshold
    scores = np.array(scores)
    thresholds = [np.percentile(scores, p) for p in [50, 60, 70, 75, 80, 85, 90]]

    print(f"[阈值候选] 50%={thresholds[0]:.4f}, 60%={thresholds[1]:.4f}, "
          f"70%={thresholds[2]:.4f}, 75%={thresholds[3]:.4f}, "
          f"80%={thresholds[4]:.4f}, 85%={thresholds[5]:.4f}, 90%={thresholds[6]:.4f}")

    # Select 75th percentile as default threshold
    return thresholds[3]


def predict_and_write(model: nn.Module,
                      test_data: List[Triple5],
                      ent2dense: Dict[int, int],
                      rel2dense: Dict[int, int],
                      threshold: float,
                      output_path: str,
                      device: str = "cpu"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model.to(device)
    model.eval()

    kept: List[Triple5] = []
    all_scores = []
    cnt_all = 0
    cnt_skipped = 0

    with torch.no_grad():
        for h, r, t, ts1, ts2 in tqdm(test_data, desc="Predicting", ncols=100):
            cnt_all += 1
            if (h not in ent2dense) or (t not in ent2dense) or (r not in rel2dense):
                cnt_skipped += 1
                continue

            hh = torch.tensor([ent2dense[h]], device=device)
            rr = torch.tensor([rel2dense[r]], device=device)
            tt = torch.tensor([ent2dense[t]], device=device)
            prob = model(hh, rr, tt).item()
            all_scores.append(prob)

            if prob >= threshold:
                kept.append((h, r, t, ts1, ts2))


    with open(output_path, "w", encoding="utf-8") as f:
        for row in kept:
            f.write("\t".join(map(str, row)) + "\n")


    # Print statistics
    print(f"[Prediction Results] Total={cnt_all}, Skipped={cnt_skipped}, Retained={len(kept)}")
    if all_scores:
        scores_array = np.array(all_scores)
        print(f"[Score Statistics] min={scores_array.min():.4f}, max={scores_array.max():.4f}, "
              f"mean={scores_array.mean():.4f}, std={scores_array.std():.4f}")
        print(f"[Quantiles] 25%={np.percentile(scores_array, 25):.4f}, "
              f"50%={np.percentile(scores_array, 50):.4f}, "
              f"75%={np.percentile(scores_array, 75):.4f}, "
              f"90%={np.percentile(scores_array, 90):.4f}")
    else:
        print("[Score Statistics] No valid scores")


# =========================
#          Main Program
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,
                        choices=["structural_distance", "hyperplane_projection",
                                 "bilinear_interaction", "complex_valued"],
                        default="structural_distance")
    parser.add_argument("--dim", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--neg_ratio", type=float, default=1.0, help="Negative to positive sample ratio")
    parser.add_argument("--threshold", type=float, default=0.5, help="Prediction threshold, None for automatic calculation")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--data_dir", type=str, default="./dataset/W-I-S1")
    parser.add_argument("--output_path", type=str,
                        default="./dataset/W-I-S1/message_pool/pred_structural_analysis.txt")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)


    ent2dense, rel2dense = build_ent_rel_mappings(args.data_dir)
    print(f"[Mappings] Entity count={len(ent2dense)}, Relation count={len(rel2dense)}")

    model_map = {
        "structural_distance": StructuralDistanceAnalyzer,
        "hyperplane_projection": HyperplaneProjectionAnalyzer,
        "bilinear_interaction": BilinearInteractionAnalyzer,
        "complex_valued": ComplexValuedAnalyzer
    }
    model = model_map[args.model](len(ent2dense), len(rel2dense), args.dim)
    print(f"[Model] Using {args.model.upper()}, dimension={args.dim}")


    pos_triples = read_triples(os.path.join(args.data_dir, "sup_triples_1_fusion"))
    print(f"[Training Data] Positive samples={len(pos_triples)}")

    train_one_model(model, pos_triples, ent2dense, rel2dense,
                    epochs=args.epochs, batch_size=args.batch_size,
                    lr=args.lr, neg_ratio=args.neg_ratio, device=str(device))


    test_pos = read_triples(os.path.join(args.data_dir, "ref_triples_1_fusion"))
    test_neg = read_triples(os.path.join(args.data_dir, "false_ref_triples_1_fusion"))
    test_all = test_pos + test_neg
    print(f"[Test Data] Total={len(test_all)} (Positive={len(test_pos)}, Negative={len(test_neg)})")


    if args.threshold is None:
        print("[Threshold] Calculating adaptive threshold...")
        # 使用部分测试数据作为验证集
        val_data = random.sample(test_all, min(1000, len(test_all)))
        threshold = calculate_adaptive_threshold(model, val_data, ent2dense, rel2dense, str(device))
        print(f"[Threshold] Adaptive threshold={threshold:.4f}")
    else:
        threshold = args.threshold
        print(f"[Threshold] Manual threshold={threshold:.4f}")


    predict_and_write(model, test_all, ent2dense, rel2dense,
                      threshold=threshold, output_path=args.output_path, device=str(device))


if __name__ == "__main__":
    main()
