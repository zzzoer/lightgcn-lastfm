from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class Config:
    top_k: int = 10
    user_knn_k: int = 40
    item_knn_k: int = 120
    factors: int = 64
    wrmf_factors: int = 16
    wrmf_alpha: float = 5.0
    wrmf_reg: float = 0.10
    wrmf_iterations: int = 5
    bpr_lr: float = 0.01
    bpr_reg: float = 1e-4
    bpr_epochs: int = 300
    lightgcn_lr: float = 0.01
    lightgcn_reg: float = 1e-4
    lightgcn_epochs: int = 800
    lightgcn_layers: int = 2
    eval_every: int = 50
    patience_checks: int = 4


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_lastfm(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, int], Dict[int, int]]:
    interactions = pd.read_csv(data_dir / "user_artists.dat", sep="\t")
    artists = pd.read_csv(data_dir / "artists.dat", sep="\t", encoding="latin-1")
    user_ids = np.sort(interactions["userID"].unique())
    item_ids = np.sort(interactions["artistID"].unique())
    user_map = {int(raw): int(idx) for idx, raw in enumerate(user_ids)}
    item_map = {int(raw): int(idx) for idx, raw in enumerate(item_ids)}
    interactions = interactions.copy()
    interactions["user"] = interactions["userID"].map(user_map).astype(np.int64)
    interactions["item"] = interactions["artistID"].map(item_map).astype(np.int64)
    return interactions, artists, user_map, item_map


def split_per_user(
    interactions: pd.DataFrame,
    n_users: int,
    n_items: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    """80/10/10 per-user split with warm-start catalog repair.

    Each user with at least three interactions contributes at least one validation and
    one test record. If an item would otherwise be absent from training, one of its
    held-out occurrences is moved into training. This prevents impossible cold-item
    test cases without looking at listening-count magnitudes.
    """
    rng = np.random.default_rng(seed)
    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []

    for _, group in interactions.groupby("user", sort=False):
        idx = group.index.to_numpy(copy=True)
        rng.shuffle(idx)
        n = len(idx)
        if n < 3:
            train_idx.extend(idx.tolist())
            continue
        n_val = max(1, int(round(0.10 * n)))
        n_test = max(1, int(round(0.10 * n)))
        while n - n_val - n_test < 1:
            if n_val >= n_test and n_val > 1:
                n_val -= 1
            elif n_test > 1:
                n_test -= 1
            else:
                break
        val_idx.extend(idx[:n_val].tolist())
        test_idx.extend(idx[n_val : n_val + n_test].tolist())
        train_idx.extend(idx[n_val + n_test :].tolist())

    train_set = set(train_idx)
    val_set = set(val_idx)
    test_set = set(test_idx)
    train_items = set(interactions.loc[list(train_set), "item"].astype(int))
    moved_to_train = 0
    for item, group in interactions.groupby("item", sort=False):
        if int(item) in train_items:
            continue
        candidates = [int(i) for i in group.index if int(i) in val_set or int(i) in test_set]
        if not candidates:
            continue
        chosen = candidates[int(rng.integers(len(candidates)))]
        if chosen in val_set:
            val_set.remove(chosen)
        else:
            test_set.remove(chosen)
        train_set.add(chosen)
        train_items.add(int(item))
        moved_to_train += 1

    train = interactions.loc[sorted(train_set)].reset_index(drop=True)
    val = interactions.loc[sorted(val_set)].reset_index(drop=True)
    test = interactions.loc[sorted(test_set)].reset_index(drop=True)
    stats = {
        "train_pairs": int(len(train)),
        "validation_pairs": int(len(val)),
        "test_pairs": int(len(test)),
        "catalog_repairs": int(moved_to_train),
        "test_users": int(test["user"].nunique()),
        "catalog_items": int(n_items),
    }
    return train, val, test, stats


def make_csr(frame: pd.DataFrame, n_users: int, n_items: int, weighted: bool = False) -> sp.csr_matrix:
    if weighted:
        values = frame["weight"].to_numpy(np.float32)
    else:
        values = np.ones(len(frame), dtype=np.float32)
    return sp.csr_matrix(
        (values, (frame["user"].to_numpy(), frame["item"].to_numpy())),
        shape=(n_users, n_items),
        dtype=np.float32,
    )


def truth_lists(frame: pd.DataFrame, n_users: int) -> List[np.ndarray]:
    grouped = frame.groupby("user")["item"].apply(lambda x: x.to_numpy(np.int64)).to_dict()
    return [grouped.get(u, np.empty(0, dtype=np.int64)) for u in range(n_users)]


def topk_from_scores(scores: np.ndarray, seen: sp.csr_matrix, k: int) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    for row in range(scores.shape[0]):
        start, end = seen.indptr[row], seen.indptr[row + 1]
        scores[row, seen.indices[start:end]] = -np.inf
    candidate = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    values = np.take_along_axis(scores, candidate, axis=1)
    order = np.argsort(-values, axis=1)
    return np.take_along_axis(candidate, order, axis=1).astype(np.int64)


def rank_all_users(
    score_block: Callable[[np.ndarray], np.ndarray],
    seen: sp.csr_matrix,
    n_users: int,
    k: int,
    block_size: int = 256,
) -> np.ndarray:
    recs = np.empty((n_users, k), dtype=np.int64)
    for start in range(0, n_users, block_size):
        stop = min(start + block_size, n_users)
        rows = np.arange(start, stop)
        block = score_block(rows)
        recs[start:stop] = topk_from_scores(block, seen[start:stop], k)
    return recs


def evaluate_recommendations(
    recs: np.ndarray,
    truth: Sequence[np.ndarray],
    train_popularity: np.ndarray,
    n_users: int,
    n_items: int,
) -> Dict[str, float]:
    recalls: List[float] = []
    ndcgs: List[float] = []
    reciprocal_ranks: List[float] = []
    valid_recs: List[np.ndarray] = []
    for u, relevant in enumerate(truth):
        if len(relevant) == 0:
            continue
        rel_set = set(int(x) for x in relevant)
        row = recs[u]
        hits = np.array([1.0 if int(i) in rel_set else 0.0 for i in row], dtype=np.float64)
        recalls.append(float(hits.sum() / len(rel_set)))
        discounts = 1.0 / np.log2(np.arange(2, len(row) + 2))
        dcg = float((hits * discounts).sum())
        idcg = float(discounts[: min(len(rel_set), len(row))].sum())
        ndcgs.append(dcg / idcg if idcg else 0.0)
        hit_positions = np.flatnonzero(hits)
        reciprocal_ranks.append(1.0 / float(hit_positions[0] + 1) if len(hit_positions) else 0.0)
        valid_recs.append(row)
    flat = np.concatenate(valid_recs) if valid_recs else np.empty(0, dtype=np.int64)
    coverage = float(np.unique(flat).size / n_items) if len(flat) else 0.0
    popularity_probability = (train_popularity + 1.0) / (n_users + 1.0)
    novelty = float(np.mean(-np.log2(popularity_probability[flat]))) if len(flat) else 0.0
    return {
        "Recall@10": float(np.mean(recalls)),
        "nDCG@10": float(np.mean(ndcgs)),
        "MRR@10": float(np.mean(reciprocal_ranks)),
        "Coverage@10": coverage,
        "Novelty@10": novelty,
        "Evaluated users": float(len(recalls)),
    }


def prune_topk(matrix: sp.csr_matrix, k: int) -> sp.csr_matrix:
    matrix = matrix.tocsr()
    indptr = [0]
    indices: List[int] = []
    values: List[float] = []
    for row in range(matrix.shape[0]):
        start, end = matrix.indptr[row], matrix.indptr[row + 1]
        row_values = matrix.data[start:end]
        row_indices = matrix.indices[start:end]
        if len(row_values) > k:
            keep = np.argpartition(row_values, -k)[-k:]
            order = keep[np.argsort(-row_values[keep])]
        else:
            order = np.argsort(-row_values)
        indices.extend(row_indices[order].tolist())
        values.extend(row_values[order].tolist())
        indptr.append(len(indices))
    return sp.csr_matrix(
        (np.asarray(values, dtype=np.float32), np.asarray(indices, dtype=np.int32), np.asarray(indptr, dtype=np.int64)),
        shape=matrix.shape,
    )


def fit_userknn(train: sp.csr_matrix, k: int) -> sp.csr_matrix:
    similarities = cosine_similarity(train, dense_output=False).tocsr().astype(np.float32)
    similarities.setdiag(0.0)
    similarities.eliminate_zeros()
    return prune_topk(similarities, k)


def fit_itemknn(train: sp.csr_matrix, k: int) -> sp.csr_matrix:
    similarities = cosine_similarity(train.T, dense_output=False).tocsr().astype(np.float32)
    similarities.setdiag(0.0)
    similarities.eliminate_zeros()
    return prune_topk(similarities, k)


def sample_negatives(users: np.ndarray, positives: np.ndarray, train: sp.csr_matrix, rng: np.random.Generator) -> np.ndarray:
    negatives = rng.integers(0, train.shape[1], size=len(users), dtype=np.int64)
    invalid = np.asarray(train[users, negatives]).reshape(-1) > 0
    while invalid.any():
        negatives[invalid] = rng.integers(0, train.shape[1], size=int(invalid.sum()), dtype=np.int64)
        invalid = np.asarray(train[users, negatives]).reshape(-1) > 0
    return negatives


class BPRMF(torch.nn.Module):
    def __init__(self, n_users: int, n_items: int, factors: int) -> None:
        super().__init__()
        self.user = torch.nn.Embedding(n_users, factors)
        self.item = torch.nn.Embedding(n_items, factors)
        torch.nn.init.normal_(self.user.weight, std=0.05)
        torch.nn.init.normal_(self.item.weight, std=0.05)


def train_bprmf(
    train: sp.csr_matrix,
    truth_val: Sequence[np.ndarray],
    cfg: Config,
    seed: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    set_seed(seed)
    coo = train.tocoo()
    users_np = coo.row.astype(np.int64)
    items_np = coo.col.astype(np.int64)
    model = BPRMF(train.shape[0], train.shape[1], cfg.factors).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.bpr_lr)
    rng = np.random.default_rng(seed + 101)
    users = torch.as_tensor(users_np, dtype=torch.long, device=device)
    pos = torch.as_tensor(items_np, dtype=torch.long, device=device)
    best_metric = -np.inf
    best_state = None
    best_epoch = 0
    checks_without_gain = 0
    pop = np.asarray(train.sum(axis=0)).reshape(-1)
    for epoch in range(1, cfg.bpr_epochs + 1):
        negatives_np = sample_negatives(users_np, items_np, train, rng)
        neg = torch.as_tensor(negatives_np, dtype=torch.long, device=device)
        pos_score = (model.user(users) * model.item(pos)).sum(dim=1)
        neg_score = (model.user(users) * model.item(neg)).sum(dim=1)
        loss = F.softplus(-(pos_score - neg_score)).mean()
        reg = (
            model.user(users).pow(2).sum(dim=1)
            + model.item(pos).pow(2).sum(dim=1)
            + model.item(neg).pow(2).sum(dim=1)
        ).mean()
        loss = loss + cfg.bpr_reg * reg
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % cfg.eval_every == 0 or epoch == cfg.bpr_epochs:
            user_f = model.user.weight.detach().cpu().numpy()
            item_f = model.item.weight.detach().cpu().numpy()
            recs = rank_all_users(lambda rows: user_f[rows] @ item_f.T, train, train.shape[0], cfg.top_k)
            score = evaluate_recommendations(recs, truth_val, pop, train.shape[0], train.shape[1])["nDCG@10"]
            if score > best_metric + 1e-6:
                best_metric = score
                best_epoch = epoch
                checks_without_gain = 0
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            else:
                checks_without_gain += 1
                if checks_without_gain >= cfg.patience_checks:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    return (
        model.user.weight.detach().cpu().numpy().astype(np.float32),
        model.item.weight.detach().cpu().numpy().astype(np.float32),
        {"best_validation_nDCG@10": float(best_metric), "best_epoch": float(best_epoch)},
    )


def normalized_bipartite_adjacency(train: sp.csr_matrix, device: torch.device) -> torch.Tensor:
    coo = train.tocoo()
    n_users, n_items = train.shape
    user_degree = np.asarray(train.sum(axis=1)).reshape(-1)
    item_degree = np.asarray(train.sum(axis=0)).reshape(-1)
    values = 1.0 / np.sqrt(user_degree[coo.row] * item_degree[coo.col])
    rows = np.concatenate([coo.row, coo.col + n_users])
    cols = np.concatenate([coo.col + n_users, coo.row])
    values = np.concatenate([values, values]).astype(np.float32)
    indices = torch.as_tensor(np.vstack([rows, cols]), dtype=torch.long, device=device)
    vals = torch.as_tensor(values, dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(indices, vals, (n_users + n_items, n_users + n_items), device=device).coalesce()


class LightGCN(torch.nn.Module):
    def __init__(self, n_users: int, n_items: int, factors: int, layers: int, adjacency: torch.Tensor) -> None:
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.layers = layers
        self.adjacency = adjacency
        self.embedding = torch.nn.Embedding(n_users + n_items, factors)
        torch.nn.init.normal_(self.embedding.weight, std=0.05)

    def propagate(self) -> Tuple[torch.Tensor, torch.Tensor]:
        current = self.embedding.weight
        layers = [current]
        for _ in range(self.layers):
            current = torch.sparse.mm(self.adjacency, current)
            layers.append(current)
        final = torch.stack(layers, dim=0).mean(dim=0)
        return final[: self.n_users], final[self.n_users :]


def train_lightgcn(
    train: sp.csr_matrix,
    truth_val: Sequence[np.ndarray],
    cfg: Config,
    seed: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    set_seed(seed)
    coo = train.tocoo()
    users_np = coo.row.astype(np.int64)
    items_np = coo.col.astype(np.int64)
    adjacency = normalized_bipartite_adjacency(train, device)
    model = LightGCN(train.shape[0], train.shape[1], cfg.factors, cfg.lightgcn_layers, adjacency).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lightgcn_lr)
    rng = np.random.default_rng(seed + 303)
    users = torch.as_tensor(users_np, dtype=torch.long, device=device)
    pos = torch.as_tensor(items_np, dtype=torch.long, device=device)
    pop = np.asarray(train.sum(axis=0)).reshape(-1)
    best_metric = -np.inf
    best_state = None
    best_epoch = 0
    checks_without_gain = 0
    for epoch in range(1, cfg.lightgcn_epochs + 1):
        negatives_np = sample_negatives(users_np, items_np, train, rng)
        neg = torch.as_tensor(negatives_np, dtype=torch.long, device=device)
        user_f, item_f = model.propagate()
        pos_score = (user_f[users] * item_f[pos]).sum(dim=1)
        neg_score = (user_f[users] * item_f[neg]).sum(dim=1)
        ranking_loss = F.softplus(-(pos_score - neg_score)).mean()
        raw_u = model.embedding(users)
        raw_p = model.embedding(pos + train.shape[0])
        raw_n = model.embedding(neg + train.shape[0])
        reg = (raw_u.pow(2).sum(dim=1) + raw_p.pow(2).sum(dim=1) + raw_n.pow(2).sum(dim=1)).mean()
        loss = ranking_loss + cfg.lightgcn_reg * reg
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % cfg.eval_every == 0 or epoch == cfg.lightgcn_epochs:
            with torch.no_grad():
                val_u, val_i = model.propagate()
                u_np = val_u.detach().cpu().numpy()
                i_np = val_i.detach().cpu().numpy()
            recs = rank_all_users(lambda rows: u_np[rows] @ i_np.T, train, train.shape[0], cfg.top_k)
            score = evaluate_recommendations(recs, truth_val, pop, train.shape[0], train.shape[1])["nDCG@10"]
            if score > best_metric + 1e-6:
                best_metric = score
                best_epoch = epoch
                checks_without_gain = 0
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            else:
                checks_without_gain += 1
                if checks_without_gain >= cfg.patience_checks:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    with torch.no_grad():
        final_u, final_i = model.propagate()
    return (
        final_u.detach().cpu().numpy().astype(np.float32),
        final_i.detach().cpu().numpy().astype(np.float32),
        {"best_validation_nDCG@10": float(best_metric), "best_epoch": float(best_epoch)},
    )


def fit_wrmf_als(
    weighted_train: sp.csr_matrix,
    factors: int,
    alpha: float,
    reg: float,
    iterations: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Confidence-weighted implicit MF using the Hu-Koren-Volinsky ALS updates."""
    rng = np.random.default_rng(seed)
    n_users, n_items = weighted_train.shape
    x = rng.normal(0, 0.05, size=(n_users, factors)).astype(np.float64)
    y = rng.normal(0, 0.05, size=(n_items, factors)).astype(np.float64)
    confidence = weighted_train.copy().astype(np.float64)
    confidence.data = 1.0 + alpha * np.log1p(confidence.data)
    confidence_t = confidence.T.tocsr()
    eye = np.eye(factors, dtype=np.float64)

    def update(target: np.ndarray, fixed: np.ndarray, matrix: sp.csr_matrix) -> None:
        gram = fixed.T @ fixed
        base = gram + reg * eye
        for row in range(matrix.shape[0]):
            start, end = matrix.indptr[row], matrix.indptr[row + 1]
            idx = matrix.indices[start:end]
            c = matrix.data[start:end]
            if len(idx) == 0:
                target[row] = 0.0
                continue
            f = fixed[idx]
            a = base + (f.T * (c - 1.0)) @ f
            b = f.T @ c
            target[row] = np.linalg.solve(a, b)

    for _ in range(iterations):
        update(x, y, confidence)
        update(y, x, confidence_t)
    return x.astype(np.float32), y.astype(np.float32)


def retrain_pairwise_model(
    kind: str,
    fit: sp.csr_matrix,
    cfg: Config,
    seed: int,
    device: torch.device,
    epochs: int,
) -> Tuple[np.ndarray, np.ndarray]:
    set_seed(seed)
    coo = fit.tocoo()
    users_np = coo.row.astype(np.int64)
    items_np = coo.col.astype(np.int64)
    users = torch.as_tensor(users_np, dtype=torch.long, device=device)
    pos = torch.as_tensor(items_np, dtype=torch.long, device=device)
    rng = np.random.default_rng(seed + 707)
    if kind == "BPR-MF":
        model = BPRMF(fit.shape[0], fit.shape[1], cfg.factors).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.bpr_lr)
        for _ in range(epochs):
            neg_np = sample_negatives(users_np, items_np, fit, rng)
            neg = torch.as_tensor(neg_np, dtype=torch.long, device=device)
            pos_score = (model.user(users) * model.item(pos)).sum(dim=1)
            neg_score = (model.user(users) * model.item(neg)).sum(dim=1)
            loss = F.softplus(-(pos_score - neg_score)).mean()
            reg = (
                model.user(users).pow(2).sum(dim=1)
                + model.item(pos).pow(2).sum(dim=1)
                + model.item(neg).pow(2).sum(dim=1)
            ).mean()
            loss = loss + cfg.bpr_reg * reg
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return (
            model.user.weight.detach().cpu().numpy().astype(np.float32),
            model.item.weight.detach().cpu().numpy().astype(np.float32),
        )

    adjacency = normalized_bipartite_adjacency(fit, device)
    model = LightGCN(fit.shape[0], fit.shape[1], cfg.factors, cfg.lightgcn_layers, adjacency).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lightgcn_lr)
    for _ in range(epochs):
        neg_np = sample_negatives(users_np, items_np, fit, rng)
        neg = torch.as_tensor(neg_np, dtype=torch.long, device=device)
        user_f, item_f = model.propagate()
        pos_score = (user_f[users] * item_f[pos]).sum(dim=1)
        neg_score = (user_f[users] * item_f[neg]).sum(dim=1)
        ranking_loss = F.softplus(-(pos_score - neg_score)).mean()
        raw_u = model.embedding(users)
        raw_p = model.embedding(pos + fit.shape[0])
        raw_n = model.embedding(neg + fit.shape[0])
        reg = (raw_u.pow(2).sum(dim=1) + raw_p.pow(2).sum(dim=1) + raw_n.pow(2).sum(dim=1)).mean()
        loss = ranking_loss + cfg.lightgcn_reg * reg
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        final_u, final_i = model.propagate()
    return final_u.detach().cpu().numpy().astype(np.float32), final_i.detach().cpu().numpy().astype(np.float32)


def evaluate_one_seed(
    interactions: pd.DataFrame,
    n_users: int,
    n_items: int,
    cfg: Config,
    seed: int,
    device: torch.device,
) -> Tuple[List[Dict[str, float]], Dict[str, object]]:
    start = time.time()
    train_df, val_df, test_df, split_stats = split_per_user(interactions, n_users, n_items, seed)
    train = make_csr(train_df, n_users, n_items)
    fit_df = pd.concat([train_df, val_df], ignore_index=True)
    fit = make_csr(fit_df, n_users, n_items)
    fit_weighted = make_csr(fit_df, n_users, n_items, weighted=True)
    val_truth = truth_lists(val_df, n_users)
    test_truth = truth_lists(test_df, n_users)
    pop = np.asarray(fit.sum(axis=0)).reshape(-1)
    rows_all = np.arange(n_users)
    results: List[Dict[str, float]] = []
    diagnostics: Dict[str, object] = {"seed": seed, "split": split_stats}

    def record(model: str, recs: np.ndarray, extra: Dict[str, float] | None = None) -> None:
        metrics = evaluate_recommendations(recs, test_truth, pop, n_users, n_items)
        row: Dict[str, float] = {"Seed": float(seed), "Model": model}  # type: ignore[assignment]
        row.update(metrics)
        results.append(row)
        if extra:
            diagnostics[model] = extra

    popularity = pop.astype(np.float32)
    recs = rank_all_users(lambda rows: np.repeat(popularity[None, :], len(rows), axis=0), fit, n_users, cfg.top_k)
    record("Most Popular", recs)

    user_sim = fit_userknn(train, cfg.user_knn_k)
    val_scores = user_sim @ train
    user_val_recs = rank_all_users(lambda rows: val_scores[rows].toarray(), train, n_users, cfg.top_k)
    diagnostics["UserKNN_validation_nDCG@10"] = evaluate_recommendations(
        user_val_recs, val_truth, np.asarray(train.sum(axis=0)).reshape(-1), n_users, n_items
    )["nDCG@10"]
    user_sim = fit_userknn(fit, cfg.user_knn_k)
    user_scores = user_sim @ fit
    recs = rank_all_users(lambda rows: user_scores[rows].toarray(), fit, n_users, cfg.top_k)
    record("UserKNN", recs)
    del user_scores, user_sim, val_scores

    item_sim = fit_itemknn(train, cfg.item_knn_k)
    val_scores = train @ item_sim
    item_val_recs = rank_all_users(lambda rows: val_scores[rows].toarray(), train, n_users, cfg.top_k)
    diagnostics["ItemKNN_validation_nDCG@10"] = evaluate_recommendations(
        item_val_recs, val_truth, np.asarray(train.sum(axis=0)).reshape(-1), n_users, n_items
    )["nDCG@10"]
    item_sim = fit_itemknn(fit, cfg.item_knn_k)
    item_scores = fit @ item_sim
    recs = rank_all_users(lambda rows: item_scores[rows].toarray(), fit, n_users, cfg.top_k)
    record("ItemKNN", recs)
    del item_scores, item_sim, val_scores

    bpr_u, bpr_i, bpr_diag = train_bprmf(train, val_truth, cfg, seed, device)
    best_bpr_epoch = max(cfg.eval_every, int(bpr_diag["best_epoch"]))
    bpr_u, bpr_i = retrain_pairwise_model("BPR-MF", fit, cfg, seed + 1000, device, best_bpr_epoch)
    recs = rank_all_users(lambda rows: bpr_u[rows] @ bpr_i.T, fit, n_users, cfg.top_k)
    record("BPR-MF", recs, bpr_diag)

    wrmf_u, wrmf_i = fit_wrmf_als(
        fit_weighted,
        cfg.wrmf_factors,
        cfg.wrmf_alpha,
        cfg.wrmf_reg,
        cfg.wrmf_iterations,
        seed,
    )
    recs = rank_all_users(lambda rows: wrmf_u[rows] @ wrmf_i.T, fit, n_users, cfg.top_k)
    record("WRMF", recs)

    lg_u, lg_i, lg_diag = train_lightgcn(train, val_truth, cfg, seed, device)
    best_lg_epoch = max(cfg.eval_every, int(lg_diag["best_epoch"]))
    lg_u, lg_i = retrain_pairwise_model("LightGCN", fit, cfg, seed + 2000, device, best_lg_epoch)
    recs = rank_all_users(lambda rows: lg_u[rows] @ lg_i.T, fit, n_users, cfg.top_k)
    record("LightGCN", recs, lg_diag)

    diagnostics["elapsed_seconds"] = float(time.time() - start)
    return results, diagnostics


def make_summary(results: pd.DataFrame) -> pd.DataFrame:
    metrics = ["Recall@10", "nDCG@10", "MRR@10", "Coverage@10", "Novelty@10"]
    grouped = results.groupby("Model")[metrics].agg(["mean", "std"])
    grouped.columns = [f"{metric}_{stat}" for metric, stat in grouped.columns]
    summary = grouped.reset_index()
    order = ["Most Popular", "UserKNN", "ItemKNN", "BPR-MF", "WRMF", "LightGCN"]
    summary["order"] = summary["Model"].map({name: i for i, name in enumerate(order)})
    return summary.sort_values("order").drop(columns="order").reset_index(drop=True)


def plot_results(summary: pd.DataFrame, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    colors = ["#C8B6A6", "#A8B6A3", "#8FA8B8", "#D39B78", "#967A67", "#2E6F73"]
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "figure.facecolor": "#F7F1E7",
        "axes.facecolor": "#F7F1E7",
    })
    models = summary["Model"].tolist()
    x = np.arange(len(models))

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.1), constrained_layout=True)
    for ax, metric in zip(axes, ["Recall@10", "nDCG@10", "MRR@10"]):
        mean = summary[f"{metric}_mean"].to_numpy()
        std = summary[f"{metric}_std"].fillna(0).to_numpy()
        bars = ax.bar(x, mean, yerr=std, capsize=3, color=colors, edgecolor="#5D534A", linewidth=0.5)
        ax.set_title(metric)
        ax.set_xticks(x, models, rotation=28, ha="right")
        ax.grid(axis="y", alpha=0.22)
        ax.set_ylim(0, max(mean + std) * 1.28)
        for bar, value in zip(bars, mean):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(mean) * 0.035, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("Top-10 ranking accuracy across five random splits", fontsize=14, fontweight="bold", color="#3A2E28")
    fig.savefig(figures_dir / "ranking_accuracy.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.6, 5.0), constrained_layout=True)
    for idx, row in summary.iterrows():
        ax.scatter(row["Coverage@10_mean"], row["nDCG@10_mean"], s=170, color=colors[idx], edgecolor="#4B4038", linewidth=0.7, zorder=3)
        ax.annotate(row["Model"], (row["Coverage@10_mean"], row["nDCG@10_mean"]), xytext=(7, 5), textcoords="offset points", fontsize=9)
    ax.set_xlabel("Catalog Coverage@10")
    ax.set_ylabel("nDCG@10")
    ax.set_title("Accuracy–coverage trade-off")
    ax.grid(alpha=0.22)
    fig.savefig(figures_dir / "accuracy_coverage.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    args = parser.parse_args()
    project_dir = args.project_dir.resolve()
    data_dir = project_dir / "data" / "raw" / "lastfm-2k"
    results_dir = project_dir / "results"
    figures_dir = project_dir / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config()
    interactions, artists, _, _ = load_lastfm(data_dir)
    n_users = int(interactions["user"].nunique())
    n_items = int(interactions["item"].nunique())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}; users={n_users:,}; artists={n_items:,}; pairs={len(interactions):,}", flush=True)
    all_rows: List[Dict[str, float]] = []
    all_diagnostics: List[Dict[str, object]] = []
    for seed in args.seeds:
        print(f"Running seed {seed}...", flush=True)
        rows, diagnostics = evaluate_one_seed(interactions, n_users, n_items, cfg, seed, device)
        all_rows.extend(rows)
        all_diagnostics.append(diagnostics)
        pd.DataFrame(all_rows).to_csv(results_dir / "metrics_by_seed.csv", index=False)
        print(f"Seed {seed} finished in {diagnostics['elapsed_seconds']:.1f}s", flush=True)

    results = pd.DataFrame(all_rows)
    summary = make_summary(results)
    summary.to_csv(results_dir / "metrics_summary.csv", index=False)
    with open(results_dir / "run_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "config": asdict(cfg),
                "seeds": args.seeds,
                "device": str(device),
                "dataset": {"users": n_users, "artists": n_items, "pairs": int(len(interactions))},
                "diagnostics": all_diagnostics,
            },
            handle,
            indent=2,
        )
    plot_results(summary, figures_dir)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
