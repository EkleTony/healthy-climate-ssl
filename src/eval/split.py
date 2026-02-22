# src/eval/split.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import torch


def _allocate_counts(n: int, ratios=(0.7, 0.1, 0.2)) -> Tuple[int, int, int]:
    """
    Return (n_train, n_val, n_test) counts that sum to n.
    Uses rounding then fixes remainder deterministically.
    """
    r_train, r_val, r_test = ratios
    n_train = int(round(n * r_train))
    n_val = int(round(n * r_val))
    n_test = n - n_train - n_val

    # Fix potential negatives (can happen for very small n)
    if n_test < 0:
        # reduce train first, then val
        deficit = -n_test
        take = min(deficit, n_train)
        n_train -= take
        deficit -= take
        if deficit > 0:
            take = min(deficit, n_val)
            n_val -= take
            deficit -= take
        n_test = n - n_train - n_val

    return n_train, n_val, n_test


def stratified_split_by_group(
    state_id: torch.Tensor,
    seed: int = 7,
    ratios=(0.7, 0.1, 0.2),
) -> Dict[str, torch.Tensor]:
    """
    Stratified split by state_id.
    - Each group is shuffled deterministically
    - Split per-group, then concatenated
    - Handles tiny groups gracefully
    """
    assert state_id.ndim == 1, "state_id must be 1D [N]"
    N = state_id.numel()

    g = torch.Generator().manual_seed(seed)

    # unique groups
    groups = torch.unique(state_id).tolist()

    train_idx_all = []
    val_idx_all = []
    test_idx_all = []

    for gid in groups:
        idx = torch.nonzero(state_id == gid, as_tuple=False).view(-1)
        n = idx.numel()

        # deterministic shuffle per group
        perm = idx[torch.randperm(n, generator=g)]

        n_train, n_val, n_test = _allocate_counts(n, ratios=ratios)

        # If group is tiny, prefer train then test, keep val minimal.
        # Ensure at least 1 in train if possible.
        if n == 1:
            n_train, n_val, n_test = 1, 0, 0
        elif n == 2:
            n_train, n_val, n_test = 1, 0, 1
        elif n == 3:
            n_train, n_val, n_test = 2, 0, 1

        train_idx_all.append(perm[:n_train])
        val_idx_all.append(perm[n_train: n_train + n_val])
        test_idx_all.append(perm[n_train + n_val: n_train + n_val + n_test])

    train_idx = torch.cat(train_idx_all) if train_idx_all else torch.empty(
        0, dtype=torch.long)
    val_idx = torch.cat(val_idx_all) if val_idx_all else torch.empty(
        0, dtype=torch.long)
    test_idx = torch.cat(test_idx_all) if test_idx_all else torch.empty(
        0, dtype=torch.long)

    # final shuffle within each split (optional but nice)
    train_idx = train_idx[torch.randperm(train_idx.numel(), generator=g)]
    val_idx = val_idx[torch.randperm(val_idx.numel(), generator=g)]
    test_idx = test_idx[torch.randperm(test_idx.numel(), generator=g)]

    # sanity: disjoint + cover all
    all_idx = torch.cat([train_idx, val_idx, test_idx])
    assert all_idx.numel() == N, "Split sizes do not sum to N"
    assert torch.unique(all_idx).numel(
    ) == N, "Splits are not disjoint / missing indices"

    return {"train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data/v1_seed7")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    d = Path(args.data_dir)
    regions = torch.load(d / "regions.pt", weights_only=True)
    state_id = regions["state_id"].long()

    splits = stratified_split_by_group(state_id=state_id, seed=args.seed)

    out_path = d / f"splits_seed{args.seed}.pt"
    torch.save(
        {
            **splits,
            "seed": args.seed,
            "ratios": (0.7, 0.1, 0.2),
            "N": int(state_id.numel()),
            "num_groups": int(torch.unique(state_id).numel()),
        },
        out_path,
    )

    print(f"[split] saved -> {out_path}")
    print(
        f"[split] N={state_id.numel()} groups={torch.unique(state_id).numel()}")
    print(
        f"[split] train={splits['train_idx'].numel()} val={splits['val_idx'].numel()} test={splits['test_idx'].numel()}")


if __name__ == "__main__":
    main()
