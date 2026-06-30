from typing import Tuple, List

import torch


def safe_log1p(x: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    return torch.log1p(torch.clamp(x, min=0.0) + eps)


def graph_in_out_degree(edge_index: torch.Tensor, num_nodes: int) -> Tuple[torch.Tensor, torch.Tensor]:
    row, col = edge_index
    out_deg = torch.zeros(num_nodes, device=edge_index.device).scatter_add(
        0,
        row,
        torch.ones_like(row, dtype=torch.float),
    )
    in_deg = torch.zeros(num_nodes, device=edge_index.device).scatter_add(
        0,
        col,
        torch.ones_like(col, dtype=torch.float),
    )
    return in_deg, out_deg


def topo_levels_dag(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    if num_nodes <= 0:
        return torch.zeros((0,), device=edge_index.device)
    if edge_index.numel() == 0:
        return torch.zeros((num_nodes,), device=edge_index.device)

    row, col = edge_index.detach().cpu()
    indeg = [0 for _ in range(num_nodes)]
    levels = [0 for _ in range(num_nodes)]
    cur_level = 0
    visited = 0
    adj = [[] for _ in range(num_nodes)]
    for u_raw, v_raw in zip(row.tolist(), col.tolist()):
        u, v = int(u_raw), int(v_raw)
        if 0 <= u < num_nodes and 0 <= v < num_nodes:
            adj[u].append(v)
            indeg[v] += 1

    q = [i for i, deg in enumerate(indeg) if deg == 0]
    visited_nodes = set()
    while q:
        next_q = []
        for u in q:
            levels[u] = cur_level
            visited_nodes.add(u)
            visited += 1
            for v in adj[u]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    next_q.append(v)
        q = next_q
        cur_level += 1
    if visited < num_nodes:
        for idx in range(num_nodes):
            if idx not in visited_nodes:
                levels[idx] = cur_level
    max_level = max(levels) if levels else 0
    if max_level > 0:
        denom = float(max_level) + 1e-6
        levels = [level / denom for level in levels]
    return torch.tensor(levels, device=edge_index.device, dtype=torch.float32)


def compute_returns_advantages(
        rewards: List[float],
        values: List[float],
        dones: List[int],
        gamma=0.99,
        lamda=0.95,
        last_value: float = 0.0,
):
    T = len(rewards)
    adv = [0.0] * T
    gae = 0.0
    values_ext = values + [float(last_value)]
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * values_ext[t + 1] * (1 - dones[t]) - values_ext[t]
        gae = delta + gamma * lamda * (1 - dones[t]) * gae
        adv[t] = gae
    returns = [adv[t] + values[t] for t in range(T)]
    return adv, returns
