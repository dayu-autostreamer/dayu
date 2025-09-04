from typing import Tuple, List
from collections import deque

import torch


def safe_log1p(x: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    return torch.log1p(torch.clamp(x, min=0.0) + eps)


def graph_in_out_degree(edge_index: torch.Tensor, num_nodes: int) -> Tuple[torch.Tensor, torch.Tensor]:
    row, col = edge_index
    out_deg = torch.zeros(num_nodes, device=edge_index.device).scatter_add_(0, row,
                                                                            torch.ones_like(row, dtype=torch.float))
    in_deg = torch.zeros(num_nodes, device=edge_index.device).scatter_add_(0, col,
                                                                           torch.ones_like(col, dtype=torch.float))
    return in_deg, out_deg


def topo_levels_dag(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    row, col = edge_index
    indeg = torch.zeros(num_nodes, dtype=torch.long, device=edge_index.device)
    indeg.scatter_add_(0, col, torch.ones_like(col, dtype=torch.long))
    q = [i for i in range(num_nodes) if indeg[i] == 0]
    levels = torch.zeros(num_nodes, device=edge_index.device)
    cur_level = 0
    visited = 0
    adj = [[] for _ in range(num_nodes)]
    for u, v in zip(row.tolist(), col.tolist()):
        adj[u].append(v)
    while q:
        next_q = []
        for u in q:
            levels[u] = cur_level
            visited += 1
            for v in adj[u]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    next_q.append(v)
        q = next_q
        cur_level += 1
    if visited < num_nodes:
        levels[levels == 0] = cur_level
    if levels.max() > 0:
        levels = levels / (levels.max() + 1e-6)
    return levels


def bfs_hop_from_source(phys_edge_index: torch.Tensor, N: int, source_idx: int) -> torch.Tensor:
    row, col = phys_edge_index
    adj = [[] for _ in range(N)]
    for u, v in zip(row.tolist(), col.tolist()):
        adj[u].append(v)
    INF = 10 ** 9
    dist = [INF] * N
    q = deque([source_idx])
    dist[source_idx] = 0
    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist[v] == INF:
                dist[v] = dist[u] + 1
                q.append(v)
    return torch.tensor(dist, device=phys_edge_index.device, dtype=torch.float)


def compute_returns_advantages(rewards: List[float], values: List[float], dones: List[int], gamma=0.99, lamda=0.95):
    T = len(rewards)
    adv = [0.0] * T
    gae = 0.0
    values_ext = values + [0.0]
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * values_ext[t + 1] * (1 - dones[t]) - values_ext[t]
        gae = delta + gamma * lamda * (1 - dones[t]) * gae
        adv[t] = gae
    returns = [adv[t] + values[t] for t in range(T)]
    return adv, returns
