from dataclasses import dataclass, fields, replace


def from_partial_dict(cls, data: dict):
    allowed = {f.name for f in fields(cls)}
    filtered = {k: v for k, v in data.items() if k in allowed}
    return replace(cls(), **filtered)


@dataclass
class OffloadingConstraintCfg:
    allow_stay: bool = True  # 允许原地执行（不算回访）
    forbid_return: bool = True  # 禁止回到任何历史节点（除当前所在）
    cloud_sticky: bool = True  # 上云后黏住云
    use_monotone_metric: bool = False  # 可选：启用单调性约束
    metric_non_decreasing: bool = True  # True: metric[node_t] >= metric[last]
    penalty_switch: float = 0.0  # 节点切换惩罚系数（软约束）
    penalty_relax: float = 0.0  # 无可行动作被迫放宽的惩罚系数（软约束）


@dataclass
class DeploymentConstraintCfg:
    enforce_capacity: bool = True  # 是否强 enforcing 显存容量
    min_edge_replicas: int = 0  # 可选：可以强制每个服务至少在多少个边端部署
    penalty_capacity_relax: float = 1.0  # 若不得不放宽容量（最后兜底）时的惩罚
