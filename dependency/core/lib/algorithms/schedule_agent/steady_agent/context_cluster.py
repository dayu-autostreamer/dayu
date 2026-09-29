"""Context bins and conservative representatives from the STEADY model."""

from bisect import bisect_right
from math import inf


class ContextCluster:
    def __init__(self, cluster_threshold):
        if not 0 <= cluster_threshold <= 1:
            raise ValueError("cluster_threshold must be in [0, 1]")
        self.cluster_threshold = cluster_threshold

    def process_context_for_cluster(self, cur_context):
        names = ("band_Mbps", "obj_size_norm", "obj_num", "obj_speed")
        if not self.cluster_threshold or cur_context is None or any(cur_context.get(name) is None for name in names):
            return None, None, None
        threshold = self.cluster_threshold
        # Each bin retains its original representative and acceptance interval.
        dimensions = (
            (
                (0.1, 1, 5, 10),
                (0, 0.1, 1, 5, 10),
                (-inf,) * 5,
                (inf, 0.1 + 0.9 * threshold, 1 + 4 * threshold, 5 + 5 * threshold, 10 + 10 * threshold),
            ),
            (
                (0.05, 0.1, 0.2, 0.3),
                (0.01, 0.05, 0.1, 0.2, 0.3),
                (-inf,) * 5,
                (
                    inf,
                    0.05 + 0.05 * threshold,
                    0.1 + 0.1 * threshold,
                    0.2 + (0.3 - 0.2) * threshold,
                    0.3 + 0.3 * threshold,
                ),
            ),
            (
                (1, 5, 10),
                (1, 5, 10, 20),
                (-inf, 5 - 4 * threshold, 10 - 5 * threshold, 20 - 10 * threshold),
                (inf, inf, inf, 20),
            ),
            (
                (260, 520, 780),
                (260, 520, 780, 1500),
                (-inf, 520 - 260 * threshold, 780 - 260 * threshold, 1500 - 720 * threshold),
                (inf, inf, inf, 1500),
            ),
        )
        cluster = []
        extreme = {}
        belongs = True
        for name, (boundaries, representatives, lower, upper) in zip(names, dimensions):
            value = cur_context[name]
            index = bisect_right(boundaries, value)
            cluster.append(str(index))
            extreme[name] = representatives[index]
            belongs = belongs and lower[index] <= value <= upper[index]
        return "".join(cluster), extreme, int(belongs)
