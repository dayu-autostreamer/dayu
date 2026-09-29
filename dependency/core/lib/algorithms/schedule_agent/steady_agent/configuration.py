"""Configuration normalization shared by STEADY and its ablations."""

import copy
import math
from itertools import product


def constraint_combinations(parameters):
    """Keep single-constraint configurations and optional experiment sweeps compatible."""
    dimensions = []
    for name in ("delay_weight", "delay_cons", "acc_cons"):
        entries = parameters.get(f"all_{name}_info")
        if entries is None:
            entries = [{"value": parameters[name], "adjust": parameters.get(f"{name}_adjust", 1)}]
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"all_{name}_info must be a nonempty list")
        dimension = []
        for entry in entries:
            value = float(entry["value"])
            adjustment = float(entry.get("adjust", 1))
            adjusted = value * adjustment
            if not all(math.isfinite(number) for number in (value, adjustment, adjusted)):
                raise ValueError(f"{name} and its adjustment must be finite")
            if name == "delay_weight":
                valid = 0 <= value <= 1 and 0 <= adjusted <= 1
            elif name == "acc_cons":
                valid = 0 < value <= 1 and 0 < adjusted <= 1
            else:
                valid = value > 0 and adjusted > 0
            if not valid:
                raise ValueError(f"invalid {name} constraint or adjustment")
            dimension.append({"value": value, "adjust": adjustment})
        dimensions.append(dimension)
    return [
        {"delay_weight_info": weight, "delay_cons_info": delay, "acc_cons_info": accuracy}
        for weight, delay, accuracy in product(*dimensions)
    ]


def constraint_values(combination, adjusted=False):
    values = {
        name: entry["value"] * (entry["adjust"] if adjusted else 1)
        for name, entry in ((name, combination[f"{name}_info"]) for name in ("delay_cons", "acc_cons", "delay_weight"))
    }
    values["acc_weight"] = 1 - values["delay_weight"]
    return values


def normalize_parameters(parameters):
    parameters = copy.deepcopy(parameters)
    # Normalize existing public template spellings at the configuration boundary.
    for old, new in (("history_lenghth", "history_length"), ("context_anylze_type", "context_analyze_type")):
        if old in parameters:
            parameters.setdefault(new, parameters.pop(old))
    parameters.setdefault("if_online_train", True)
    parameters.setdefault("cluster_threshold", 0.5)
    if parameters["if_online_train"] not in (False, True):
        raise ValueError("if_online_train must be a boolean or 0/1")
    for name in ("acc_sample_interval", "history_length"):
        value = parameters[name]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if parameters["context_analyze_type"] not in (0, 1):
        raise ValueError("context_analyze_type must be 0 (conservative) or 1 (mean)")
    interval = parameters["macro_update_interval"]
    if not math.isfinite(interval) or interval <= 0:
        raise ValueError("macro_update_interval must be finite and positive")
    if not 0 <= parameters["stop_threshold"] < 1:
        raise ValueError("stop_threshold must be in [0, 1)")
    step = parameters["coeff_info"]["step_coeff"]
    if not (0 < step["min_value"] <= step["start_value"] <= step["max_value"] and step["add_interval"] > 0):
        raise ValueError("invalid feedback step coefficient bounds")
    return parameters
