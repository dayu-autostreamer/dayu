from functools import wraps


def reverse_key_value_in_dict(in_dict: dict) -> dict:
    """
    reverse the key and value in dict object
    {(k:v)} -> {(v:k)}
    :param in_dict: input dict
    :return: output dict
    """

    return {v: k for k, v in in_dict.items()}


def convert_ndarray_to_list(obj):
    import numpy as np
    if isinstance(obj, np.ndarray):
        return convert_ndarray_to_list(obj.tolist())
    elif isinstance(obj, list):
        return [convert_ndarray_to_list(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_ndarray_to_list(item) for item in obj)
    elif isinstance(obj, dict):
        return {convert_ndarray_to_list(key): convert_ndarray_to_list(value) for key, value in obj.items()}
    else:
        return obj

def deep_merge(target, source):
    """
    Deeply merge two nested structures (dictionaries and lists), processing rules:
        - For dictionaries: Recursively merge all keys.
        - For lists:
            1. Prioritize element merging based on the `name` key.
            2. Remaining elements are merged by index (if the target list length is sufficient) or appended.
        - When types differ, use the source value to replace the target value.
    """
    if isinstance(target, dict) and isinstance(source, dict):
        # Recursive merge dictionary
        for key in source:
            if key in target:
                target[key] = deep_merge(target[key], source[key])
            else:
                target[key] = source[key]
        return target
    elif isinstance(target, list) and isinstance(source, list):
        name_key = "name"

        processed_indices = set()
        for src_idx, src_item in enumerate(source):
            if isinstance(src_item, dict) and name_key in src_item:
                src_name = src_item[name_key]
                # Find an item with the same name in target
                for tgt_idx, tgt_item in enumerate(target):
                    if isinstance(tgt_item, dict) and tgt_item.get(name_key) == src_name:
                        target[tgt_idx] = deep_merge(tgt_item, src_item)
                        processed_indices.add(src_idx)
                        break
                else:
                    # Item with the same name not found, appended to target
                    target.append(src_item)
                    processed_indices.add(src_idx)

        # Merge remaining elements by index
        for src_idx, src_item in enumerate(source):
            if src_idx in processed_indices:
                continue
            if src_idx < len(target):
                target[src_idx] = deep_merge(target[src_idx], src_item)
            else:
                target.append(src_item)
        return target
    else:
        return source


def singleton(cls):
    """Set class to singleton class.

    :param cls: class
    :return: instance
    """
    __instances__ = {}

    @wraps(cls)
    def get_instance(*args, **kw):
        """Get class instance and save it into glob list."""
        if cls not in __instances__:
            __instances__[cls] = cls(*args, **kw)
        return __instances__[cls]

    return get_instance
