import json
import time
import hashlib
import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple


def _canonical_json(o: Any) -> str:
    """Stable JSON serialization used for hashing (order-insensitive)."""
    return json.dumps(o, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def default_stable_key_fn(cfg: Dict[str, Any]) -> str:
    """
    Produce a stable identity key for a logical slot.
    Priority: id > name > (type or hook_name) + variables.
    """
    if "id" in cfg: return f"id:{cfg['id']}"
    if "name" in cfg: return f"name:{cfg['name']}"
    t = cfg.get("type") or cfg.get("hook_name") or "unknown"
    vars_ser = _canonical_json(cfg.get("variables", {}))
    return f"{t}|vars:{vars_ser}"


def default_config_hash_fn(cfg: Dict[str, Any], ignored_keys: Iterable[str] = ("id", "name")) -> str:
    """
    Hash only behavior-affecting fields, ignoring identity keys such as id/name.
    """
    filt = {k: v for k, v in cfg.items() if k not in set(ignored_keys)}
    s = _canonical_json(filt)
    return hashlib.blake2b(s.encode("utf-8"), digest_size=16).hexdigest()


@dataclass
class _Entry:
    instance: Any
    cfg_hash: str
    last_used_ts: float


class ConfigBoundInstanceCache:
    """
    A generic 'config -> instance' persistent cache with incremental sync.
    Features:
      - Namespace isolation with a default namespace fallback
      - Diff-based reconciliation (add/remove/change -> reconfigure or rebuild)
      - Order-preserving return (aligns with incoming cfg list)
      - Optional global capacity via LRU eviction
      - Thread-safe (RLock)

    Args:
      - factory(cfg) -> instance
      - reconfigure(instance, cfg) -> bool (optional; True means in-place update succeeded)
      - closer(instance) (optional; resource cleanup)
      - stable_key_fn(cfg) (optional; identity of a logical slot)
      - config_hash_fn(cfg) (optional; change detection)

    Default namespace behavior:
      - If a method's `namespace` is None, `self.default_namespace` is used.
      - Set `default_namespace` in __init__ (default: "__default__").
    """

    def __init__(
            self,
            factory: Callable[[Dict[str, Any]], Any],
            reconfigure: Optional[Callable[[Any, Dict[str, Any]], bool]] = None,
            closer: Optional[Callable[[Any], None]] = None,
            stable_key_fn: Callable[[Dict[str, Any]], str] = default_stable_key_fn,
            config_hash_fn: Callable[[Dict[str, Any]], str] = default_config_hash_fn,
            capacity: Optional[int] = None,  # Global max instances across namespaces
            default_namespace: str = "__default__",
    ) -> None:
        self._factory = factory
        self._reconfigure = reconfigure
        self._closer = closer
        self._stable_key_fn = stable_key_fn
        self._config_hash_fn = config_hash_fn
        self._capacity = capacity
        self.default_namespace = default_namespace

        # cache: {namespace: {stable_key: _Entry}}
        self._cache: Dict[str, Dict[str, _Entry]] = {}
        self._lock = threading.RLock()

    def sync_and_get(
            self,
            cfg_list: List[Dict[str, Any]],
            namespace: Optional[str] = None,
    ) -> List[Any]:
        """
        Reconcile instances in `namespace` against `cfg_list`:
          - Create missing instances
          - Reconfigure or rebuild changed ones
          - Remove obsolete ones
        Return instances in the same order as `cfg_list`.

        If `namespace` is None, `self.default_namespace` is used.
        """
        ns = namespace or self.default_namespace
        now = time.time()

        with self._lock:
            slot = self._cache.setdefault(ns, {})

            # 1) Compute desired state: [(stable_key, cfg_hash, cfg)]
            desired: List[Tuple[str, str, Dict[str, Any]]] = []
            desired_keys = set()
            for cfg in cfg_list:
                sk = self._stable_key_fn(cfg)
                ch = self._config_hash_fn(cfg)
                desired.append((sk, ch, cfg))
                desired_keys.add(sk)

            # 2) Remove no-longer-needed entries
            for sk in list(slot.keys()):
                if sk not in desired_keys:
                    self._dispose_entry(slot.pop(sk))

            # 3) Add or update, preserving order
            ordered_instances: List[Any] = []
            for sk, ch, cfg in desired:
                entry = slot.get(sk)
                if entry is None:
                    inst = self._factory(cfg)
                    entry = _Entry(instance=inst, cfg_hash=ch, last_used_ts=now)
                    slot[sk] = entry
                elif entry.cfg_hash != ch:
                    # Try in-place reconfigure first
                    ok = False
                    if self._reconfigure is not None:
                        try:
                            ok = self._reconfigure(entry.instance, cfg)
                        except Exception:
                            ok = False
                    if ok:
                        entry.cfg_hash = ch
                    else:
                        # Rebuild when reconfigure is missing or failed
                        self._dispose_entry(entry)
                        inst = self._factory(cfg)
                        entry = _Entry(instance=inst, cfg_hash=ch, last_used_ts=now)
                        slot[sk] = entry

                entry.last_used_ts = now
                ordered_instances.append(entry.instance)

            # 4) Global capacity control (LRU across namespaces)
            if self._capacity is not None:
                self._evict_lru_if_needed()

            return ordered_instances

    def get_existing(self, stable_key: str, namespace: Optional[str] = None) -> Optional[Any]:
        """
        Get an existing instance by stable_key within `namespace`, without syncing.
        Updates the LRU timestamp on hit. Returns None if not found.
        """
        ns = namespace or self.default_namespace
        with self._lock:
            entry = self._cache.get(ns, {}).get(stable_key)
            if entry:
                entry.last_used_ts = time.time()
                return entry.instance
            return None

    def remove(self, stable_key: str, namespace: Optional[str] = None) -> None:
        """
        Remove a specific instance by stable_key from `namespace` (and dispose it).
        """
        ns = namespace or self.default_namespace
        with self._lock:
            slot = self._cache.get(ns)
            if not slot:
                return
            entry = slot.pop(stable_key, None)
            if entry:
                self._dispose_entry(entry)

    def clear_namespace(self, namespace: Optional[str] = None) -> None:
        """
        Clear all instances under the given `namespace` (default namespace if None).
        """
        ns = namespace or self.default_namespace
        with self._lock:
            slot = self._cache.pop(ns, {})
            for e in slot.values():
                self._dispose_entry(e)

    def clear_all(self) -> None:
        """Clear all namespaces and dispose all instances."""
        with self._lock:
            for slot in self._cache.values():
                for e in slot.values():
                    self._dispose_entry(e)
            self._cache.clear()

    def prune_idle(self, idle_seconds: float) -> int:
        """
        Dispose instances that have not been used for `idle_seconds`.
        Returns the number of entries removed.
        """
        cutoff = time.time() - idle_seconds
        removed = 0
        with self._lock:
            for ns in list(self._cache.keys()):
                slot = self._cache[ns]
                for sk in list(slot.keys()):
                    if slot[sk].last_used_ts < cutoff:
                        self._dispose_entry(slot.pop(sk))
                        removed += 1
                if not slot:
                    self._cache.pop(ns, None)
        return removed

    def _evict_lru_if_needed(self) -> None:
        """Global LRU eviction across namespaces when capacity is exceeded."""
        total = sum(len(s) for s in self._cache.values())
        if self._capacity is None or total <= self._capacity:
            return
        # Collect all entries
        all_entries: List[Tuple[str, str, _Entry]] = []
        for ns, slot in self._cache.items():
            for sk, e in slot.items():
                all_entries.append((ns, sk, e))
        # Oldest first
        all_entries.sort(key=lambda t: t[2].last_used_ts)
        to_remove = total - self._capacity
        for i in range(to_remove):
            ns, sk, e = all_entries[i]
            slot = self._cache.get(ns, {})
            if slot.pop(sk, None):
                self._dispose_entry(e)

    def _dispose_entry(self, entry: _Entry) -> None:
        """Best-effort resource disposal; errors are swallowed."""
        if self._closer:
            try:
                self._closer(entry.instance)
            except Exception:
                pass
