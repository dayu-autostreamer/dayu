import copy
import heapq
import math
import random
from collections import defaultdict, deque

from core.lib.common import TaskConstant


START = TaskConstant.START.value
END = TaskConstant.END.value


def service_names(dag):
    return [name for name in dag if name not in (START, END)]


def topological_order(dag):
    nodes = list(dag)
    indegree = {
        name: len([item for item in dag[name].get("prev_nodes", []) if item in dag])
        for name in nodes
    }
    ready = sorted(name for name, degree in indegree.items() if degree == 0)
    result = []
    while ready:
        current = ready.pop(0)
        result.append(current)
        for successor in dag[current].get("next_nodes", []):
            if successor not in indegree:
                continue
            indegree[successor] -= 1
            if indegree[successor] == 0:
                ready.append(successor)
                ready.sort()
    if len(result) != len(nodes):
        raise ValueError("FragSplice requires an acyclic DAG")
    return result


def plan_from_dag(dag):
    result = {}
    for service in service_names(dag):
        node = dag.get(service, {})
        spec = node.get("service", {}) if isinstance(node, dict) else {}
        device = str(spec.get("execute_device") or "")
        if device:
            result[str(service)] = device
    return result


def _identity_root(value):
    return str(value.get("root_uuid") or "") if isinstance(value, dict) else ""


def _identity_revision(value):
    if not isinstance(value, dict):
        return None
    try:
        revision = int(value.get("runtime_directory_revision"))
    except (TypeError, ValueError):
        return None
    return revision if revision > 0 else None


def _record_dag(record):
    if not isinstance(record, dict):
        return None
    dag = record.get("dag")
    if isinstance(dag, dict):
        return dag
    plan = record.get("plan")
    dag = plan.get("dag") if isinstance(plan, dict) else None
    return dag if isinstance(dag, dict) else None


def _record_started(record, fallback):
    """Return the exact SLO clock origin for an admitted root."""

    for key in ("slo_started_at", "created_at", "reserved_at", "admitted_at"):
        try:
            value = float(record.get(key))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and value > 0.0:
            return value
    return float(fallback)


def _non_negative_int(value):
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _deadline(commitment, default_slo):
    metadata = commitment.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    for key in ("slo_seconds", "slo", "latency_slo_s", "deadline_seconds"):
        try:
            value = float(metadata.get(key))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and value > 0:
            return value
    return float(default_slo)


class FragSpliceExecutionState:
    """White-box reconstruction of committed FIFO DAG execution.

    Each processor ``service@device`` is one independent FIFO resource.  The
    model anchors the present using ordered queue telemetry, then releases the
    unobserved remainder of immutable task plans through the actual fork/join
    dependencies.
    """

    def __init__(
        self,
        snapshot,
        latency_model,
        default_slo_s,
        queue_state_max_age_s=1.5,
    ):
        self.snapshot = copy.deepcopy(snapshot or {})
        self.latency_model = latency_model
        self.default_slo_s = max(1e-6, float(default_slo_s))
        self.queue_state_max_age_s = max(0.0, float(queue_state_max_age_s))
        self.now = float(self.snapshot.get("captured_at") or 0.0)
        try:
            self.revision = int(
                self.snapshot.get("runtime_directory_revision") or 0
            )
        except (TypeError, ValueError):
            self.revision = 0

        # Active commitments supersede pending reservations with the same root.
        # Both are immutable future workload once a task-bound decision exists.
        records = {}
        ordered_records = (
            [(item, False) for item in self.snapshot.get("reservations", [])]
            + [(item, True) for item in self.snapshot.get("commitments", [])]
        )
        for item, admitted in ordered_records:
            dag = _record_dag(item)
            root = str(item.get("root_uuid") or "") if isinstance(item, dict) else ""
            if not root or dag is None:
                continue
            try:
                record_revision = int(
                    item.get("runtime_directory_revision") or self.revision
                )
            except (TypeError, ValueError):
                continue
            if self.revision and record_revision != self.revision:
                continue
            normalized = copy.deepcopy(item)
            normalized["dag"] = copy.deepcopy(dag)
            normalized["runtime_directory_revision"] = record_revision
            normalized["_slo_admitted"] = admitted
            records[root] = normalized
        self.commitments = list(records.values())
        self.barriers = [
            item for item in self.snapshot.get("task_barriers", [])
            if isinstance(item, dict)
        ]
        # One execution state is evaluated against many candidate plans.  The
        # observed queue anchor and the sampled in-flight tasks are identical
        # for all candidates under a common scenario seed, so build them once
        # instead of reconstructing them for every leaf visited by BnB.
        self._queue_state_cache = self._queue_states()
        self._sample_root_cache = {}
        self._dag_order_cache = {}

    def _dag_order(self, dag):
        key = id(dag)
        cached = self._dag_order_cache.get(key)
        if cached is not None and cached[0] is dag:
            return cached[1]
        order = tuple(topological_order(dag))
        self._dag_order_cache[key] = (dag, order)
        return order

    def _queue_states(self):
        states = {}
        received_at = self.snapshot.get("resource_received_at") or {}
        resource_revisions = self.snapshot.get("resource_runtime_revision") or {}
        for device, resource in (self.snapshot.get("resources") or {}).items():
            if not isinstance(resource, dict):
                continue
            try:
                resource_revision = int(resource_revisions.get(device))
            except (TypeError, ValueError):
                resource_revision = None
            if self.revision and resource_revision != self.revision:
                continue
            for service, state in (resource.get("queue_state") or {}).items():
                if isinstance(state, dict):
                    normalized = copy.deepcopy(state)
                    # The Scheduler receive timestamp is in the same clock
                    # domain as captured_at. Processor observed_at remains a
                    # diagnostic field and is only a legacy fallback.
                    try:
                        observed_at = float(received_at.get(device))
                    except (TypeError, ValueError):
                        try:
                            observed_at = float(normalized.get("observed_at"))
                        except (TypeError, ValueError):
                            observed_at = self.now
                    if not math.isfinite(observed_at) or observed_at <= 0.0:
                        observed_at = self.now
                    normalized["_age_s"] = max(0.0, self.now - observed_at)
                    states[(str(service), str(device))] = normalized
        return states

    @staticmethod
    def _ancestors(dag, services):
        found = set()
        stack = list(services)
        while stack:
            current = stack.pop()
            for predecessor in dag.get(current, {}).get("prev_nodes", []):
                if predecessor in (START, END) or predecessor in found:
                    continue
                found.add(predecessor)
                stack.append(predecessor)
        return found

    def _sample_roots(self, candidate, seed):
        seed = int(seed)
        cached = self._sample_root_cache.get(seed)
        if cached is None:
            rng = random.Random(seed)
            base_roots = []
            for commitment in sorted(
                self.commitments,
                key=lambda item: (
                    _record_started(item, self.now),
                    str(item.get("root_uuid") or ""),
                ),
            ):
                root = str(commitment.get("root_uuid") or "")
                if not root:
                    continue
                dag = commitment["dag"]
                plan = plan_from_dag(dag)
                order = self._dag_order(dag)
                base_roots.append({
                    "root": root,
                    "source": commitment.get("source_id", ""),
                    "dag": dag,
                    "order": order,
                    "services": tuple(
                        service for service in order if service not in (START, END)
                    ),
                    "plan": plan,
                    "durations": self.latency_model.sample_task(
                        commitment.get("source_id", ""), plan, rng
                    ),
                    "handoffs": self.latency_model.sample_handoffs(plan, rng),
                    "overheads": self.latency_model.sample_stage_overheads(
                        commitment.get("source_id", ""), dag, plan, rng
                    ),
                    "source_device": str(
                        commitment.get("source_device")
                        or dag.get(START, {}).get("service", {}).get("execute_device")
                        or ""
                    ),
                    "started": (
                        _record_started(commitment, self.now)
                        if commitment.get("_slo_admitted") else self.now
                    ),
                    "simulation_now": self.now,
                    "slo": _deadline(commitment, self.default_slo_s),
                    "candidate": False,
                })
            cached = (tuple(base_roots), rng.getstate())
            self._sample_root_cache[seed] = cached
        roots = list(cached[0])
        if candidate is not None:
            rng = random.Random()
            rng.setstate(cached[1])
            dag = candidate["dag"]
            order = self._dag_order(dag)
            roots.append({
                "root": candidate["root"],
                "source": candidate.get("source", ""),
                "dag": dag,
                "order": order,
                "services": tuple(
                    service for service in order if service not in (START, END)
                ),
                "plan": candidate["plan"],
                "durations": self.latency_model.sample_task(candidate.get("source", ""), candidate["plan"], rng),
                "handoffs": self.latency_model.sample_handoffs(
                    candidate["plan"], rng
                ),
                "overheads": self.latency_model.sample_stage_overheads(
                    candidate.get("source", ""), dag, candidate["plan"], rng
                ),
                "source_device": str(
                    candidate.get("source_device")
                    or dag.get(START, {}).get("service", {}).get("execute_device")
                    or ""
                ),
                # A candidate has not entered the generator-to-distributor SLO
                # interval yet. Identity reservation and search time are not
                # application latency.
                "started": self.now,
                "simulation_now": self.now,
                "slo": float(candidate["slo"]),
                "candidate": True,
            })
        return roots

    @staticmethod
    def _node_device(root, service):
        if service == START:
            return str(root.get("source_device") or "")
        if service in root["plan"]:
            return str(root["plan"].get(service) or "")
        node = root["dag"].get(service, {})
        spec = node.get("service", {}) if isinstance(node, dict) else {}
        return str(spec.get("execute_device") or "")

    @classmethod
    def _release_delay(cls, root, service, predecessor):
        overheads = root.get("overheads", {})
        delay = float(overheads.get("control", {}).get(service, 0.0))
        source_device = cls._node_device(root, predecessor)
        target_device = cls._node_device(root, service)
        if source_device and target_device and source_device != target_device:
            delay += float(overheads.get("transfer", {}).get(service, 0.0))
        if service != END:
            delay += float(overheads.get("dispatch", {}).get(service, 0.0))
        return max(0.0, delay)

    @classmethod
    def _noqueue_latency(cls, root):
        dag = root["dag"]
        finish = {START: 0.0}
        for service in root["order"]:
            if service == START:
                continue
            predecessors = dag[service].get("prev_nodes", [])
            predecessor = max(
                predecessors,
                key=lambda item: finish.get(item, 0.0),
                default=START,
            )
            ready = finish.get(predecessor, 0.0) + cls._release_delay(
                root, service, predecessor
            )
            if service == END:
                finish[service] = ready
            else:
                finish[service] = (
                    ready
                    + float(root["durations"].get(service, 0.0))
                    + float(root["handoffs"].get(service, 0.0))
                )
        if END in finish:
            path_latency = finish[END]
        else:
            sinks = [name for name in dag if not dag[name].get("next_nodes")]
            path_latency = max(
                (finish.get(name, 0.0) for name in sinks), default=0.0
            )
        path_latency += float(
            root.get("overheads", {}).get("completion", 0.0)
        )
        return (
            max(0.0, root.get("simulation_now", 0.0) - root["started"])
            + path_latency
        )

    @staticmethod
    def _calendar_start(intervals, ready, duration):
        """Find the first non-overlapping slot in one projected FIFO calendar.

        This helper is used only for inexpensive search ordering. Existing
        committed intervals are treated as soft reservations that the new task
        does not displace; the full event simulation remains authoritative for
        the objective and captures any externality on older tasks.
        """

        cursor = float(ready)
        for start, finish in intervals:
            if finish <= cursor:
                continue
            if cursor + duration <= start:
                break
            cursor = max(cursor, finish)
        return cursor

    def screen_candidate(self, candidate, seed, baseline):
        """Cheap candidate-only projection over one committed-work calendar.

        The result is a primal-search heuristic, never a pruning bound or a
        final plan score. It lets the anytime optimizer inspect combinations
        suggested by latent committed work before spending scenario simulation
        budget on them.
        """

        roots = self._sample_roots(candidate, seed)
        root = next(item for item in roots if item["root"] == candidate["root"])
        calendars = {
            replica: list(intervals)
            for replica, intervals in baseline.get("replica_intervals", {}).items()
        }
        finish = {START: self.now}
        added_work = defaultdict(float)
        for service in root["order"]:
            if service == START:
                continue
            predecessors = root["dag"][service].get("prev_nodes", [])
            predecessor = max(
                predecessors,
                key=lambda item: finish.get(item, self.now),
                default=START,
            )
            ready = finish.get(predecessor, self.now) + self._release_delay(
                root, service, predecessor
            )
            if service == END:
                finish[service] = ready
                continue
            device = str(root["plan"].get(service) or "")
            replica = (service, device)
            duration = (
                float(root["durations"].get(service, 0.0))
                + float(root["handoffs"].get(service, 0.0))
            )
            intervals = calendars.setdefault(replica, [])
            start = self._calendar_start(intervals, ready, duration)
            end = start + duration
            intervals.append((start, end))
            intervals.sort()
            added_work[replica] += duration
            finish[service] = end

        completed = finish.get(END)
        if completed is None:
            sinks = [
                name for name in root["dag"]
                if not root["dag"][name].get("next_nodes")
            ]
            completed = max(
                (finish.get(name, self.now) for name in sinks),
                default=self.now,
            )
        completed += float(root.get("overheads", {}).get("completion", 0.0))
        latency = max(0.0, completed - root["started"])
        noqueue = self._noqueue_latency(root)
        replica_work = dict(baseline.get("replica_work", {}))
        for replica, duration in added_work.items():
            replica_work[replica] = replica_work.get(replica, 0.0) + duration
        return {
            "latency": latency,
            "noqueue": noqueue,
            "queue_inflation": max(0.0, latency - noqueue),
            "max_replica_work": max(replica_work.values(), default=0.0),
        }

    def simulate(self, candidate, seed, include_calendar=False):
        roots = self._sample_roots(candidate, seed)
        root_by_id = {root["root"]: root for root in roots}
        status = {
            root["root"]: {name: "PENDING" for name in root["services"]}
            for root in roots
        }
        finish_time = defaultdict(dict)
        queue_wait = defaultdict(float)
        queue_states = self._queue_state_cache
        queues = defaultdict(deque)
        running = {}
        events = []
        event_sequence = 0
        stage_evidence = defaultdict(set)
        replica_work = defaultdict(float)
        replica_intervals = defaultdict(list)

        def processing_duration(root_id, service, device):
            root = root_by_id.get(root_id)
            if root is None:
                return self.latency_model.estimate(service, device, 0.5)
            return float(root["durations"].get(service, self.latency_model.estimate(service, device, 0.5)))

        def handoff_duration(root_id, service, device):
            root = root_by_id.get(root_id)
            if root is None:
                return self.latency_model.estimate_handoff(
                    service, device, 0.5
                )
            return float(root["handoffs"].get(
                service,
                self.latency_model.estimate_handoff(service, device, 0.5),
            ))

        def occupancy_duration(root_id, service, device):
            return (
                processing_duration(root_id, service, device)
                + handoff_duration(root_id, service, device)
            )

        def identity_matches_revision(identity):
            revision = _identity_revision(identity)
            return not self.revision or revision is None or revision == self.revision

        # Ordered telemetry is authoritative for work already visible in a
        # processor.  It is installed before any latent commitment is released.
        for replica, state in sorted(queue_states.items()):
            service, device = replica
            age = max(0.0, float(state.get("_age_s") or 0.0))
            fresh = age <= self.queue_state_max_age_s
            running_task = state.get("running_task")
            running_root = (
                _identity_root(running_task)
                if identity_matches_revision(running_task) else ""
            )
            if running_root:
                stage_evidence[running_root].add(service)
            if fresh and state.get("busy"):
                processing = processing_duration(running_root, service, device)
                handoff = handoff_duration(running_root, service, device)
                phase = str(state.get("running_phase") or "processing")
                phase_elapsed = max(0.0, float(
                    state.get("phase_elapsed_s", state.get("running_elapsed_s"))
                    or 0.0
                )) + age
                if phase == "processing":
                    processing_remaining = processing - phase_elapsed
                    if processing_remaining <= 1e-6:
                        # A busy observation proves that this sampled duration
                        # was too short. Condition on survival instead of
                        # incorrectly releasing the replica immediately.
                        processing_remaining = self.latency_model.estimate(
                            service, device, 0.5
                        )
                    remaining = processing_remaining + handoff
                elif phase in ("handoff", "sending", "returning"):
                    remaining = max(0.0, handoff - phase_elapsed)
                    if remaining <= 1e-6:
                        remaining = max(
                            1e-3,
                            self.latency_model.estimate_handoff(
                                service, device, 0.5
                            ),
                        )
                else:
                    # Receiving/preparing precedes processor demand, so none of
                    # the predicted real_execute_time has been consumed yet.
                    remaining = processing + handoff
                running[replica] = (
                    running_root, service, self.now, self.now + remaining
                )
                replica_work[replica] += remaining
                if include_calendar:
                    replica_intervals[replica].append(
                        (self.now, self.now + remaining)
                    )
                if running_root in status and service in status[running_root]:
                    status[running_root][service] = "RUNNING"
                event_sequence += 1
                heapq.heappush(
                    events,
                    (
                        self.now + remaining,
                        event_sequence,
                        "FINISH",
                        replica,
                        running_root,
                        service,
                    ),
                )
            waiting_tasks = state.get("waiting_tasks")
            if isinstance(waiting_tasks, list):
                valid_waiting = [
                    identity for identity in waiting_tasks
                    if identity_matches_revision(identity)
                ]
                for identity in valid_waiting:
                    root_id = _identity_root(identity)
                    if root_id:
                        stage_evidence[root_id].add(service)
                    if fresh:
                        queues[replica].append((root_id, service, self.now))
                        if root_id in status and service in status[root_id]:
                            status[root_id][service] = "QUEUED"
                if fresh:
                    unknown_count = max(
                        0,
                        _non_negative_int(state.get("waiting_count"))
                        - len(waiting_tasks),
                    )
                    for index in range(unknown_count):
                        anonymous = (
                            f"__fragsplice_queue__:{service}:{device}:{index}"
                        )
                        queues[replica].append((anonymous, service, self.now))
            elif fresh:
                for index in range(_non_negative_int(state.get("waiting_count"))):
                    anonymous = f"__fragsplice_queue__:{service}:{device}:{index}"
                    queues[replica].append((anonymous, service, self.now))

        # Barrier arrivals are exact evidence that the named predecessor and
        # its ancestors have completed, even though no processor queue owns it.
        barrier_done = defaultdict(set)
        for barrier in self.barriers:
            root_id = str(barrier.get("root_uuid") or "")
            barrier_done[root_id].update(str(item) for item in barrier.get("arrived_branches", []))

        for root in roots:
            root_id = root["root"]
            dag = root["dag"]
            completed = self._ancestors(
                dag, stage_evidence[root_id] | barrier_done[root_id]
            )
            completed.update(barrier_done[root_id])
            for service in completed:
                if service in status[root_id] and status[root_id][service] == "PENDING":
                    status[root_id][service] = "DONE"
                    finish_time[root_id][service] = self.now

        def predecessors_done(root_id, service):
            root = root_by_id[root_id]
            for predecessor in root["dag"][service].get("prev_nodes", []):
                if predecessor == START:
                    continue
                if predecessor == END:
                    return False
                if status[root_id].get(predecessor) != "DONE":
                    return False
            return True

        def ready_time(root_id, service):
            root = root_by_id[root_id]
            predecessors = root["dag"][service].get("prev_nodes", [])
            predecessor = max(
                predecessors,
                key=lambda item: finish_time[root_id].get(item, self.now),
                default=START,
            )
            predecessor_finish = finish_time[root_id].get(
                predecessor, self.now
            )
            return predecessor_finish + self._release_delay(
                root, service, predecessor
            )

        def enqueue_frontier(root_id):
            nonlocal event_sequence
            root = root_by_id[root_id]
            changed = True
            while changed:
                changed = False
                for service in root["order"]:
                    if service in (START, END) or status[root_id].get(service) != "PENDING":
                        continue
                    if not predecessors_done(root_id, service):
                        continue
                    device = root["plan"].get(service)
                    if not device:
                        continue
                    replica = (service, str(device))
                    released = max(self.now, ready_time(root_id, service))
                    status[root_id][service] = "RELEASING"
                    event_sequence += 1
                    heapq.heappush(
                        events,
                        (
                            released,
                            event_sequence,
                            "RELEASE",
                            replica,
                            root_id,
                            service,
                        ),
                    )
                    changed = True

        def start_idle(at):
            nonlocal event_sequence
            for replica in sorted(set(queues) | set(queue_states)):
                if replica in running or not queues[replica]:
                    continue
                root_id, service, released = queues[replica][0]
                if released > at:
                    continue
                queues[replica].popleft()
                service_duration = occupancy_duration(
                    root_id, service, replica[1]
                )
                end = at + service_duration
                running[replica] = (root_id, service, at, end)
                replica_work[replica] += service_duration
                if include_calendar:
                    replica_intervals[replica].append((at, end))
                if root_id in status and service in status[root_id]:
                    status[root_id][service] = "RUNNING"
                    queue_wait[root_id] += max(0.0, at - released)
                event_sequence += 1
                heapq.heappush(
                    events,
                    (end, event_sequence, "FINISH", replica, root_id, service),
                )

        for root in sorted(roots, key=lambda item: (item["started"], item["root"])):
            enqueue_frontier(root["root"])
        start_idle(self.now)

        completed_roots = {}
        while events:
            at, _, event_type, replica, root_id, service = heapq.heappop(events)
            if event_type == "RELEASE":
                queues[replica].append((root_id, service, at))
                if root_id in status and service in status[root_id]:
                    status[root_id][service] = "QUEUED"
                start_idle(at)
                continue
            current = running.get(replica)
            if current is None or current[0] != root_id or current[1] != service:
                continue
            running.pop(replica, None)
            if root_id in status and service in status[root_id]:
                status[root_id][service] = "DONE"
                finish_time[root_id][service] = at
                root = root_by_id[root_id]
                enqueue_frontier(root_id)
                pending = [value for value in status[root_id].values() if value != "DONE"]
                if not pending:
                    completed_roots[root_id] = (
                        ready_time(root_id, END)
                        + float(root.get("overheads", {}).get("completion", 0.0))
                    )
            start_idle(at)

        for root in roots:
            root_id = root["root"]
            if root_id not in completed_roots:
                if status[root_id] and all(
                    value == "DONE" for value in status[root_id].values()
                ):
                    completed_roots[root_id] = (
                        ready_time(root_id, END)
                        + float(root.get("overheads", {}).get("completion", 0.0))
                    )
                else:
                    completed_roots[root_id] = (
                        max(finish_time[root_id].values(), default=self.now)
                        + float(root.get("overheads", {}).get("completion", 0.0))
                    )

        latency = {
            root["root"]: max(
                0.0, completed_roots[root["root"]] - root["started"]
            )
            for root in roots
        }
        deadlines = {root["root"]: root["slo"] for root in roots}
        candidate_root = candidate["root"] if candidate else None
        candidate_noqueue = (
            self._noqueue_latency(root_by_id[candidate_root]) if candidate_root else 0.0
        )
        result = {
            "latency": latency,
            "deadlines": deadlines,
            "queue_wait": dict(queue_wait),
            "candidate_root": candidate_root,
            "candidate_noqueue": candidate_noqueue,
            "replica_work": dict(replica_work),
        }
        if include_calendar:
            result["replica_intervals"] = {
                replica: sorted(intervals)
                for replica, intervals in replica_intervals.items()
            }
        return result


__all__ = (
    "FragSpliceExecutionState",
    "plan_from_dag",
    "service_names",
    "topological_order",
)
