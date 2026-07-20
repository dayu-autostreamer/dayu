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


def _record_created(record, fallback):
    for key in ("created_at", "reserved_at", "admitted_at"):
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
            list(self.snapshot.get("reservations", []))
            + list(self.snapshot.get("commitments", []))
        )
        for item in ordered_records:
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
            records[root] = normalized
        self.commitments = list(records.values())
        self.barriers = [
            item for item in self.snapshot.get("task_barriers", [])
            if isinstance(item, dict)
        ]

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
        rng = random.Random(int(seed))
        roots = []
        for commitment in sorted(
            self.commitments,
            key=lambda item: (
                _record_created(item, self.now),
                str(item.get("root_uuid") or ""),
            ),
        ):
            root = str(commitment.get("root_uuid") or "")
            if not root:
                continue
            dag = commitment["dag"]
            plan = plan_from_dag(dag)
            roots.append({
                "root": root,
                "source": commitment.get("source_id", ""),
                "dag": dag,
                "plan": plan,
                "durations": self.latency_model.sample_task(commitment.get("source_id", ""), plan, rng),
                "handoffs": self.latency_model.sample_handoffs(plan, rng),
                "created": _record_created(commitment, self.now),
                "simulation_now": self.now,
                "slo": _deadline(commitment, self.default_slo_s),
                "candidate": False,
            })
        if candidate is not None:
            roots.append({
                "root": candidate["root"],
                "source": candidate.get("source", ""),
                "dag": candidate["dag"],
                "plan": candidate["plan"],
                "durations": self.latency_model.sample_task(candidate.get("source", ""), candidate["plan"], rng),
                "handoffs": self.latency_model.sample_handoffs(
                    candidate["plan"], rng
                ),
                "created": _record_created(candidate, self.now),
                "simulation_now": self.now,
                "slo": float(candidate["slo"]),
                "candidate": True,
            })
        return roots

    @staticmethod
    def _noqueue_latency(root):
        dag = root["dag"]
        finish = {START: 0.0}
        for service in topological_order(dag):
            if service == START:
                continue
            predecessors = dag[service].get("prev_nodes", [])
            ready = max((finish.get(item, 0.0) for item in predecessors), default=0.0)
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
        return max(0.0, root.get("simulation_now", 0.0) - root["created"]) + path_latency

    def simulate(self, candidate, seed):
        roots = self._sample_roots(candidate, seed)
        root_by_id = {root["root"]: root for root in roots}
        status = {
            root["root"]: {name: "PENDING" for name in service_names(root["dag"])}
            for root in roots
        }
        finish_time = defaultdict(dict)
        queue_wait = defaultdict(float)
        queue_states = self._queue_states()
        queues = defaultdict(deque)
        running = {}
        events = []
        event_sequence = 0
        stage_evidence = defaultdict(set)
        replica_work = defaultdict(float)

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
                if running_root in status and service in status[running_root]:
                    status[running_root][service] = "RUNNING"
                event_sequence += 1
                heapq.heappush(
                    events,
                    (
                        self.now + remaining,
                        event_sequence,
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
            return max(
                (finish_time[root_id].get(item, self.now) for item in root["dag"][service].get("prev_nodes", [])),
                default=self.now,
            )

        def enqueue_frontier(root_id):
            root = root_by_id[root_id]
            changed = True
            while changed:
                changed = False
                for service in topological_order(root["dag"]):
                    if service in (START, END) or status[root_id].get(service) != "PENDING":
                        continue
                    if not predecessors_done(root_id, service):
                        continue
                    device = root["plan"].get(service)
                    if not device:
                        continue
                    replica = (service, str(device))
                    released = max(self.now, ready_time(root_id, service))
                    queues[replica].append((root_id, service, released))
                    status[root_id][service] = "QUEUED"
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
                if root_id in status and service in status[root_id]:
                    status[root_id][service] = "RUNNING"
                    queue_wait[root_id] += max(0.0, at - released)
                event_sequence += 1
                heapq.heappush(events, (end, event_sequence, replica, root_id, service))

        for root in sorted(roots, key=lambda item: (item["created"], item["root"])):
            enqueue_frontier(root["root"])
        start_idle(self.now)

        completed_roots = {}
        while events:
            at, _, replica, root_id, service = heapq.heappop(events)
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
                    completed_roots[root_id] = at
            start_idle(at)

        for root in roots:
            root_id = root["root"]
            if root_id not in completed_roots:
                completed_roots[root_id] = max(finish_time[root_id].values(), default=self.now)

        latency = {
            root["root"]: max(
                0.0, completed_roots[root["root"]] - root["created"]
            )
            for root in roots
        }
        deadlines = {root["root"]: root["slo"] for root in roots}
        candidate_root = candidate["root"] if candidate else None
        candidate_noqueue = (
            self._noqueue_latency(root_by_id[candidate_root]) if candidate_root else 0.0
        )
        return {
            "latency": latency,
            "deadlines": deadlines,
            "queue_wait": dict(queue_wait),
            "candidate_root": candidate_root,
            "candidate_noqueue": candidate_noqueue,
            "replica_work": dict(replica_work),
        }


__all__ = (
    "FragSpliceExecutionState",
    "plan_from_dag",
    "service_names",
    "topological_order",
)
