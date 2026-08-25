import copy
import math
import threading
import time

from core.lib.common import Context, LOGGER, ResourceLockManager
from core.lib.scheduling import (
    SchedulingSnapshotScope,
    normalize_scheduling_snapshot_scope,
)
from core.lib.scheduling.deployment_plan import validate_plan
from core.lib.runtime import RuntimeContext, TaskBarrierStore

from .runtime_directory import (
    RedisRuntimeDirectoryStore,
    RuntimeDirectoryConflict,
    RuntimeDirectoryError,
    RuntimeDirectorySnapshot,
    create_runtime_directory_store,
    retirement_deadline,
)
from .task_lease import RedisTaskLeaseStore, create_task_lease_store


class Scheduler:
    def __init__(
        self,
        runtime_context=None,
        runtime_directory=None,
        task_lease_store=None,
        task_barrier_store=None,
    ):
        self.schedule_table = {}
        self.resource_table = {}
        self._resource_received_at = {}
        self._resource_runtime_revision = {}

        self.resource_lock_manager = ResourceLockManager()
        self._runtime_state_lock = threading.RLock()
        self._scheduling_state_lock = threading.RLock()
        # Pending task-bound plans are in-process bookkeeping.  Keep them
        # separate from the resource/schedule-table lock so a slow durable
        # lease admission cannot stall the next /schedule request.  Redis
        # lease operations are already atomic inside their Lua transactions;
        # the in-memory store owns its own lock as well.
        self._staged_reservation_lock = threading.RLock()
        self._schedule_decision_lock = threading.RLock()

        self.runtime_context = runtime_context or RuntimeContext.get_default()
        initial_directory = runtime_directory
        if initial_directory is None:
            initial_directory = self.runtime_context.bootstrap.get("runtime_directory")
        if initial_directory is None:
            bootstrap_routes = self.runtime_context.bootstrap.get("runtime_routes")
            if bootstrap_routes:
                initial_directory = {
                    "install_id": self.runtime_context.install_id,
                    "revision": self.runtime_context.directory_revision,
                    "routes": bootstrap_routes,
                }
        if runtime_directory is not None and hasattr(runtime_directory, "snapshot_model"):
            self.runtime_directory = runtime_directory
        else:
            self.runtime_directory = create_runtime_directory_store(
                self.runtime_context,
                initial=initial_directory,
            )
        # RuntimeDirectory revisions are immutable and all active-revision
        # mutations are serialized through this Scheduler.  Keep the current
        # snapshot in-process so the per-task schedule path does not pay a
        # remote Redis GET for the same revision several times.  A Pod restart
        # rebuilds this cache from the durable store below.
        self._runtime_snapshot_cache = self.runtime_directory.snapshot_model()
        self.task_leases = task_lease_store or create_task_lease_store(self.runtime_context)
        if task_barrier_store is not None:
            self.task_barriers = task_barrier_store
        elif isinstance(self.task_leases, RedisTaskLeaseStore):
            self.task_barriers = TaskBarrierStore(
                self.task_leases.redis,
                ttl_seconds=self.runtime_context.lease_ttl_seconds,
            )
        else:
            self.task_barriers = None
        self._runtime_clock = (
            getattr(self.task_leases, "clock", None)
            or getattr(self.task_leases, "_clock", None)
            or time.time
        )

        self.cloud_device = self.runtime_context.cloud_node or self.runtime_context.local_node

        self.config_extraction = Context.get_algorithm('SCH_CONFIG_EXTRACTION')
        self.scenario_retrieval = Context.get_algorithm('SCH_SCENARIO_RETRIEVAL')
        self.policy_retrieval = Context.get_algorithm('SCH_POLICY_RETRIEVAL')
        self.startup_policy = Context.get_algorithm('SCH_STARTUP_POLICY')

        self.extract_necessary_configuration_setting()

    def extract_necessary_configuration_setting(self):
        self.config_extraction(self)

    def get_scenario_from_task(self, task):
        return self.scenario_retrieval(task)

    def get_policy_from_task(self, task):
        return self.policy_retrieval(task)

    def get_startup_policy(self, info):
        startup_info = copy.deepcopy(info)
        startup_info.setdefault('cloud_device', self.cloud_device)
        return self.startup_policy(startup_info)

    def add_scheduler_agent(self, source_id):
        agent = Context.get_algorithm('SCH_AGENT', system=self, agent_id=source_id)
        self.schedule_table[source_id] = agent
        threading.Thread(target=agent.run).start()
        return agent

    def register_schedule_table(self, source_id):
        self._ensure_scheduling_state()
        with self._scheduling_state_lock:
            if source_id in self.schedule_table:
                return self.schedule_table[source_id]
            return self.add_scheduler_agent(source_id)

    def schedule_transaction(self):
        """Serialize task-aware scheduling decisions and their reservations."""

        self._ensure_scheduling_state()
        return self._schedule_decision_lock

    def get_schedule_plan(self, info):
        source_id = info['source_id']
        agent = self.schedule_table[source_id]

        plan = agent.get_schedule_plan(info)

        if plan is None:
            # ``None`` means that this per-task extension has no update for
            # the current request.  The server composes this empty partial
            # plan with the Generator's current configuration and DAG.  A
            # startup policy is consulted there only when the current DAG is
            # not yet routable.
            LOGGER.debug('No schedule update; preserve the current plan')
            plan = {}

        # LOGGER.info(f'[Schedule Plan] Source {source_id}: {plan}')

        return plan

    def runtime_directory_snapshot(self):
        return copy.deepcopy(self._runtime_snapshot_model().to_dict())

    def runtime_directory_revision(self):
        return self._runtime_snapshot_model().revision

    def _runtime_snapshot_model(self):
        """Return the immutable active snapshot without remote I/O.

        Low-level tests may instantiate ``Scheduler`` through ``__new__``;
        lazily initialize the cache for those callers as well.
        """

        if not hasattr(self, '_runtime_state_lock'):
            self._runtime_state_lock = threading.RLock()
        with self._runtime_state_lock:
            snapshot = getattr(self, '_runtime_snapshot_cache', None)
            if snapshot is None:
                snapshot = self.runtime_directory.snapshot_model()
                self._runtime_snapshot_cache = snapshot
            return snapshot

    def _set_runtime_snapshot_cache(self, value):
        snapshot = (
            value
            if isinstance(value, RuntimeDirectorySnapshot)
            else RuntimeDirectorySnapshot.from_value(value)
        )
        self._runtime_snapshot_cache = snapshot
        return snapshot

    def runtime_routes(self, component=None, target_node=None, logical_service=None):
        return [
            route.to_dict()
            for route in self._runtime_snapshot_model().find(
                component=component,
                target_node=target_node,
                logical_service=logical_service,
            )
        ]

    def runtime_service_nodes(self):
        return self._runtime_snapshot_model().processor_deployment()

    def runtime_nodes_for_service(self, logical_service):
        return list(self.runtime_service_nodes().get(str(logical_service), []))

    def resolve_runtime_route(self, component, target_node=None, logical_service=None):
        return self._runtime_snapshot_model().resolve(
            component=component,
            target_node=target_node,
            logical_service=logical_service,
        ).to_dict()

    def compact_runtime_routes(self, plan, source_device=""):
        return self._runtime_snapshot_model().compact_routes_for_plan(
            plan,
            source_device=source_device,
            cloud_node=self.cloud_device,
        )

    def schedule_runtime_state(self, plan, source_device=""):
        """Return routes, deployment and revision from one directory snapshot."""

        with self._runtime_state_lock:
            snapshot = self._runtime_snapshot_model()
            return {
                "revision": snapshot.revision,
                "hash": snapshot.content_hash,
                "deployment": snapshot.processor_deployment(),
                "routes": snapshot.compact_routes_for_plan(
                    plan,
                    source_device=source_device,
                    cloud_node=self.cloud_device,
                ),
            }

    def replace_runtime_directory(self, directory, expected_revision):
        # A schedule transaction must observe one immutable active revision
        # from state reconstruction through reservation creation.
        with self._schedule_decision_lock, self._runtime_state_lock:
            result = self.runtime_directory.replace(directory, expected_revision)
            self._set_runtime_snapshot_cache(result)
            return result

    def propose_runtime_directory(self, directory, base_revision, proposal_id=None, ttl_seconds=60.0):
        with self._runtime_state_lock:
            return self.runtime_directory.propose(
                directory,
                base_revision=base_revision,
                proposal_id=proposal_id,
                ttl_seconds=ttl_seconds,
            )

    def commit_runtime_directory(
        self,
        proposal_id,
        expected_revision,
        retirement_grace_seconds,
    ):
        """Commit N+1 and establish the immutable retirement bound for N.

        The shared lock is sufficient for the in-memory implementation. In
        production both stores use Redis, where one Lua transaction performs
        the directory CAS, retirement marker write, and lease-score clamp.
        """

        with self._schedule_decision_lock, self._runtime_state_lock:
            try:
                expected_revision = int(expected_revision)
            except (TypeError, ValueError) as exc:
                raise RuntimeDirectoryError(
                    "retiring runtime directory revision must be an integer"
                ) from exc
            if expected_revision < 1:
                raise RuntimeDirectoryError(
                    "retiring runtime directory revision must be positive"
                )
            now = float(self._runtime_clock())
            deadline = retirement_deadline(now, retirement_grace_seconds)
            if (
                isinstance(self.runtime_directory, RedisRuntimeDirectoryStore)
                and isinstance(self.task_leases, RedisTaskLeaseStore)
            ):
                if (
                    self.runtime_directory.install_id != self.task_leases.install_id
                    or self.runtime_directory._active_key != self.task_leases._active_key
                ):
                    raise RuntimeDirectoryError(
                        "runtime directory and task leases do not share one Redis scope"
                    )
                result = self.runtime_directory.commit_with_retirement(
                    proposal_id,
                    expected_revision,
                    retirement_grace_seconds,
                    lease_key=self.task_leases._key(int(expected_revision)),
                    retirement_key=self.task_leases._retirement_key(
                        int(expected_revision)
                    ),
                    now=now,
                )
                self._set_runtime_snapshot_cache(result)
                return result

            directory = self.runtime_directory.commit(
                proposal_id,
                expected_revision,
            )
            status = self.task_leases.retire(expected_revision, deadline)
            result = dict(directory)
            result["retirement"] = status
            self._set_runtime_snapshot_cache(directory)
            return result

    def reject_runtime_directory(self, proposal_id, reason=""):
        with self._runtime_state_lock:
            return self.runtime_directory.reject(proposal_id, reason)

    def clear_runtime_directory(self, install_id):
        with self._schedule_decision_lock, self._runtime_state_lock:
            result = self.runtime_directory.clear(install_id)
            self._runtime_snapshot_cache = RuntimeDirectorySnapshot.empty()
            return result

    def _ensure_scheduling_state(self):
        # A few low-level tests construct Scheduler through __new__. Keeping the
        # state initialization lazy also makes this helper safe for subclasses.
        if not hasattr(self, '_scheduling_state_lock'):
            self._scheduling_state_lock = threading.RLock()
        if not hasattr(self, '_staged_reservation_lock'):
            self._staged_reservation_lock = threading.RLock()
        if not hasattr(self, '_schedule_decision_lock'):
            self._schedule_decision_lock = threading.RLock()
        if not hasattr(self, '_resource_received_at'):
            self._resource_received_at = {}
        if not hasattr(self, '_resource_runtime_revision'):
            self._resource_runtime_revision = {}
        if not hasattr(self, 'resource_table'):
            self.resource_table = {}
        if not hasattr(self, 'task_barriers'):
            self.task_barriers = None
        if not hasattr(self, '_staged_task_reservations'):
            self._staged_task_reservations = {}
        if not hasattr(self, '_runtime_clock'):
            self._runtime_clock = time.time

    @staticmethod
    def _task_barrier_requests(commitments):
        requests = []
        seen = set()
        for commitment in commitments:
            if not isinstance(commitment, dict):
                continue
            root_uuid = str(commitment.get('root_uuid') or '')
            dag = commitment.get('dag')
            if not root_uuid or not isinstance(dag, dict):
                continue
            for barrier, node in dag.items():
                if not isinstance(node, dict):
                    continue
                predecessors = sorted({
                    str(item) for item in node.get('prev_nodes', [])
                    if str(item or '').strip()
                })
                if len(predecessors) < 2:
                    continue
                identity = (root_uuid, str(barrier))
                if identity in seen:
                    continue
                seen.add(identity)
                requests.append({
                    'root_uuid': root_uuid,
                    'barrier': str(barrier),
                    'expected_branches': predecessors,
                    'required_count': len(predecessors),
                })
        return requests

    @staticmethod
    def _normalize_task_commitment(commitment, revision, root_uuid):
        if commitment is None:
            commitment = {}
        if not isinstance(commitment, dict):
            raise RuntimeDirectoryError('task commitment must be an object')
        normalized = copy.deepcopy(commitment)
        committed_root = str(normalized.get('root_uuid') or root_uuid)
        if committed_root != str(root_uuid):
            raise RuntimeDirectoryError('task commitment root_uuid does not match lease')
        committed_revision = normalized.get('runtime_directory_revision', revision)
        try:
            committed_revision = int(committed_revision)
        except (TypeError, ValueError) as exc:
            raise RuntimeDirectoryError(
                'task commitment runtime_directory_revision must be an integer'
            ) from exc
        if committed_revision != int(revision):
            raise RuntimeDirectoryError(
                'task commitment runtime_directory_revision does not match lease'
            )
        normalized['root_uuid'] = committed_root
        normalized['runtime_directory_revision'] = committed_revision
        return normalized

    @staticmethod
    def _reservation_matches_commitment(reservation, commitment):
        for field in (
            'source_id',
            'task_id',
            'decision_id',
            'plan_digest',
            'deployment_version',
        ):
            expected = reservation.get(field)
            actual = commitment.get(field)
            if expected not in (None, '') and expected != actual:
                return False
        return True

    def _prune_staged_reservations_locked(self, now=None):
        now = float(self._runtime_clock() if now is None else now)
        self._staged_task_reservations = {
            key: record
            for key, record in self._staged_task_reservations.items()
            if float(record.get('expires_at') or 0.0) > now
        }

    def stage_task_context(self, revision, root_uuid, context, ttl_seconds=None):
        """Stage a pre-admission plan without a synchronous Redis write.

        A schedule response is not yet an execution commitment: the Generator
        may still fail to materialize or submit the Task.  The immutable plan
        is therefore kept in the owning Scheduler for the short schedule-to-
        lease window.  ``acquire_task_lease`` receives the full commitment and
        persists it atomically once the Task actually exists.
        """

        try:
            revision = int(revision)
        except (TypeError, ValueError) as exc:
            raise RuntimeDirectoryError(
                'task reservation runtime_directory_revision must be an integer'
            ) from exc
        root_uuid = str(root_uuid or '').strip()
        if revision < 1:
            raise RuntimeDirectoryError(
                'task reservation runtime_directory_revision must be positive'
            )
        if not root_uuid:
            raise RuntimeDirectoryError('task reservation root_uuid is required')
        normalized = self._normalize_task_commitment(context, revision, root_uuid)
        ttl_seconds = (
            getattr(getattr(self, 'runtime_context', None), 'lease_ttl_seconds', 60.0)
            if ttl_seconds is None else ttl_seconds
        )
        try:
            ttl_seconds = float(ttl_seconds)
        except (TypeError, ValueError) as exc:
            raise RuntimeDirectoryError(
                'task reservation ttl_seconds must be numeric'
            ) from exc
        if not math.isfinite(ttl_seconds) or ttl_seconds <= 0.0:
            raise RuntimeDirectoryError(
                'task reservation ttl_seconds must be finite and positive'
            )

        self._ensure_scheduling_state()
        now = float(self._runtime_clock())
        key = (revision, root_uuid)
        with self._staged_reservation_lock:
            self._prune_staged_reservations_locked(now)
            previous = self._staged_task_reservations.get(key)
            if previous is not None and previous.get('context') != normalized:
                raise RuntimeDirectoryConflict(
                    'task reservation changed for an existing root_uuid'
                )
            reserved_at = (
                float(previous['reserved_at']) if previous is not None else now
            )
            record = {
                'context': normalized,
                'reserved_at': reserved_at,
                'expires_at': now + ttl_seconds,
            }
            self._staged_task_reservations[key] = record
            return {
                **copy.deepcopy(normalized),
                'reserved_at': reserved_at,
                'expires_at': record['expires_at'],
                'status': 'pending',
            }

    def _staged_task_reservation(self, revision, root_uuid):
        self._ensure_scheduling_state()
        key = (int(revision), str(root_uuid or '').strip())
        with self._staged_reservation_lock:
            self._prune_staged_reservations_locked()
            record = self._staged_task_reservations.get(key)
            if record is None:
                return None
            return {
                **copy.deepcopy(record['context']),
                'reserved_at': float(record['reserved_at']),
                'expires_at': float(record['expires_at']),
                'status': 'pending',
            }

    def _staged_task_reservation_snapshot(self, revision):
        self._ensure_scheduling_state()
        revision = int(revision)
        with self._staged_reservation_lock:
            self._prune_staged_reservations_locked()
            return [
                {
                    **copy.deepcopy(record['context']),
                    'reserved_at': float(record['reserved_at']),
                    'expires_at': float(record['expires_at']),
                    'status': 'pending',
                }
                for (record_revision, _), record
                in self._staged_task_reservations.items()
                if record_revision == revision
            ]

    def _drop_staged_task_reservation(self, revision, root_uuid):
        self._ensure_scheduling_state()
        key = (int(revision), str(root_uuid or '').strip())
        with self._staged_reservation_lock:
            return self._staged_task_reservations.pop(key, None)

    def reserve_task_context(self, revision, root_uuid, context, ttl_seconds=None):
        normalized = self._normalize_task_commitment(context, revision, root_uuid)
        ttl_seconds = (
            self.runtime_context.lease_ttl_seconds
            if ttl_seconds is None else ttl_seconds
        )
        self._ensure_scheduling_state()
        if isinstance(self.task_leases, RedisTaskLeaseStore):
            return self.task_leases.reserve(
                revision=revision,
                root_uuid=root_uuid,
                context=normalized,
                active_revision=None,
                ttl_seconds=ttl_seconds,
            )
        with self._runtime_state_lock:
            active_revision = self.runtime_directory_revision()
        return self.task_leases.reserve(
            revision=revision,
            root_uuid=root_uuid,
            context=normalized,
            active_revision=active_revision,
            ttl_seconds=ttl_seconds,
        )

    def cancel_task_reservation(self, revision, root_uuid, decision_id=None):
        """Remove a task-bound plan that never materialized into a lease."""

        self._ensure_scheduling_state()
        staged = self._staged_task_reservation(revision, root_uuid)
        if staged is not None:
            expected_decision = str(staged.get('decision_id') or '')
            requested_decision = str(decision_id or '')
            if requested_decision and requested_decision != expected_decision:
                raise RuntimeDirectoryConflict(
                    'task reservation decision_id does not match cancellation request'
                )
            self._drop_staged_task_reservation(revision, root_uuid)
        result = self.task_leases.cancel_reservation(
            revision,
            root_uuid,
            decision_id=decision_id,
        )
        if staged is not None:
            result = dict(result)
            result['already_cancelled'] = False
            result['decision_id'] = str(staged.get('decision_id') or '')
        return result

    def get_task_reservation(self, revision, root_uuid, task_context=None):
        """Return the current task-bound decision, if one is still pending."""

        try:
            revision = int(revision)
        except (TypeError, ValueError) as exc:
            raise RuntimeDirectoryError(
                'task reservation runtime_directory_revision must be an integer'
            ) from exc
        root_uuid = str(root_uuid or '').strip()
        if not root_uuid:
            return None
        task_context = task_context if isinstance(task_context, dict) else {}
        reservation = self._staged_task_reservation(revision, root_uuid)
        if reservation is None:
            self._ensure_scheduling_state()
            reservation = self.task_leases.get_reservation(revision, root_uuid)
        if reservation is None:
            return None
        for field in ('source_id', 'task_id'):
            expected = reservation.get(field)
            actual = task_context.get(field)
            if expected not in (None, '') and expected != actual:
                raise RuntimeDirectoryConflict(
                    f'task reservation {field} does not match schedule request'
                )
        return copy.deepcopy(reservation)

    def acquire_task_lease(
        self,
        revision,
        root_uuid,
        ttl_seconds=60.0,
        commitment=None,
    ):
        # Validate the optional scheduling payload before changing lease state.
        # This keeps malformed admission requests from leaving a valid lease
        # behind even though the request itself is rejected.
        normalized_commitment = self._normalize_task_commitment(
            commitment,
            revision,
            root_uuid,
        )
        self._ensure_scheduling_state()
        staged = self._staged_task_reservation(revision, root_uuid)
        if staged is not None and not self._reservation_matches_commitment(
            staged,
            normalized_commitment,
        ):
            raise RuntimeDirectoryConflict(
                'task execution context does not match its reservation'
            )
        if isinstance(self.task_leases, RedisTaskLeaseStore):
            # The Redis Lua transaction reads the active directory and admits
            # the lease atomically.  Do not hold either local Scheduler state
            # lock across that network round trip: doing so couples admission
            # latency back into the next task's /schedule path.
            lease = self.task_leases.acquire(
                revision=revision,
                root_uuid=root_uuid,
                active_revision=None,
                ttl_seconds=ttl_seconds,
                context=normalized_commitment,
            )
        else:
            with self._runtime_state_lock:
                active_revision = self.runtime_directory_revision()
            lease = self.task_leases.acquire(
                revision=revision,
                root_uuid=root_uuid,
                active_revision=active_revision,
                ttl_seconds=ttl_seconds,
                context=normalized_commitment,
            )
        with self._staged_reservation_lock:
            self._staged_task_reservations.pop(
                (int(revision), str(root_uuid or '').strip()),
                None,
            )
        return lease

    def renew_task_lease(self, revision, root_uuid, ttl_seconds=60.0):
        # Existing work may renew an inactive revision only while the atomic
        # directory commit's persisted retirement bound remains open.
        if isinstance(self.task_leases, RedisTaskLeaseStore):
            lease = self.task_leases.renew(
                revision,
                root_uuid,
                ttl_seconds=ttl_seconds,
            )
        else:
            with self._runtime_state_lock:
                lease = self.task_leases.renew(
                    revision,
                    root_uuid,
                    ttl_seconds=ttl_seconds,
                    active_revision=self.runtime_directory_revision(),
                )
        return lease

    def release_task_lease(self, revision, root_uuid):
        result = self.task_leases.release(revision, root_uuid)
        return result

    def count_task_leases(self, revision):
        return self.task_leases.count(revision)

    def task_lease_status(self, revision):
        return self.task_leases.status(revision)

    def retire_task_leases(self, revision, deadline):
        # Shares the directory state critical section so an in-memory
        # Scheduler cannot admit a task while the same revision is fenced.
        # Redis stores additionally enforce this ordering inside Lua across
        # Scheduler replicas.
        with self._runtime_state_lock:
            return self.task_leases.retire(revision, deadline)

    def update_scheduler_scenario(self, task):
        source_id = task.get_source_id()
        if source_id not in self.schedule_table:
            LOGGER.warning(f'Scheduler agent for source {source_id} not exists!')
            return False
        scenario = self.get_scenario_from_task(task)
        policy = self.get_policy_from_task(task)
        agent = self.schedule_table[source_id]
        agent.update_scenario(scenario)
        agent.update_policy(policy)
        agent.update_task(task)
        # LOGGER.info(f'[Update Scenario] Source {source_id}: {scenario}')
        return True

    def register_resource_table(self, device):
        self._ensure_scheduling_state()
        with self._scheduling_state_lock:
            if device in self.resource_table:
                return
            self.resource_table[device] = {}

    def update_scheduler_resource(self, info):
        device = info['device']
        resource = copy.deepcopy(info['resource'])
        reported_revision = info.get('runtime_directory_revision')
        try:
            reported_revision = int(reported_revision)
        except (TypeError, ValueError):
            reported_revision = None
        if reported_revision is not None and reported_revision < 1:
            reported_revision = None
        self._ensure_scheduling_state()
        received_at = time.time()
        with self._scheduling_state_lock:
            self.resource_table[device] = resource
            self._resource_received_at[device] = received_at
            self._resource_runtime_revision[device] = reported_revision
            agents = list(self.schedule_table.values())

        for agent in agents:
            agent.update_resource(device, resource)

        # LOGGER.info(f'[Update Resource] Device {device}: {resource}')

    def get_scheduler_resource(self):
        self._ensure_scheduling_state()
        with self._scheduling_state_lock:
            return copy.deepcopy(self.resource_table)

    def get_scheduling_snapshot(
        self,
        scope=SchedulingSnapshotScope.COMMITTED,
    ):
        """Return a mutation-safe runtime view at the requested scope.

        ``LIVE`` contains deployment and telemetry only. ``COMMITTED`` also
        includes staged reservations, active commitments and join barriers.
        Extensions select the smallest scope their decision actually needs;
        the framework does not infer scope from an algorithm identity.
        """

        scope = normalize_scheduling_snapshot_scope(scope)
        self._ensure_scheduling_state()
        with self._scheduling_state_lock:
            # Read the revision and its deployment while directory commits are
            # excluded.  A scheduling snapshot must never combine revision N
            # commitments with revision N+1 replicas.
            with self._runtime_state_lock:
                directory_revision = self.runtime_directory_revision()
                deployment = copy.deepcopy(self.runtime_service_nodes())
            captured_at = time.time()
            resources = copy.deepcopy(self.resource_table)
            resource_received_at = copy.deepcopy(self._resource_received_at)
            resource_runtime_revision = copy.deepcopy(
                self._resource_runtime_revision
            )
        if scope is SchedulingSnapshotScope.LIVE:
            return {
                'captured_at': captured_at,
                'runtime_directory_revision': directory_revision,
                'deployment': deployment,
                'resources': resources,
                'resource_received_at': resource_received_at,
                'resource_runtime_revision': resource_runtime_revision,
                'reservations': [],
                'commitments': [],
                'task_barriers': [],
            }
        # Redis list operations are already atomic per command and may require
        # network I/O.  Keeping the global Scheduler state lock while reading
        # every in-flight context stalls the millisecond-scale /schedule fast
        # path.  Revision filtering below makes a concurrent directory change
        # harmless: a stale snapshot can only publish a stale-revision rolling
        # plan, which the request path rejects before consumption.
        persisted_reservations = [
            copy.deepcopy(record)
            for record in self.task_leases.list_reservations()
            if record.get('runtime_directory_revision') == directory_revision
        ]
        reservation_by_root = {
            str(record.get('root_uuid') or ''): record
            for record in persisted_reservations
        }
        # The local staged record is newer than a durable compatibility record
        # for the same root.  These entries exist only until lease admission.
        for record in self._staged_task_reservation_snapshot(directory_revision):
            reservation_by_root[str(record.get('root_uuid') or '')] = record
        reservations = list(reservation_by_root.values())
        # Task leases deliberately retain old revisions while their immutable
        # routes drain. Scheduling decisions for revision N must not project
        # those tasks onto revision N+1 processor queues.
        commitments = [
            copy.deepcopy(record)
            for record in self.task_leases.list_active()
            if record.get('runtime_directory_revision') == directory_revision
        ]
        reservations.sort(key=lambda item: (
            float(item.get('reserved_at') or 0.0),
            str(item.get('root_uuid') or ''),
        ))
        commitments.sort(key=lambda item: (
            float(item.get('admitted_at') or 0.0),
            str(item.get('root_uuid') or ''),
        ))
        task_barriers = (
            self.task_barriers.snapshot(self._task_barrier_requests(commitments))
            if self.task_barriers is not None
            else []
        )
        return {
            'captured_at': captured_at,
            'runtime_directory_revision': directory_revision,
            'deployment': deployment,
            'resources': resources,
            'resource_received_at': resource_received_at,
            'resource_runtime_revision': resource_runtime_revision,
            'reservations': reservations,
            'commitments': commitments,
            'task_barriers': task_barriers,
        }

    async def get_resource_lock(self, info):
        return await self.resource_lock_manager.acquire_lock(
            info['resource'], info['device']
        )

    def get_source_node_selection_plan(self, source_id, data):
        agent = self.schedule_table[source_id]
        plan = agent.get_source_selection_plan(data)
        return plan

    def get_initial_deployment_plan(self, source_id, data):
        agent = self.schedule_table[source_id]
        plan = agent.get_initial_deployment_plan(data)
        try:
            return validate_plan(plan, data, cloud_node=self.cloud_device)
        except ValueError as exc:
            raise RuntimeDirectoryError(str(exc)) from exc

    def get_redeployment_plan(self, source_id, data):
        agent = self.schedule_table[source_id]
        plan = agent.get_redeployment_plan(data)
        try:
            return validate_plan(plan, data, cloud_node=self.cloud_device)
        except ValueError as exc:
            raise RuntimeDirectoryError(str(exc)) from exc

    @staticmethod
    def _normalize_generation_decision(decision):
        if isinstance(decision, bool):
            return {
                "generate": bool(decision),
                "reason": "agent_bool",
            }
        if not isinstance(decision, dict):
            return {
                "generate": True,
                "reason": "default_allow_invalid_decision",
            }

        generate = decision.get("generate", decision.get("allow", True))
        normalized = dict(decision)
        normalized["generate"] = bool(generate)
        normalized.setdefault("reason", "agent_decision")
        return normalized

    def should_generate(self, source_id, data):
        # Runtime services start reporting telemetry before the backend has
        # atomically committed the first complete runtime directory.  A
        # generator may therefore reach this endpoint while revision 0 still
        # exposes only the bootstrap view.  No task can be routed or leased
        # against that view, so keep generation closed instead of invoking an
        # agent with a transient, incomplete deployment.
        revision = self.runtime_directory_revision()
        if revision < 1:
            return {
                "generate": False,
                "reason": "runtime_directory_not_ready",
                "runtime_directory_revision": revision,
            }

        agent = self.schedule_table[source_id]
        hook = getattr(agent, "should_generate", None)
        if not callable(hook):
            return {
                "generate": True,
                "reason": "default_allow_no_hook",
                "cache_for_s": 2.0,
                "runtime_directory_revision": revision,
            }
        decision = self._normalize_generation_decision(hook(data))
        revision = self.runtime_directory_revision()
        decision["runtime_directory_revision"] = revision
        # Actions are opaque extension data.  When an extension emits one,
        # attach the current generic runtime context once at the response
        # boundary.  Action executors may resolve whatever components they
        # need without teaching Scheduler about action types or targets.
        if decision.get("actions") or decision.get("commands"):
            decision["runtime_directory"] = self.runtime_directory_snapshot()
        return decision

    def get_schedule_overhead(self):
        overheads = []
        for source_id in self.schedule_table:
            agent = self.schedule_table[source_id]
            overheads.append(agent.get_schedule_overhead())

        return sum(overheads) / len(overheads) if overheads else 0
