import copy
import hashlib
import json
import threading
import time

from core.lib.common import Context, Counter, LOGGER
from core.lib.content import Task, TaskIdentity
from core.lib.network import deliver_task, http_request, NetworkAPIPath, NetworkAPIMethod
from core.lib.estimation import TimeEstimator
from core.lib.scheduling import build_schedule_decision
from core.lib.runtime import (
    RuntimeContext,
    RuntimeLeaseError,
    RuntimeLeaseClient,
    RuntimeLeaseRetired,
    RuntimeLeaseUnavailable,
    RuntimeResolver,
)


_HOST_SCHEDULE_CONTEXT_FIELDS = (
    'source_id',
    'meta_data',
    'current_configuration',
    'source_device',
    'all_edge_devices',
    'dag',
    'deployment_version',
    'runtime_directory_revision',
    'runtime_directory_hash',
    'runtime_route_cache_keys',
)
_SCHEDULER_REQUEST_TIMEOUT_SECONDS = 5.0


class Generator:
    def __init__(self, source_id: int, metadata: dict, task_dag: dict, ):
        """ Initialize the generator."""

        """task base information"""
        # source_id points to the corresponding source and is unique for generator
        self.source_id = source_id
        # task_dag contains offloading decisions
        self.task_dag = Task.extract_dag_from_dict(task_dag)
        # service_deployment contains service deployment situations in system
        self.service_deployment = None
        # Optional scheduler-supplied deployment version for attributing task feedback.
        # Version 0 means the scheduler does not distinguish deployment versions.
        self.deployment_version = 0
        self.runtime_directory_revision = 0
        self.runtime_directory_hash = ""
        self.runtime_routes = {}
        self._runtime_route_cache = {}
        self.active_schedule_decision = {}
        self._schedule_request_attempts = {}
        self._runtime_schedule_refresh_required = threading.Event()
        # raw_meta_data contains meta configuration of source
        self.raw_meta_data = metadata.copy()
        # meta_data contains data configuration decisions
        self.meta_data = metadata.copy()
        # Existing cross-layer policies may feed materialized-task features
        # into the next scheduling request.
        self.temp_encoded_frame = ''
        self.temp_hash_code = ''

        """distributed devices information"""
        self.runtime_context = RuntimeContext.get_default()
        self.runtime_resolver = RuntimeResolver(self.runtime_context)
        self.runtime_lease_client = RuntimeLeaseClient(
            self.runtime_context,
            requester=http_request,
        )
        self.local_device = self.runtime_context.local_node
        self.cloud_device = self.runtime_context.cloud_node
        self.all_edge_devices = (
            Context.get_parameter('ALL_EDGE_DEVICES', direct=False)
            or self.runtime_context.edge_nodes()
        )

        """network communication base information"""
        self.schedule_address = self.runtime_resolver.resolve_url(
            "scheduler",
            path=NetworkAPIPath.SCHEDULER_SCHEDULE,
            target_node=self.cloud_device or None,
        )

        """hook functions"""
        self.before_schedule_operation = Context.get_algorithm('GEN_BSO')
        self.after_schedule_operation = Context.get_algorithm('GEN_ASO')
        self.data_getter = Context.get_algorithm('GEN_GETTER')
        self.before_submit_task_operation = Context.get_algorithm('GEN_BSTO')
        self.request_scheduling_interval = Context.get_parameter('REQUEST_SCHEDULING_INTERVAL', direct=False)

    def create_task_identity(self):
        return TaskIdentity.create(
            source_id=self.source_id,
            task_id=Counter.get_count('task_id'),
        )

    def schedule_request_context(self):
        """Return host-owned context shared by every scheduling extension."""

        context = {
            'source_id': self.source_id,
            'meta_data': copy.deepcopy(self.raw_meta_data),
            'current_configuration': copy.deepcopy(self.meta_data),
            'source_device': self.local_device,
            'all_edge_devices': copy.deepcopy(self.all_edge_devices),
            'dag': Task.extract_dag_deployment_from_dag(self.task_dag),
            'deployment_version': self.deployment_version,
        }
        revision = int(self.runtime_directory_revision or 0)
        directory_hash = str(self.runtime_directory_hash or '').strip()
        if revision < 1 or not directory_hash:
            return context
        context.update({
            'runtime_directory_revision': revision,
            'runtime_directory_hash': directory_hash,
            'runtime_route_cache_keys': self.runtime_route_cache_keys(),
        })
        return context

    @staticmethod
    def _task_context(task_identity):
        if task_identity is None:
            return None
        if isinstance(task_identity, TaskIdentity):
            return task_identity.to_dict()
        if isinstance(task_identity, dict):
            return copy.deepcopy(task_identity)
        raise TypeError('task identity must be TaskIdentity, dict, or None')

    def _schedule_decision_from_response(self, response, request_params, task_identity):
        task_context = self._task_context(task_identity) or {}
        provided = response.get('schedule_decision')
        provided = provided if isinstance(provided, dict) else {}
        expected = build_schedule_decision(
            request_params,
            response.get('plan'),
            response.get('deployment_version', 0),
            response.get('runtime_directory_revision'),
        )
        if not provided:
            return expected

        root_uuid = str(provided.get('root_uuid') or '')
        if task_context and root_uuid != str(task_context.get('root_uuid') or ''):
            raise ValueError('scheduler decision root_uuid does not match task identity')
        if str(provided.get('plan_digest') or '') != expected['plan_digest']:
            raise ValueError('scheduler decision plan_digest does not match response plan')
        if task_context and str(provided.get('decision_id') or '') != expected['decision_id']:
            raise ValueError('scheduler decision_id does not match task and plan')
        decision = copy.deepcopy(provided)
        decision['decision_id'] = str(decision.get('decision_id') or expected['decision_id'])
        decision['plan_digest'] = expected['plan_digest']
        return decision

    def request_schedule_policy(self, task_identity=None):
        params = self.before_schedule_operation(self)
        if not isinstance(params, dict):
            raise TypeError('before-schedule operation must return a dictionary')
        params = copy.deepcopy(params)
        # The host owns the current scheduling state; hooks only contribute
        # algorithm-specific observations. Apply host fields after the hook so
        # extensions cannot accidentally freeze or fabricate another state.
        for field in _HOST_SCHEDULE_CONTEXT_FIELDS:
            params.pop(field, None)
        params.update(self.schedule_request_context())
        task_context = self._task_context(task_identity)
        if task_context is not None:
            params['task_context'] = task_context
            root_uuid = str(task_context.get('root_uuid') or '')
            attempt = self._schedule_request_attempts.get(root_uuid, 0) + 1
            self._schedule_request_attempts[root_uuid] = attempt
            params['schedule_request_attempt'] = attempt
        response = http_request(url=self.schedule_address,
                                method=NetworkAPIMethod.SCHEDULER_SCHEDULE,
                                timeout=_SCHEDULER_REQUEST_TIMEOUT_SECONDS,
                                data={'data': json.dumps(params)})
        if response is None:
            # A previous route snapshot proves only that the old decision was
            # routable.  It must never stand in for the fresh task-bound
            # decision requested above.
            return False
        if not isinstance(response, dict):
            LOGGER.error('[Scheduling Decision] Scheduler response must be an object.')
            return False
        try:
            schedule_decision = self._schedule_decision_from_response(
                response,
                params,
                task_identity,
            )
        except (TypeError, ValueError) as exc:
            LOGGER.error(f'[Scheduling Decision] Reject invalid scheduler decision: {exc}')
            return False
        if not self._accept_runtime_directory(response):
            return False
        self.after_schedule_operation(self, copy.deepcopy(response))
        self.active_schedule_decision = schedule_decision
        if task_context is not None:
            self._schedule_request_attempts.pop(
                str(task_context.get('root_uuid') or ''), None
            )
        return self.runtime_routes_ready()

    def cancel_schedule_reservation(self, task_identity):
        """Cancel the exact task-bound decision if no source data materialized."""

        task_context = self._task_context(task_identity) or {}
        root_uuid = str(task_context.get("root_uuid") or "")
        decision = self.active_schedule_decision
        if not root_uuid or str(decision.get("root_uuid") or "") != root_uuid:
            # Periodic policies may intentionally reuse an older plan without
            # creating a reservation for this identity.
            return True
        try:
            self.runtime_lease_client.cancel_reservation(
                self.runtime_directory_revision,
                root_uuid,
                decision_id=decision.get("decision_id"),
            )
        except RuntimeLeaseError as exc:
            LOGGER.warning(
                f'[Scheduling Decision] Unable to cancel unmaterialized task '
                f'reservation root={root_uuid}: {exc}'
            )
            return False
        self.active_schedule_decision = {}
        return True

    @staticmethod
    def _extract_runtime_directory(response):
        if not isinstance(response, dict):
            return None, None, None
        directory = response.get("runtime_directory", response.get("runtimeDirectory"))
        directory = directory if isinstance(directory, dict) else {}
        revision = response.get(
            "runtime_directory_revision",
            response.get("runtimeDirectoryRevision", directory.get("revision")),
        )
        directory_hash = response.get(
            "runtime_directory_hash",
            response.get("runtimeDirectoryHash", directory.get("hash")),
        )
        routes = response.get(
            "runtime_routes",
            response.get("runtimeRoutes", directory.get("routes")),
        )
        return revision, directory_hash, routes

    @staticmethod
    def _runtime_routes_cache_key(routes):
        payload = json.dumps(
            routes,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def runtime_route_cache_keys(self):
        return list(self._runtime_route_cache)[-16:]

    def _remember_runtime_routes(self, routes, cache_key):
        self._runtime_route_cache.pop(cache_key, None)
        self._runtime_route_cache[cache_key] = copy.deepcopy(routes)
        while len(self._runtime_route_cache) > 16:
            oldest = next(iter(self._runtime_route_cache))
            self._runtime_route_cache.pop(oldest, None)

    def _accept_runtime_directory(self, response):
        revision, directory_hash, routes = self._extract_runtime_directory(response)
        try:
            revision = int(revision)
        except (TypeError, ValueError):
            LOGGER.error("[Runtime Directory] Scheduler response has no valid directory revision.")
            return False
        if revision < 1:
            LOGGER.error("[Runtime Directory] Directory revision must be positive.")
            return False
        if not isinstance(directory_hash, str) or not directory_hash.strip():
            LOGGER.error("[Runtime Directory] Scheduler response has no valid directory hash.")
            return False
        directory_hash = directory_hash.strip()
        if revision < self.runtime_directory_revision:
            LOGGER.warning(
                f"[Runtime Directory] Ignore stale scheduler response revision {revision}; "
                f"current revision is {self.runtime_directory_revision}."
            )
            return False
        if revision == self.runtime_directory_revision and revision > 0:
            if not self.runtime_directory_hash:
                LOGGER.error(
                    f"[Runtime Directory] Revision {revision} has no previously accepted "
                    "full-directory hash."
                )
                return False
            if directory_hash != self.runtime_directory_hash:
                LOGGER.error(
                    f"[Runtime Directory] Revision {revision} changed full-directory hash; "
                    "reject non-immutable snapshot."
                )
                return False
        route_cache_key = str(
            response.get(
                "runtime_routes_cache_key",
                response.get("runtimeRoutesCacheKey", ""),
            )
            or ""
        ).strip()
        routes_cached = bool(
            response.get(
                "runtime_routes_cached",
                response.get("runtimeRoutesCached", False),
            )
        )
        if routes is None and routes_cached:
            routes = self._runtime_route_cache.get(route_cache_key)
            if routes is None:
                LOGGER.error(
                    "[Runtime Directory] Scheduler referenced an unknown "
                    f"route cache key {route_cache_key!r}."
                )
                return False
        try:
            endpoints = RuntimeResolver.list_routes(routes or {})
        except (TypeError, ValueError) as exc:
            LOGGER.error(f"[Runtime Directory] Invalid scheduler routes: {exc}")
            return False
        if not endpoints:
            LOGGER.error("[Runtime Directory] Scheduler returned an empty exact-route snapshot.")
            return False
        try:
            for endpoint in endpoints:
                if endpoint.component in RuntimeResolver.TASK_ROUTED_COMPONENTS:
                    endpoint.validate_exact()
        except ValueError as exc:
            LOGGER.error(f"[Runtime Directory] Incomplete exact route identity: {exc}")
            return False
        computed_cache_key = self._runtime_routes_cache_key(routes)
        if route_cache_key and route_cache_key != computed_cache_key:
            LOGGER.error(
                "[Runtime Directory] Route cache key does not match exact routes."
            )
            return False
        route_cache_key = route_cache_key or computed_cache_key
        if revision != self.runtime_directory_revision:
            self._runtime_route_cache.clear()
        self._remember_runtime_routes(routes, route_cache_key)
        self.runtime_directory_revision = revision
        self.runtime_directory_hash = directory_hash
        self.runtime_routes = copy.deepcopy(routes)
        return True

    def runtime_routes_ready(self):
        """Check that the accepted snapshot can route the current DAG exactly."""
        if self.runtime_directory_revision < 1 or not self.runtime_routes:
            return False
        try:
            devices = set()
            for node_name, node in self.task_dag.nodes.items():
                if node_name in ("_start", "_end"):
                    continue
                service = node.service
                service_name = service.get_service_name()
                device = service.get_execute_device()
                devices.add(device)
                self.runtime_resolver.resolve(
                    "processor",
                    task=self.runtime_routes,
                    target_node=device,
                    logical_service=service_name,
                    exact=True,
                )
            for device in devices:
                self.runtime_resolver.resolve(
                    "controller",
                    task=self.runtime_routes,
                    target_node=device,
                    exact=True,
                )
        except (LookupError, ValueError) as exc:
            LOGGER.error(f"[Runtime Directory] Current scheduling plan is not exactly routable: {exc}")
            return False
        return bool(devices)

    @staticmethod
    def record_total_start_ts(cur_task: Task):
        TimeEstimator.record_task_ts(cur_task,
                                     'total_start_time',
                                     is_end=False)

    @staticmethod
    def record_transmit_start_ts(cur_task: Task):
        TimeEstimator.record_dag_ts(cur_task,
                                    is_end=False,
                                    sub_tag='transmit')

    def generate_task(
        self,
        task_id,
        task_dag,
        service_deployment,
        meta_data,
        compressed_path,
        hash_codes,
        task_identity=None,
    ):
        """generate a new task"""
        if task_identity is not None and not isinstance(task_identity, TaskIdentity):
            raise TypeError('task_identity must be a TaskIdentity')
        if task_identity is not None:
            if task_identity.source_id != int(self.source_id):
                raise ValueError('task identity source_id does not match generator')
            if task_identity.task_id != int(task_id):
                raise ValueError('task identity task_id does not match generated task')
        decision = self.active_schedule_decision
        return Task(source_id=self.source_id,
                    task_id=task_id,
                    source_device=self.local_device,
                    all_edge_devices=self.all_edge_devices,
                    dag=task_dag,
                    deployment=service_deployment,
                    deployment_version=self.deployment_version,
                    runtime_directory_revision=self.runtime_directory_revision,
                    runtime_routes=self.runtime_routes,
                    metadata=meta_data,
                    raw_metadata=self.raw_meta_data,
                    hash_data=hash_codes,
                    file_path=compressed_path,
                    task_uuid=task_identity.task_uuid if task_identity else '',
                    root_uuid=task_identity.root_uuid if task_identity else '',
                    schedule_decision_id=decision.get('decision_id', ''),
                    schedule_plan_digest=decision.get('plan_digest', ''))

    def _submit_task_to_controller(
        self,
        cur_task,
        *,
        file_path=None,
        file_content=None,
    ):
        assert cur_task, 'Task is empty when submit to controller!'

        self.before_submit_task_operation(self, cur_task)

        # Once source data has been materialized, transient admission failures
        # apply backpressure instead of discarding the task. A retired snapshot
        # is an explicit fence: refresh scheduling before accepting new data.
        while True:
            try:
                self.runtime_lease_client.acquire(cur_task)
                break
            except RuntimeLeaseRetired as exc:
                self._runtime_schedule_refresh_required.set()
                LOGGER.warning(
                    f'[Runtime Lease] Reject task from retired directory: '
                    f'source={cur_task.get_source_id()}, task={cur_task.get_task_id()}, error={exc}'
                )
                return False
            except RuntimeLeaseUnavailable as exc:
                LOGGER.warning(
                    f'[Runtime Lease] Admission unavailable; retain task and retry. '
                    f'source={cur_task.get_source_id()}, task={cur_task.get_task_id()}, error={exc}'
                )
                time.sleep(0.5)

        dst_device = cur_task.get_current_stage_device()
        controller_address = self.runtime_resolver.resolve_url(
            "controller",
            path=NetworkAPIPath.CONTROLLER_TASK,
            task=cur_task,
            target_node=dst_device,
            exact=True,
        )
        self.record_transmit_start_ts(cur_task)
        deliver_task(
            url=controller_address,
            method=NetworkAPIMethod.CONTROLLER_TASK,
            task=cur_task,
            file_path=file_path,
            file_content=file_content,
            persistent=True,
        )
        LOGGER.info(f'[To Controller {dst_device}] source: {cur_task.get_source_id()}  '
                    f'task: {cur_task.get_task_id()}  '
                    f'file: {cur_task.get_file_path()}')
        return True

    def submit_task_to_controller(self, cur_task):
        return self._submit_task_to_controller(
            cur_task,
            file_path=cur_task.get_file_path() if cur_task else None,
        )

    def run(self):
        assert None, 'Base Generator should not be invoked directly!'
