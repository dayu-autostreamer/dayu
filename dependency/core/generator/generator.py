import copy
import json
import threading
import time

from core.lib.common import Context, LOGGER
from core.lib.content import Task
from core.lib.network import deliver_task, http_request, NetworkAPIPath, NetworkAPIMethod
from core.lib.estimation import TimeEstimator
from core.lib.runtime import (
    RuntimeContext,
    RuntimeLeaseClient,
    RuntimeLeaseRetired,
    RuntimeLeaseUnavailable,
    RuntimeResolver,
)


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
        self._runtime_schedule_refresh_required = threading.Event()
        # raw_meta_data contains meta configuration of source
        self.raw_meta_data = metadata.copy()
        # meta_data contains data configuration decisions
        self.meta_data = metadata.copy()

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
        self.task_dag = Task.set_execute_device(self.task_dag, self.local_device)

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

    def request_schedule_policy(self):
        params = self.before_schedule_operation(self)
        response = http_request(url=self.schedule_address,
                                method=NetworkAPIMethod.SCHEDULER_SCHEDULE,
                                data={'data': json.dumps(params)})
        if response is None:
            return self.runtime_routes_ready()
        if not self._accept_runtime_directory(response):
            return False
        self.after_schedule_operation(self, response)
        return self.runtime_routes_ready()

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

    def generate_task(self, task_id, task_dag, service_deployment, meta_data, compressed_path, hash_codes):
        """generate a new task"""
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
                    file_path=compressed_path)

    def submit_task_to_controller(self, cur_task):
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
            file_path=cur_task.get_file_path(),
            persistent=True,
        )
        LOGGER.info(f'[To Controller {dst_device}] source: {cur_task.get_source_id()}  '
                    f'task: {cur_task.get_task_id()}  '
                    f'file: {cur_task.get_file_path()}')
        return True

    def run(self):
        assert None, 'Base Generator should not be invoked directly!'
