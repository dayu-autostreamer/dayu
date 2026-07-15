import json
import os

from core.lib.estimation import TimeEstimator
from core.lib.network import http_request, NetworkAPIPath, NetworkAPIMethod
from core.lib.common import LOGGER, Context, TaskConstant, FileOps
from core.lib.content import Task
from core.lib.runtime import RuntimeContext, RuntimeResolver

from .task_coordinator import TaskCoordinator


class Controller:
    def __init__(self):
        self.runtime_context = RuntimeContext.get_default()
        self.runtime_resolver = RuntimeResolver(self.runtime_context)
        self.task_coordinator = TaskCoordinator(runtime_context=self.runtime_context)

        self.is_display = Context.get_parameter('DISPLAY', direct=False)

        self.local_device = self.runtime_context.local_node
        self.cloud_device = self.runtime_context.cloud_node
        self.distribute_address = self.runtime_resolver.resolve_url(
            "distributor",
            path=NetworkAPIPath.DISTRIBUTOR_DISTRIBUTE,
            target_node=self.cloud_device or None,
        )

    def _get_runtime_resolver(self):
        resolver = getattr(self, "runtime_resolver", None)
        if resolver is None:
            context = getattr(self, "runtime_context", None) or RuntimeContext.get_default()
            resolver = RuntimeResolver(context)
            self.runtime_resolver = resolver
        return resolver

    @staticmethod
    def _routes_from_request(request):
        if not isinstance(request, dict):
            return None
        directory = request.get("runtime_directory", request.get("runtimeDirectory"))
        if isinstance(directory, dict):
            return directory.get("routes", directory.get("runtime_routes", directory.get("runtimeRoutes")))
        return request.get("runtime_routes", request.get("runtimeRoutes"))

    def check_processor_health(self, request=None):
        routes = self._routes_from_request(request)
        processors = self._get_runtime_resolver().list_routes(
            routes or {}, component="processor", target_node=self.local_device
        )
        try:
            processors = [endpoint.validate_exact() for endpoint in processors]
        except ValueError as exc:
            LOGGER.warning(f'[HEALTH CHECK] Invalid exact processor route: {exc}')
            return False
        if not processors:
            LOGGER.warning('[HEALTH CHECK] Exact processor routes are required for health checking.')
            return False
        LOGGER.debug(f'[HEALTH CHECK] health checking services: {[route.logical_service for route in processors]}')
        for endpoint in processors:
            processor_health_address = endpoint.url(NetworkAPIPath.PROCESSOR_HEALTH)
            response = http_request(url=processor_health_address, method=NetworkAPIMethod.PROCESSOR_HEALTH)
            if not response or response.get('status') != 'ok':
                LOGGER.debug(f'[HEALTH CHECK] service {endpoint.logical_service} processor health check failed.')
                return False
            LOGGER.debug(f'[HEALTH CHECK] service {endpoint.logical_service} processor health check succeed.')
        return True

    @staticmethod
    def _normalize_service_filter(value):
        if value is None:
            return None
        if isinstance(value, str):
            return {value}
        if isinstance(value, (list, tuple, set)):
            return {str(item) for item in value}
        return None

    def clear_processor_queues(self, request=None):
        """Clear queued, not-yet-running tasks from processor servers on this node."""
        request = request or {}
        if not isinstance(request, dict):
            request = {}

        service_filter = self._normalize_service_filter(request.get("services"))
        timeout_s = request.get("timeout_s", 5.0)
        try:
            timeout_s = max(0.1, float(timeout_s))
        except (TypeError, ValueError):
            timeout_s = 5.0

        processor_payload = {
            "reason": request.get("reason") or "controller_processor_queue_clear",
            "max_count": request.get("max_count"),
            "dry_run": bool(request.get("dry_run", False)),
        }

        routes = self._routes_from_request(request)
        processors = self._get_runtime_resolver().list_routes(
            routes or {}, component="processor", target_node=self.local_device
        )
        try:
            processors = [endpoint.validate_exact() for endpoint in processors]
        except ValueError as exc:
            return {
                "ok": False,
                "device": self.local_device,
                "error": f"invalid exact processor runtime_routes: {exc}",
                "service_count": 0,
                "matched_count": 0,
                "cleared_count": 0,
                "remaining_count": 0,
                "services": {},
            }
        if service_filter is not None:
            processors = [endpoint for endpoint in processors if endpoint.logical_service in service_filter]
        if not processors:
            return {
                "ok": False,
                "device": self.local_device,
                "error": "exact processor runtime_routes are required",
                "service_count": 0,
                "matched_count": 0,
                "cleared_count": 0,
                "remaining_count": 0,
                "services": {},
            }

        results = {}
        total_cleared = 0
        total_matched = 0
        total_remaining = 0
        for endpoint in processors:
            service = endpoint.logical_service
            processor_address = endpoint.url(NetworkAPIPath.PROCESSOR_CLEAR_QUEUE)
            response = http_request(
                url=processor_address,
                method=NetworkAPIMethod.PROCESSOR_CLEAR_QUEUE,
                timeout=timeout_s,
                data={"data": json.dumps(processor_payload)},
            )
            if not isinstance(response, dict):
                results[service] = {
                    "ok": False,
                    "error": "processor queue clear request failed",
                }
                continue

            results[service] = response
            total_cleared += int(response.get("cleared_count") or 0)
            total_matched += int(response.get("matched_count") or 0)
            total_remaining += int(response.get("remaining_count") or 0)

        LOGGER.warning(
            f"[Processor Queue Clear] device={self.local_device}, "
            f"services={[endpoint.logical_service for endpoint in processors]}, "
            f"matched={total_matched}, cleared={total_cleared}, remaining={total_remaining}"
        )
        return {
            "ok": True,
            "device": self.local_device,
            "service_count": len(processors),
            "matched_count": total_matched,
            "cleared_count": total_cleared,
            "remaining_count": total_remaining,
            "services": results,
        }

    def send_task_to_other_device(self, cur_task: Task, device: str = ''):
        self.record_transmit_ts(cur_task=cur_task, is_end=False)
        controller_address = self._get_runtime_resolver().resolve_url(
            "controller",
            path=NetworkAPIPath.CONTROLLER_TASK,
            task=cur_task,
            target_node=device,
            exact=True,
        )

        http_request(url=controller_address,
                     method=NetworkAPIMethod.CONTROLLER_TASK,
                     data={'data': cur_task.serialize()},
                     files={'file': (cur_task.get_file_path(),
                                     open(FileOps.get_task_file_in_temp(cur_task), 'rb'),
                                     'multipart/form-data')})

        LOGGER.info(f'[To Device {device}] source: {cur_task.get_source_id()}  '
                    f'task: {cur_task.get_task_id()} current service: {cur_task.get_flow_index()}')

    def send_task_to_service(self, cur_task: Task, service: str = ''):
        self.record_execute_ts(cur_task=cur_task, is_end=False)

        try:
            service_address = self._get_runtime_resolver().resolve_url(
                "processor",
                path=NetworkAPIPath.PROCESSOR_PROCESS_LOCAL,
                task=cur_task,
                target_node=self.local_device,
                logical_service=service,
                exact=True,
            )
        except (LookupError, ValueError) as exc:
            LOGGER.error(
                f'[Runtime Route Missing] Refuse to reroute service {service} on {self.local_device}: {exc}'
            )
            return 'error'

        if not os.path.exists(FileOps.get_task_file_in_temp(cur_task)):
            LOGGER.warning(f'[Task File Lost] source: {cur_task.get_source_id()}  '
                           f'task: {cur_task.get_task_id()} '
                           f'file: {FileOps.get_task_file_in_temp(cur_task)}')
            return 'error'

        # Local fast path: only send metadata
        http_request(url=service_address,
                     method=NetworkAPIMethod.PROCESSOR_PROCESS_LOCAL,
                     data={'data': cur_task.serialize()})

        LOGGER.info(f'[To Service {service} Local] source: {cur_task.get_source_id()}  '
                    f'task: {cur_task.get_task_id()} current service: {cur_task.get_flow_index()}')

        return 'execute'

    def send_task_to_distributor(self, cur_task: Task):
        self.record_transmit_ts(cur_task=cur_task, is_end=False)
        task_file_path = FileOps.get_task_file_in_temp(cur_task)
        if self.is_display and not os.path.exists(task_file_path):
            LOGGER.warning(f'[Task File Lost] source: {cur_task.get_source_id()}  '
                           f'task: {cur_task.get_task_id()} '
                           f'file: {task_file_path}')
            return
        file_content = open(task_file_path, 'rb') if self.is_display else b''

        http_request(url=self.distribute_address,
                     method=NetworkAPIMethod.DISTRIBUTOR_DISTRIBUTE,
                     files={'file': (cur_task.get_file_path(), file_content, 'multipart/form-data')},
                     data={'data': cur_task.serialize()})

        LOGGER.info(f'[To Distributor] source: {cur_task.get_source_id()}  task: {cur_task.get_task_id()} '
                    f'current service: {cur_task.get_flow_index()}')

    def submit_task(self, cur_task: Task):
        if not cur_task:
            LOGGER.warning('Current task is None')
            return 'error'

        LOGGER.info(f'[Submit Task] source: {cur_task.get_source_id()}  task: {cur_task.get_task_id()} '
                    f'current service: {cur_task.get_flow_index()} dst device: {cur_task.get_current_stage_device()} '
                    f'current device: {self.local_device}')

        service_name, _ = cur_task.get_current_service_info()
        dst_device = cur_task.get_current_stage_device()

        if service_name == TaskConstant.START.value:
            next_tasks = cur_task.step_to_next_stage()
            actions = [self.submit_task(new_task) for new_task in next_tasks]
            action = 'execute' if 'execute' in actions else 'transmit'
        elif service_name == TaskConstant.END.value:
            self.send_task_to_distributor(cur_task)
            action = 'transmit'
        elif dst_device != self.local_device:
            self.send_task_to_other_device(cur_task, dst_device)
            action = 'transmit'
        else:
            action = self.send_task_to_service(cur_task, service_name)

        return action

    def process_return(self, cur_task):
        """step to next step and submit task"""
        assert cur_task, 'Current task is None'

        LOGGER.info(f'[Process Return] source: {cur_task.get_source_id()}  task: {cur_task.get_task_id()}')

        actions = []
        parallel_joints = cur_task.get_parallel_info_for_merge()
        for parallel_joint in parallel_joints:
            joint_service_name = parallel_joint['joint_service']
            parallel_service_names = parallel_joint['parallel_services']
            required_parallel_task_count = len(parallel_service_names)
            new_task = cur_task.fork_task(joint_service_name)

            # node with parallel nodes should merge before step to next stage
            if required_parallel_task_count > 1:
                parallel_count = self.task_coordinator.store_task_data(new_task, joint_service_name)
                # wait when some duplicated tasks (with parallel nodes) not arrive
                if parallel_count != required_parallel_task_count:
                    actions.append('wait')
                    continue
                # retrieve parallel nodes in redis
                tasks = self.task_coordinator.retrieve_task_data(new_task.get_root_uuid(),
                                                                 joint_service_name,
                                                                 required_parallel_task_count)
                # something wrong causes invalid task retrieving
                if not tasks:
                    LOGGER.warning('Invalid task retrieving from task coordinator!')
                    actions.append('wait')
                    continue

                # merge parallel tasks
                for task in tasks:
                    new_task.merge_task(task)
                LOGGER.debug(f"Merge task with services {[task.get_past_flow_index() for task in tasks]} "
                             f"into task with service '{joint_service_name}'")

            actions.append(self.submit_task(new_task))

        return actions

    @staticmethod
    def record_transmit_ts(cur_task: Task, is_end: bool = False):
        assert cur_task, 'Current task of controller is NOT set!'

        try:
            duration = TimeEstimator.record_dag_ts(cur_task, is_end=is_end, sub_tag='transmit')
        except Exception as e:
            LOGGER.warning(f'[Source {cur_task.get_source_id()} / Task {cur_task.get_task_id()}] '
                           f'transmit time record failed of stage {cur_task.get_flow_index()}: {str(e)}')
            duration = 0

        if is_end:
            cur_task.save_transmit_time(duration)
            LOGGER.info(f'[Source {cur_task.get_source_id()} / Task {cur_task.get_task_id()}] '
                        f'record transmit time of stage {cur_task.get_flow_index()}: {duration:.3f}s')

    @staticmethod
    def record_execute_ts(cur_task: Task, is_end: bool = False):
        assert cur_task, 'Current task of controller is NOT set!'

        try:
            duration = TimeEstimator.record_dag_ts(cur_task, is_end=is_end, sub_tag='execute')
        except Exception as e:
            LOGGER.warning(f'[Source {cur_task.get_source_id()} / Task {cur_task.get_task_id()}] '
                           f'execute time record failed of stage {cur_task.get_flow_index()}: {str(e)}')
            duration = 0

        if is_end:
            cur_task.save_execute_time(duration)
            LOGGER.info(f'[Source {cur_task.get_source_id()} / Task {cur_task.get_task_id()}] '
                        f'record execute time of stage {cur_task.get_flow_index()}: {duration:.3f}s')

    @staticmethod
    def erase_transmit_ts(cur_task: Task):
        assert cur_task, 'Current task of controller is NOT set!'

        try:
            TimeEstimator.erase_dag_ts(cur_task, is_end=False, sub_tag='transmit')
            TimeEstimator.erase_dag_ts(cur_task, is_end=True, sub_tag='transmit')
            LOGGER.info(f'[Source {cur_task.get_source_id()} / Task {cur_task.get_task_id()}] '
                        f'erase transmit time of stage {cur_task.get_flow_index()}')
        except Exception as e:
            LOGGER.warning(f'[Source {cur_task.get_source_id()} / Task {cur_task.get_task_id()}]'
                           f' Erase transmit time failed for stage {cur_task.get_flow_index()}: {str(e)}')

    @staticmethod
    def erase_execute_ts(cur_task: Task):
        assert cur_task, 'Current task of controller is NOT set!'

        try:
            TimeEstimator.erase_dag_ts(cur_task, is_end=False, sub_tag='execute')
            LOGGER.info(f'[Source {cur_task.get_source_id()} / Task {cur_task.get_task_id()}] '
                        f'erase execute time of stage {cur_task.get_flow_index()}')
        except Exception as e:
            LOGGER.warning(f'[Source {cur_task.get_source_id()} / Task {cur_task.get_task_id()}]'
                           f' Erase execute time failed for stage {cur_task.get_flow_index()}: {str(e)}')
