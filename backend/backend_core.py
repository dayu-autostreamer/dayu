import copy
import json
import gzip
import shutil
import tempfile
import threading
import re
from collections import deque

import os
import time
from core.lib.content import Task
from core.lib.common import LOGGER, Context, YamlOps, FileOps, Counter, TaskConstant, \
    ConfigBoundInstanceCache
from core.lib.network import http_request, NetworkAPIPath, NetworkAPIMethod
from core.lib.estimation import Timer

from runtime_orchestrator import RuntimeOrchestrator
from template_helper import TemplateHelper


def _indent_json_block(text, prefix='    '):
    return '\n'.join(f'{prefix}{line}' if line else prefix for line in text.splitlines())


class BackendCore:
    def __init__(self):

        self.template_helper = TemplateHelper(Context.get_default_file_path())

        self.namespace = ''
        self.image_meta = None
        self.schedulers = None
        self.services = None

        self.result_visualization_configs = None
        self.system_visualization_configs = None
        self.customized_source_result_visualization_configs = {}
        self.result_visualization_cache = ConfigBoundInstanceCache(
            factory=lambda vf: Context.get_algorithm(
                'RESULT_VISUALIZER',
                al_name=vf['hook_name'],
                **(dict(eval(vf['hook_params'])) if 'hook_params' in vf else {}),
                variables=vf['variables']
            )
        )
        self.system_visualization_cache = ConfigBoundInstanceCache(
            factory=lambda vf: Context.get_algorithm(
                'SYSTEM_VISUALIZER',
                al_name=vf['hook_name'],
                **(dict(eval(vf['hook_params'])) if 'hook_params' in vf else {}),
                variables=vf['variables']
            )
        )

        self.parse_base_info()

        self.source_configs = []

        self.dags = []

        self.time_ticket = 0

        self.result_url = None
        self.result_file_url = None
        self.resource_url = None
        self.log_fetch_url = None

        self.inner_datasource = self.check_simulation_datasource()
        self.source_open = False
        self.source_label = ''
        self.query_lock = threading.Lock()

        self.task_results = {}

        self.is_get_result = False
        self._redeployment_lock = threading.Lock()
        self._redeployment_stop_event = None
        self._redeployment_thread = None
        self.runtime_orchestrator = RuntimeOrchestrator(self.template_helper, self.namespace)
        redeploy_interval = Context.get_parameter('REDEPLOYMENT_REQUEST_INTERVAL', default=20, direct=False)
        self.processor_redeployment_interval_s = max(0.0, float(redeploy_interval))
        if os.getenv('DAYU_RUNTIME_CONTROL_PLANE', '').lower() == 'true':
            self._recover_runtime_session()
        self.system_log_store_path = 'system_log_store.jsonl'
        self.system_log_lock = threading.Lock()
        self.system_log_retention_records = max(
            0,
            int(Context.get_parameter('SYSTEM_LOG_RETENTION_RECORDS', 0, direct=False))
        )
        self.system_log_compact_interval = max(
            1,
            int(Context.get_parameter('SYSTEM_LOG_COMPACT_INTERVAL', 200, direct=False))
        )
        self.system_log_record_count = self._count_jsonl_records(self.system_log_store_path)

        self.default_visualization_image = 'default_visualization.png'

    def parse_base_info(self):
        try:
            base_info = self.template_helper.load_base_info()
            self.namespace = base_info['namespace']
            self.image_meta = base_info['default-image-meta']
            self.schedulers = base_info['scheduler-policies']
            self.services = base_info['services']
            self.result_visualization_configs = base_info['result-visualizations']
            self.system_visualization_configs = base_info['system-visualizations']
        except KeyError as e:
            LOGGER.warning(f'Parse base info failed: {str(e)}')

    def _recover_runtime_session(self):
        """Resume the compact lifecycle record after a backend Pod restart."""
        try:
            session = self.runtime_orchestrator.recover()
            if session is None:
                return
            if session.phase in {'clearing-directory', 'finalizing-uninstall'}:
                # Drain already completed before either phase is persisted.
                # Recovery resumes the exact remaining uninstall boundary and
                # cannot re-open task admission.
                self.runtime_orchestrator.uninstall()
                return
            if session.phase != 'active':
                LOGGER.warning(
                    f'[Runtime Recovery] Session {session.install_id} requires operator cleanup: '
                    f'phase={session.phase}, error={session.last_error}'
                )
                return
            directory = self.runtime_orchestrator.active_directory()
            if directory is None:
                raise RuntimeError('recovered active session has no RuntimeDirectory')
            self._bind_runtime_urls(directory)
            self._start_redeployment_loop(session.install_id)
        except Exception as exc:
            # Keep the management API available for inspection/uninstall even
            # when an external dependency is unavailable during process start.
            LOGGER.warning(f'[Runtime Recovery] Managed runtime recovery failed: {exc}')
            LOGGER.exception(exc)

    def get_log_file_name(self):
        base_info = self.template_helper.load_base_info()
        load_file_name = base_info['log-file-name']
        if not load_file_name:
            return None
        return load_file_name.split('.')[0]

    def parse_and_apply_templates(self, policy, source_deploy, source_label=''):
        """Install one transactional managed-runtime session."""
        try:
            directory = self.runtime_orchestrator.install(
                policy=policy,
                source_deploy=source_deploy,
                source_label=source_label,
            )
        except Exception as exc:
            LOGGER.warning(f'Managed runtime install failed: {exc}')
            LOGGER.exception(exc)
            return False, str(exc)

        self._bind_runtime_urls(directory)
        self._start_redeployment_loop(directory.install_id)
        return True, 'Install services successfully'

    def parse_and_delete_templates(self):
        """Stop generators, drain exact task leases and remove the session."""
        self._stop_redeployment_loop()
        try:
            self.runtime_orchestrator.uninstall()
        except Exception as exc:
            LOGGER.warning(f'Managed runtime uninstall failed: {exc}')
            LOGGER.exception(exc)
            return False, str(exc)
        self.resource_url = None
        self.result_url = None
        self.result_file_url = None
        self.log_fetch_url = None
        return True, 'Uninstall services successfully'

    def parse_and_redeploy_services(self, policy=None):
        """Publish a processor rollout; unchanged plans are a successful no-op."""
        session = self.runtime_orchestrator.current_session()
        if session is None:
            return False, 'no managed runtime session exists'
        policy = policy or self.find_scheduler_policy_by_id(session.policy_id)
        if policy is None:
            return False, f'scheduler policy {session.policy_id!r} does not exist'
        try:
            changed = self.runtime_orchestrator.redeploy(policy)
        except Exception as exc:
            LOGGER.warning(f'Managed processor rollout failed: {exc}')
            LOGGER.exception(exc)
            return False, str(exc)
        if changed:
            directory = self.runtime_orchestrator.active_directory()
            if directory is not None:
                self._bind_runtime_urls(directory)
        return True, 'Redeployment succeeded' if changed else 'Deployment is unchanged'

    def find_service_by_id(self, service_id):
        for service in self.services:
            if service['id'] == service_id:
                return service
        return None

    @staticmethod
    def service_io_labels(service, field):
        service_id = service.get('id') or service.get('service') or '<unknown>'
        value = service.get(field)
        if not isinstance(value, list):
            return None, f"Service '{service_id}' field '{field}' must be a list of type labels"
        if any(not isinstance(item, str) or not item for item in value):
            return None, f"Service '{service_id}' field '{field}' must contain non-empty string labels"
        return value, None

    @classmethod
    def service_io_compatible(cls, parent_service, child_service):
        parent_outputs, error_msg = cls.service_io_labels(parent_service, 'output')
        if error_msg:
            return False, error_msg
        child_inputs, error_msg = cls.service_io_labels(child_service, 'input')
        if error_msg:
            return False, error_msg
        return bool(set(parent_outputs) & set(child_inputs)), None

    def find_dag_by_id(self, dag_id):
        for dag in self.dags:
            if dag['dag_id'] == dag_id:
                return dag['dag']
        return None

    def find_scheduler_policy_by_id(self, policy_id):
        for policy in self.schedulers:
            if policy['id'] == policy_id:
                return policy
        return None

    def find_datasource_configuration_by_label(self, source_label):
        for source_config in self.source_configs:
            if source_config['source_label'] == source_label:
                return source_config
        return None

    def fill_datasource_config(self, config):
        config['source_label'] = f'source_config_{Counter.get_count("source_label")}'
        source_list = config['source_list']
        for index, source in enumerate(source_list):
            source['id'] = index
            source['url'] = self.fill_datasource_url(source['url'], config['source_type'], config['source_mode'], index)

        config['source_list'] = source_list
        return config

    def fill_datasource_url(self, url, source_type, source_mode, source_id):
        if not self.inner_datasource:
            return url
        source_protocol = source_mode.split('_')[0]
        datasource_fqdn = f'datasource-edge.{self.namespace}.svc.cluster.local'
        return f'{source_protocol}://{datasource_fqdn}:8000/{source_type}{source_id}'

    def check_node_exist(self, node):
        record = self.runtime_orchestrator.node_inventory().get(str(node))
        return bool(record and record.get('ready'))

    def get_edge_nodes(self):
        def sort_key(item):
            name = item['name']
            patterns = [
                (r'^edge(\d+)$', 0),
                (r'^edgexn(\d+)$', 1),
                (r'^edgex(\d+)$', 2),
                (r'^edgen(\d+)$', 3),
            ]
            for pattern, group in patterns:
                match = re.match(pattern, name)
                if match:
                    num = int(match.group(1))
                    return group, num
            return len(patterns), 0

        inventory = self.runtime_orchestrator.node_inventory()
        edge_nodes = [
            {'name': node_name}
            for node_name, record in inventory.items()
            if record.get('role') == 'edge' and record.get('ready')
        ]
        edge_nodes.sort(key=sort_key)
        return edge_nodes

    def check_install_state(self):
        # Any CAS session owns RuntimeServices and must be uninstalled before a
        # new transaction. Failed or recovering sessions are therefore still
        # "installed" from the management UI's lifecycle perspective.
        return self.runtime_orchestrator.current_session() is not None

    def check_pods_running_state(self):
        return self.runtime_orchestrator.active_directory() is not None

    def check_simulation_datasource(self):
        return bool(self.template_helper.load_base_info().get('datasource', {}).get('use-simulation'))

    def check_dag(self, dag):

        def topo_sort(graph):
            for node, node_info in graph.items():
                if node == TaskConstant.START.value:
                    continue
                service = self.find_service_by_id(node_info['id'])
                if not service:
                    error_msg = f"Missing service definition for node {node}"
                    LOGGER.error(f"DAG Validation Error: {error_msg}")
                    return False, error_msg
                for field in ('input', 'output'):
                    _, error_msg = self.service_io_labels(service, field)
                    if error_msg:
                        LOGGER.error(f"DAG Validation Error: {error_msg}")
                        return False, error_msg

            in_degree = {}
            for node in graph.keys():
                if node != TaskConstant.START.value:
                    in_degree[node] = len(graph[node]['prev'])
            queue = copy.deepcopy(graph[TaskConstant.START.value])
            topo_order = []

            while queue:
                parent = queue.pop(0)
                topo_order.append(parent)
                for child in graph[parent]['succ']:
                    parent_service = self.find_service_by_id(parent)
                    child_service = self.find_service_by_id(child)
                    if not parent_service or not child_service:
                        error_msg = f"Missing service definition for node {parent if not parent_service else child}"
                        LOGGER.error(f"DAG Validation Error: {error_msg}")
                        return False, error_msg
                    is_compatible, error_msg = self.service_io_compatible(parent_service, child_service)
                    if error_msg:
                        LOGGER.error(f"DAG Validation Error: {error_msg}")
                        return False, error_msg
                    if not is_compatible:
                        error_msg = (
                            f"Node connection mismatch, '{parent}' output '{parent_service['output']}', '{child}' input '{child_service['input']}' "
                        )
                        LOGGER.error(f"DAG Validation Error: {error_msg}")
                        return False, error_msg

                    in_degree[child] -= 1
                    if in_degree[child] == 0:
                        queue.append(child)

            if len(topo_order) != len(in_degree):
                error_msg = "DAG contains cycles or unreachable nodes"
                LOGGER.warning(f"DAG Validation Error: {error_msg}")
                return False, error_msg

            return True, "DAG validation passed"

        return topo_sort(dag.copy())

    def get_source_ids(self):
        source_ids = []
        source_config = self.find_datasource_configuration_by_label(self.source_label)
        if not source_config:
            return []
        for source in source_config['source_list']:
            source_ids.append(source['id'])

        return source_ids

    def prepare_result_visualization_data(self, task, is_last=False):
        source_id = task.get_source_id()
        viz_configs = self.customized_source_result_visualization_configs[source_id] \
            if source_id in self.customized_source_result_visualization_configs else self.result_visualization_configs
        viz_functions = self.result_visualization_cache.sync_and_get(viz_configs, namespace='result_visualizer')

        resource_snapshot = None
        if any(config.get('hook_name') == 'service_queue_length' for config in viz_configs):
            self.get_resource_url()
            if self.resource_url:
                resource_snapshot = http_request(
                    self.resource_url,
                    method=NetworkAPIMethod.SCHEDULER_GET_RESOURCE,
                )

        visualization_data = []
        for idx, (viz_config, viz_func) in enumerate(zip(viz_configs, viz_functions)):
            try:
                if 'save_expense' in viz_config and viz_config['save_expense'] and not is_last:
                    visualization_data.append({"id": idx, "data": {v: None for v in viz_config['variables']}})
                else:
                    if viz_config.get('hook_name') == 'service_queue_length':
                        data = viz_func(task, resource=resource_snapshot)
                    else:
                        data = viz_func(task)
                    visualization_data.append({"id": idx, "data": data})
            except Exception as e:
                LOGGER.warning(f'Failed to load result visualization data: {str(e)}')
                LOGGER.exception(e)

        return visualization_data

    def prepare_system_visualizations_data(self):
        viz_configs = self.system_visualization_configs
        viz_functions = self.system_visualization_cache.sync_and_get(viz_configs, namespace='system_visualizer')

        resource_snapshot = None
        scheduling_overhead = None
        try:
            # Fetch each scheduler snapshot once; visualizers are pure transforms.
            self.get_resource_url()
            if self.resource_url:
                resource_snapshot = http_request(self.resource_url, method=NetworkAPIMethod.SCHEDULER_GET_RESOURCE)
                scheduler_base = self.resource_url.rsplit(NetworkAPIPath.SCHEDULER_GET_RESOURCE, 1)[0]
                scheduling_overhead = http_request(
                    f'{scheduler_base}{NetworkAPIPath.SCHEDULER_OVERHEAD}',
                    method=NetworkAPIMethod.SCHEDULER_OVERHEAD,
                )
        except Exception as e:
            LOGGER.warning(f'Failed to fetch scheduler resource for system viz: {str(e)}')
            LOGGER.exception(e)

        visualization_data = []
        for idx, (viz_config, viz_func) in enumerate(zip(viz_configs, viz_functions)):
            try:
                hook_name = viz_config.get('hook_name')
                if hook_name in {'cpu_usage', 'memory_usage'}:
                    data = viz_func(resource=resource_snapshot)
                elif hook_name == 'schedule_overhead':
                    data = viz_func(scheduling_overhead=scheduling_overhead)
                else:
                    data = viz_func()
                visualization_data.append({"id": idx, "data": data})
            except Exception as e:
                LOGGER.warning(f'Failed to load result visualization data: {str(e)}')
                LOGGER.exception(e)

        return visualization_data

    def parse_task_result(self, results):
        for result in results:
            if result is None or result == '':
                continue

            task = Task.deserialize(result)

            source_id = task.get_source_id()
            LOGGER.debug(task.get_delay_info())

            if not self.source_open:
                break

            self.task_results[source_id].put(copy.deepcopy(task))

    def fetch_visualization_data(self, source_id):
        assert source_id in self.task_results, f'Source_id {source_id} not found in task results!'
        tasks = self.task_results[source_id].get_all()
        vis_results = []

        with Timer(f'Visualization preparation for {len(tasks)} tasks'):
            for idx, task in enumerate(tasks):
                file_path = self.get_file_result(task.get_file_path())
                try:
                    visualization_data = self.prepare_result_visualization_data(task, idx == len(tasks) - 1)
                except Exception as e:
                    LOGGER.warning(f'Prepare visualization data failed: {str(e)}')
                    LOGGER.exception(e)
                    continue

                FileOps.remove_file(file_path)

                vis_results.append({
                    'task_id': task.get_task_id(),
                    'data': visualization_data,
                })

        return vis_results

    def run_get_result(self):
        time_ticket = 0
        while self.is_get_result:
            try:
                time.sleep(1)
                self.get_result_url()
                if not self.result_url:
                    LOGGER.debug('[NO RESULT] Fetch result url failed.')
                    continue
                response = http_request(self.result_url,
                                        method=NetworkAPIMethod.DISTRIBUTOR_RESULT,
                                        json={'time_ticket': time_ticket, 'size': 0})

                if not response:
                    self.result_url = None
                    self.result_file_url = None
                    LOGGER.debug('[NO RESULT] Request result url failed.')
                    continue

                time_ticket = response["time_ticket"]
                results = response['result']
                LOGGER.debug(f'Fetch {len(results)} tasks from time ticket: {time_ticket}')
                self.parse_task_result(results)

            except Exception as e:
                LOGGER.warning(f'Unexpected error occurred in getting task result: {str(e)}')
                LOGGER.exception(e)

    def _start_redeployment_loop(self, install_id):
        """Replace the rollout worker with one scoped to this installation."""
        install_id = str(install_id or '').strip()
        if not install_id:
            raise ValueError('redeployment loop requires an install_id')
        with self._redeployment_lock:
            if self._redeployment_stop_event is not None:
                self._redeployment_stop_event.set()
            if max(0.0, float(self.processor_redeployment_interval_s)) <= 0.0:
                self._redeployment_stop_event = None
                self._redeployment_thread = None
                LOGGER.info('[Redeployment] Automatic processor rollout is disabled.')
                return
            stop_event = threading.Event()
            thread = threading.Thread(
                target=self.run_cycle_deploy,
                args=(stop_event, install_id),
                name=f'dayu-runtime-redeployment-{install_id}',
                daemon=True,
            )
            self._redeployment_stop_event = stop_event
            self._redeployment_thread = thread
            try:
                thread.start()
            except Exception:
                stop_event.set()
                self._redeployment_stop_event = None
                self._redeployment_thread = None
                raise

    def _stop_redeployment_loop(self):
        """Invalidate the current rollout worker before lifecycle mutation."""
        with self._redeployment_lock:
            if self._redeployment_stop_event is not None:
                self._redeployment_stop_event.set()
            self._redeployment_stop_event = None
            self._redeployment_thread = None

    def _wait_until_next_redeployment_cycle(self, stop_event, cycle_started_t):
        interval_s = max(0.0, float(self.processor_redeployment_interval_s))
        if interval_s <= 0.0:
            return stop_event.is_set()

        elapsed_s = max(0.0, time.monotonic() - cycle_started_t)
        sleep_s = max(0.0, interval_s - elapsed_s)
        return stop_event.wait(sleep_s) if sleep_s > 0.0 else stop_event.is_set()

    def run_cycle_deploy(self, stop_event, install_id):
        interval = max(0.0, float(self.processor_redeployment_interval_s))
        if interval <= 0:
            LOGGER.info('[Redeployment] Automatic processor rollout is disabled.')
            return
        try:
            while not stop_event.is_set():
                cycle_started_t = time.monotonic()
                try:
                    # Serialize the final token check with stop/replacement. Once
                    # uninstall returns from _stop_redeployment_loop, an old
                    # worker can neither start nor repeat a redeployment call.
                    with self._redeployment_lock:
                        if self._redeployment_stop_event is not stop_event or stop_event.is_set():
                            return
                        session = self.runtime_orchestrator.current_session()
                        if (
                            session is None
                            or session.phase != 'active'
                            or session.install_id != install_id
                        ):
                            LOGGER.debug(
                                '[Redeployment] Managed runtime session changed; stop rollout loop.'
                            )
                            return
                        policy = self.find_scheduler_policy_by_id(session.policy_id)
                        result, message = self.parse_and_redeploy_services(policy)
                    if not result:
                        LOGGER.warning(f'[Redeployment] {message}')
                except Exception as exc:
                    LOGGER.warning(f'[Redeployment] Unexpected rollout error: {exc}')
                    LOGGER.exception(exc)
                if self._wait_until_next_redeployment_cycle(stop_event, cycle_started_t):
                    return
        finally:
            with self._redeployment_lock:
                if self._redeployment_stop_event is stop_event:
                    self._redeployment_stop_event = None
                    self._redeployment_thread = None

    @staticmethod
    def _count_jsonl_records(file_path):
        if not os.path.exists(file_path):
            return 0
        with open(file_path, 'r', encoding='utf-8') as f:
            return sum(1 for line in f if line.strip())

    def _append_system_log_snapshot(self, snapshot):
        with open(self.system_log_store_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(snapshot, ensure_ascii=False))
            f.write('\n')

    def _maybe_compact_system_log_store_locked(self):
        if not self.system_log_retention_records:
            return
        if self.system_log_record_count <= self.system_log_retention_records + self.system_log_compact_interval:
            return

        recent_lines = deque(maxlen=self.system_log_retention_records)
        try:
            with open(self.system_log_store_path, 'r', encoding='utf-8') as src:
                for line in src:
                    line = line.strip()
                    if line:
                        recent_lines.append(line)

            temp_handle = tempfile.NamedTemporaryFile(
                prefix='dayu-system-log-compact-',
                suffix='.jsonl',
                delete=False
            )
            temp_path = temp_handle.name
            temp_handle.close()

            try:
                with open(temp_path, 'w', encoding='utf-8') as dst:
                    for line in recent_lines:
                        dst.write(line)
                        dst.write('\n')
                os.replace(temp_path, self.system_log_store_path)
            except Exception:
                FileOps.remove_file(temp_path)
                raise

            self.system_log_record_count = len(recent_lines)
            LOGGER.info(f'[Backend] Compacted system log store to {self.system_log_record_count} records.')
        except Exception as e:
            LOGGER.warning(f'Compact system log store failed: {str(e)}')
            LOGGER.exception(e)

    def _create_system_log_snapshot_file(self):
        snapshot_handle = tempfile.NamedTemporaryFile(
            prefix='dayu-system-log-snapshot-',
            suffix='.jsonl',
            delete=False
        )
        snapshot_path = snapshot_handle.name
        snapshot_handle.close()

        with self.system_log_lock:
            if os.path.exists(self.system_log_store_path):
                shutil.copyfile(self.system_log_store_path, snapshot_path)
            else:
                with open(snapshot_path, 'w', encoding='utf-8'):
                    pass

        return snapshot_path

    def create_system_log_export_file(self):
        snapshot_path = self._create_system_log_snapshot_file()
        export_handle = tempfile.NamedTemporaryFile(
            prefix='dayu-system-log-',
            suffix='.json.gz',
            delete=False
        )
        export_path = export_handle.name
        export_handle.close()

        try:
            with gzip.open(export_path, 'wt', encoding='utf-8') as fh:
                fh.write('[\n')
                first = True
                with open(snapshot_path, 'r', encoding='utf-8') as src:
                    for line in src:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError:
                            LOGGER.warning('[Backend] Skip malformed system log record during export.')
                            continue

                        if not first:
                            fh.write(',\n')
                        fh.write(_indent_json_block(json.dumps(record, ensure_ascii=False, indent=4)))
                        first = False

                if not first:
                    fh.write('\n')
                fh.write(']\n')
        except Exception:
            FileOps.remove_file(export_path)
            raise
        finally:
            FileOps.remove_file(snapshot_path)

        return export_path

    def get_system_parameters(self):
        # Skip system parameters retrieving when not installed
        if not self.check_install_state():
            return []

        # Backend-controlled timestamp and single resource fetch per request
        timestamp = time.strftime('%H:%M:%S', time.localtime())

        data = self.prepare_system_visualizations_data()
        snapshot = {"timestamp": timestamp, "data": data}

        try:
            with self.system_log_lock:
                self._append_system_log_snapshot(snapshot)
                self.system_log_record_count += 1
                self._maybe_compact_system_log_store_locked()
        except Exception as e:
            LOGGER.warning(f'Append system log failed: {str(e)}')
            LOGGER.exception(e)

        return [snapshot]

    def check_datasource_config(self, config_path):
        if not YamlOps.is_yaml_file(config_path):
            return None

        config = YamlOps.read_yaml(config_path)
        try:
            _ = config['source_name']
            _ = config['source_type']
            _ = config['source_mode']
            for camera in config['source_list']:
                _ = camera['name']
                if self.inner_datasource:
                    _ = camera['dir']
                else:
                    _ = camera['url']
                _ = camera['metadata']

        except Exception as e:
            LOGGER.warning(f'Datasource config file format error: {str(e)}')
            LOGGER.exception(e)
            return None

        return config

    def check_visualization_config(self, config_path):
        if not YamlOps.is_yaml_file(config_path):
            return None

        config = YamlOps.read_yaml(config_path)

        try:
            for visualization in config:
                viz_name = visualization['name']
                assert isinstance(viz_name, str), '"name" is not a string'
                viz_type = visualization['type']
                assert isinstance(viz_type, str), '"type" is not a string'
                viz_var = visualization['variables']
                assert isinstance(viz_var, list), '"variables" is not a list'
                viz_size = visualization['size']
                assert isinstance(viz_size, int), '"size" is not an integer'
                if 'hook_name' in visualization:
                    assert isinstance(visualization['hook_name'], str), '"hook_name" is not a string'
                if 'hook_params' in visualization:
                    assert isinstance(visualization['hook_params'], str), '"hook_params" is not a string(dict)'
                    assert isinstance(eval(visualization['hook_params']), dict), '"hook_params" is not a string(dict)'
                if 'x_axis' in visualization:
                    assert isinstance(visualization['x_axis'], str), '"x_axis" is not a string'
                if 'y_axis' in visualization:
                    assert isinstance(visualization['y_axis'], str), '"y_axis" is not a string'
            return config
        except Exception as e:
            LOGGER.warning(f'Visualization config file format error: {str(e)}')
            LOGGER.exception(e)
            return None

    @staticmethod
    def _runtime_unit(directory, component):
        matches = [unit for unit in directory.routes if unit.slot.component == component]
        if len(matches) != 1 or matches[0].endpoint is None:
            raise RuntimeError(f'RuntimeDirectory requires exactly one endpoint for {component!r}')
        return matches[0]

    def _bind_runtime_urls(self, directory):
        scheduler = self._runtime_unit(directory, 'scheduler').endpoint
        distributor = self._runtime_unit(directory, 'distributor').endpoint
        scheduler_base = f'http://{scheduler.dns_name}:{scheduler.port}'
        distributor_base = f'http://{distributor.dns_name}:{distributor.port}'
        self.resource_url = f'{scheduler_base}{NetworkAPIPath.SCHEDULER_GET_RESOURCE}'
        self.result_url = f'{distributor_base}{NetworkAPIPath.DISTRIBUTOR_RESULT}'
        self.result_file_url = f'{distributor_base}{NetworkAPIPath.DISTRIBUTOR_FILE}'
        self.log_fetch_url = f'{distributor_base}{NetworkAPIPath.DISTRIBUTOR_EXPORT_RESULT_LOG}'

    def _refresh_runtime_urls(self):
        directory = self.runtime_orchestrator.active_directory()
        if directory is None:
            return False
        self._bind_runtime_urls(directory)
        return True

    def get_resource_url(self):
        if not self.resource_url:
            self._refresh_runtime_urls()

    def get_result_url(self):
        if not self.result_url or not self.result_file_url:
            self._refresh_runtime_urls()

    def get_log_url(self):
        if not self.log_fetch_url:
            self._refresh_runtime_urls()

    def get_file_result(self, file_name):
        if not self.result_file_url:
            return ''
        response = http_request(self.result_file_url,
                                method=NetworkAPIMethod.DISTRIBUTOR_FILE,
                                no_decode=True,
                                json={'file': file_name},
                                stream=True)
        if response is None:
            self.result_file_url = None
            return ''
        with open(file_name, 'wb') as file_out:
            for chunk in response.iter_content(chunk_size=8192):
                file_out.write(chunk)
        return file_name

    def open_result_log_export_stream(self):
        self.parse_base_info()
        self.get_log_url()
        if not self.log_fetch_url:
            return None

        response = http_request(
            self.log_fetch_url,
            method=NetworkAPIMethod.DISTRIBUTOR_EXPORT_RESULT_LOG,
            no_decode=True,
            stream=True
        )
        if response is None:
            self.log_fetch_url = None
            return None
        return response

    def get_result_visualization_config(self, source_id):
        self.parse_base_info()
        visualizations = self.customized_source_result_visualization_configs[source_id] \
            if source_id in self.customized_source_result_visualization_configs else self.result_visualization_configs
        return [{'id': idx, **vf} for idx, vf in enumerate(visualizations)]

    def get_system_visualization_config(self):
        self.parse_base_info()
        return [{'id': idx, **vf} for idx, vf in enumerate(self.system_visualization_configs)]
