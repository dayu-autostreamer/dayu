import copy
import json
import math
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Body, BackgroundTasks, HTTPException, Request
from fastapi.routing import APIRoute
from starlette.responses import JSONResponse, FileResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool

from core.lib.common import LOGGER, Counter, FileOps
from core.lib.network import NetworkAPIMethod, NetworkAPIPath

from backend_core import BackendCore


_RESOURCE_FIELDS = {
    "cpu": ("usage_millicores", "reference_millicores"),
    "memory": ("usage_bytes", "reference_bytes"),
}
_UNINSTALL_STALL_WARNING_SECONDS = 180
_CLEANUP_BLOCKER_DETAIL_LIMIT = 25


def _seconds_since(timestamp):
    try:
        value = datetime.fromisoformat(str(timestamp).replace('Z', '+00:00'))
    except (TypeError, ValueError):
        return 0
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return max(0, int((datetime.now(timezone.utc) - value).total_seconds()))


def _cleanup_diagnostics(session, phase):
    if session is None:
        return None
    progress = getattr(session, 'uninstall', None)
    if progress is None:
        if phase not in {'uninstalling', 'finalizing-uninstall'}:
            return None
        return {
            'status': 'progressing',
            'started_at': '',
            'last_progress_at': '',
            'seconds_without_progress': 0,
            'warning_after_seconds': _UNINSTALL_STALL_WARNING_SECONDS,
            'remaining_count': 0,
            'remaining_by_kind': {},
            'affected_nodes': [],
            'blocking_objects': [],
            'truncated_count': 0,
        }

    remaining = tuple(progress.remaining)
    remaining_by_kind = {}
    for resource in remaining:
        remaining_by_kind[resource.kind] = (
            remaining_by_kind.get(resource.kind, 0) + 1
        )
    seconds_without_progress = _seconds_since(progress.last_progress_at)
    return {
        'status': (
            'delayed'
            if seconds_without_progress >= _UNINSTALL_STALL_WARNING_SECONDS
            else 'progressing'
        ),
        'started_at': progress.started_at,
        'last_progress_at': progress.last_progress_at,
        'seconds_without_progress': seconds_without_progress,
        'warning_after_seconds': _UNINSTALL_STALL_WARNING_SECONDS,
        'remaining_count': len(remaining),
        'remaining_by_kind': {
            kind: remaining_by_kind[kind]
            for kind in sorted(remaining_by_kind)
        },
        'affected_nodes': sorted({
            resource.node for resource in remaining if resource.node
        }),
        'blocking_objects': [
            resource.to_dict()
            for resource in remaining[:_CLEANUP_BLOCKER_DETAIL_LIMIT]
        ],
        'truncated_count': max(
            0, len(remaining) - _CLEANUP_BLOCKER_DETAIL_LIMIT,
        ),
    }


def _json_object(data):
    if isinstance(data, dict):
        return data
    if isinstance(data, bytes):
        data = str(data, encoding='utf-8')
    if isinstance(data, str):
        value = json.loads(data) if data else {}
        if isinstance(value, dict):
            return value
    if data is None:
        return {}
    raise ValueError('request body must be a JSON object')


def _optional_non_negative_number(value):
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    value = float(value)
    return value if math.isfinite(value) and value >= 0 else None


def _resource_detail(summary, resource, has_sample):
    usage_key, reference_key = _RESOURCE_FIELDS[resource]
    summary = summary if isinstance(summary, dict) else {}
    status = str(summary.get("status") or "")
    if status not in {"available", "stale", "collecting", "unavailable"}:
        status = "unavailable" if has_sample else "collecting"
    usage = _optional_non_negative_number(summary.get(usage_key))
    reference = _optional_non_negative_number(summary.get(reference_key))
    if reference is not None and reference <= 0:
        reference = None
    if status in {"available", "stale"} and usage is None:
        status = "unavailable"
    utilization = (
        usage * 100 / reference
        if status in {"available", "stale"}
        and usage is not None
        and reference is not None
        else None
    )
    basis = str(summary.get("basis") or "")
    if basis not in {"node_allocatable", "node_capacity"}:
        basis = ""
    return {
        "status": status,
        usage_key: usage,
        reference_key: reference,
        "utilization_percent": utilization,
        "basis": basis,
    }


def _shared_bandwidth(resource_data, has_sample, stale=False):
    """Project the singleton edge-to-cloud iperf probe as a shared view."""

    candidates = []
    if isinstance(resource_data, dict):
        for node, resources in sorted(resource_data.items(), key=lambda item: str(item[0])):
            if not isinstance(resources, dict):
                continue
            value = _optional_non_negative_number(resources.get("available_bandwidth"))
            # AvailableBandwidthMonitor uses both -1 (not the lock holder) and
            # 0 (iperf failure) as non-measurements.
            if value is not None and value > 0:
                candidates.append((str(node), value))
    if len(candidates) == 1:
        node, value = candidates[0]
        return {
            "status": "stale" if stale else "available",
            "mbps": value,
            "probe_node": node,
        }
    if len(candidates) > 1:
        # Multiple positive probes violate the Scheduler lock invariant. Do
        # not pick a dict-order-dependent value and mislabel it as canonical.
        return {
            "status": "ambiguous",
            "mbps": None,
            "probe_node": "",
        }
    return {
        "status": "unavailable" if has_sample else "collecting",
        "mbps": None,
        "probe_node": "",
    }


class BackendServer:
    def __init__(self):

        self.server = BackendCore()

        self.app = FastAPI(routes=[
            APIRoute(NetworkAPIPath.BACKEND_GET_POLICY,
                     self.get_all_schedule_policies,
                     response_class=JSONResponse,
                     methods=[NetworkAPIMethod.BACKEND_GET_POLICY]
                     ),
            APIRoute(NetworkAPIPath.BACKEND_INSTALL_SERVICE,
                     self.install_service,
                     response_class=JSONResponse,
                     methods=[NetworkAPIMethod.BACKEND_INSTALL_SERVICE]
                     ),
            APIRoute(NetworkAPIPath.BACKEND_GET_INSTALLED,
                     self.get_installed_services,
                     response_class=JSONResponse,
                     methods=[NetworkAPIMethod.BACKEND_GET_INSTALLED]
                     ),
            APIRoute(NetworkAPIPath.BACKEND_GET_DAG,
                     self.get_dag_workflows,
                     response_class=JSONResponse,
                     methods=[NetworkAPIMethod.BACKEND_GET_DAG]
                     ),
            APIRoute(NetworkAPIPath.BACKEND_GET_ALL_SERVICES,
                     self.get_all_services,
                     response_class=JSONResponse,
                     methods=[NetworkAPIMethod.BACKEND_GET_ALL_SERVICES]
                     ),
            APIRoute(NetworkAPIPath.BACKEND_POST_DAG,
                     self.update_dag_workflows,
                     response_class=JSONResponse,
                     methods=[NetworkAPIMethod.BACKEND_POST_DAG]
                     ),
            APIRoute(NetworkAPIPath.BACKEND_DELETE_DAG,
                     self.delete_dag_workflow,
                     response_class=JSONResponse,
                     methods=[NetworkAPIMethod.BACKEND_DELETE_DAG]
                     ),
            APIRoute(NetworkAPIPath.BACKEND_POST_DATASOURCE,
                     self.upload_datasource_config_file,
                     response_class=JSONResponse,
                     methods=[NetworkAPIMethod.BACKEND_POST_DATASOURCE]
                     ),
            APIRoute(NetworkAPIPath.BACKEND_GET_SERVICE_INFO,
                     self.get_service_info,
                     response_class=JSONResponse,
                     methods=[NetworkAPIMethod.BACKEND_GET_SERVICE_INFO]
                     ),
            APIRoute(NetworkAPIPath.BACKEND_GET_DATASOURCE,
                     self.get_datasource_info,
                     response_class=JSONResponse,
                     methods=[NetworkAPIMethod.BACKEND_GET_DATASOURCE]
                     ),
            APIRoute(NetworkAPIPath.BACKEND_DELETE_DATASOURCE,
                     self.delete_datasource_info,
                     response_class=JSONResponse,
                     methods=[NetworkAPIMethod.BACKEND_DELETE_DATASOURCE]),
            APIRoute(NetworkAPIPath.BACKEND_SUBMIT_QUERY,
                     self.submit_query,
                     response_class=JSONResponse,
                     methods=[NetworkAPIMethod.BACKEND_SUBMIT_QUERY]
                     ),
            APIRoute(NetworkAPIPath.BACKEND_UNINSTALL_SERVICE,
                     self.uninstall_service,
                     response_class=JSONResponse,
                     methods=[NetworkAPIMethod.BACKEND_UNINSTALL_SERVICE]
                     ),
            APIRoute(NetworkAPIPath.BACKEND_STOP_QUERY,
                     self.stop_query,
                     response_class=JSONResponse,
                     methods=[NetworkAPIMethod.BACKEND_STOP_QUERY]
                     ),
            APIRoute(NetworkAPIPath.BACKEND_INSTALL_STATE,
                     self.get_install_state,
                     response_class=JSONResponse,
                     methods=[NetworkAPIMethod.BACKEND_INSTALL_STATE]
                     ),
            APIRoute(NetworkAPIPath.BACKEND_QUERY_STATE,
                     self.get_query_state,
                     response_class=JSONResponse,
                     methods=[NetworkAPIMethod.BACKEND_QUERY_STATE]
                     ),
            APIRoute(NetworkAPIPath.BACKEND_SOURCE_LIST,
                     self.get_source_list,
                     response_class=JSONResponse,
                     methods=[NetworkAPIMethod.BACKEND_SOURCE_LIST]
                     ),
            APIRoute(NetworkAPIPath.BACKEND_TASK_RESULT,
                     self.get_task_result,
                     response_class=JSONResponse,
                     methods=[NetworkAPIMethod.BACKEND_TASK_RESULT]
                     ),
            APIRoute(NetworkAPIPath.BACKEND_SYSTEM_PARAMETERS,
                     self.get_system_parameters,
                     response_class=JSONResponse,
                     methods=[NetworkAPIMethod.BACKEND_SYSTEM_PARAMETERS]
                     ),
            APIRoute(NetworkAPIPath.BACKEND_POST_RESULT_VISUALIZATION_CONFIG,
                     self.upload_result_visualization_config,
                     response_class=JSONResponse,
                     methods=[NetworkAPIMethod.BACKEND_POST_RESULT_VISUALIZATION_CONFIG]
                     ),
            APIRoute(NetworkAPIPath.BACKEND_GET_RESULT_VISUALIZATION_CONFIG,
                     self.get_result_visualization_config,
                     response_class=JSONResponse,
                     methods=[NetworkAPIMethod.BACKEND_GET_RESULT_VISUALIZATION_CONFIG]
                     ),
            APIRoute(NetworkAPIPath.BACKEND_GET_SYSTEM_VISUALIZATION_CONFIG,
                     self.get_system_visualization_config,
                     response_class=JSONResponse,
                     methods=[NetworkAPIMethod.BACKEND_GET_SYSTEM_VISUALIZATION_CONFIG]
                     ),
            APIRoute(NetworkAPIPath.BACKEND_DOWNLOAD_LOG,
                     self.download_log,
                     response_class=StreamingResponse,
                     methods=[NetworkAPIMethod.BACKEND_DOWNLOAD_LOG]
                     ),
            APIRoute(NetworkAPIPath.BACKEND_DOWNLOAD_SYSTEM_LOG,
                     self.download_system_log,
                     response_class=FileResponse,
                     methods=[NetworkAPIMethod.BACKEND_DOWNLOAD_SYSTEM_LOG]
                     ),
            APIRoute(NetworkAPIPath.BACKEND_EDGE_NODE,
                     self.get_edge_nodes,
                     response_class=JSONResponse,
                     methods=[NetworkAPIMethod.BACKEND_EDGE_NODE]
                     ),
            APIRoute(NetworkAPIPath.BACKEND_DATASOURCE_STATE,
                     self.get_datasource_state,
                     response_class=JSONResponse,
                     methods=[NetworkAPIMethod.BACKEND_DATASOURCE_STATE]
                     ),
            APIRoute(NetworkAPIPath.BACKEND_RESET_DATASOURCE,
                     self.reset_datasource,
                     response_class=JSONResponse,
                     methods=[NetworkAPIMethod.BACKEND_RESET_DATASOURCE]
                     ),
        ], log_level='trace', timeout=6000)

        if hasattr(self.server, 'start'):
            self.app.add_event_handler('startup', self.server.start)
        if hasattr(self.server, 'close'):
            self.app.add_event_handler('shutdown', self.server.close)

    async def get_all_schedule_policies(self):
        """
        :return:
        display all scheduled policies
        {
            policy_id:id,
            policy_name：name
        }
        """
        self.server.parse_base_info()
        cur_policy = []
        for policy in self.server.schedulers:
            cur_policy.append(
                {'policy_id': policy['id'],
                 'policy_name': policy['name']})
        return cur_policy

    async def get_installed_services(self):
        """
        get current installed services
        :return:
        ["face_detection", "..."]
        """
        directory = await run_in_threadpool(
            self.server.runtime_orchestrator.active_directory,
        )
        if directory is None:
            return []
        return sorted({
            unit.slot.logical_service
            for unit in directory.routes
            if unit.slot.component == 'processor' and unit.slot.logical_service
        })

    async def get_dag_workflows(self):
        """
        get current dag workflows
        [
                    {
                        "dag_id":1,
                        "dag_name":"...",
                        "dag":{
                            "node_id_A":{
                                "id" : "...",
                                "prev" : [],
                                "succ" : ["node_id_B", ...],
                                "service_id" : "car_detection"
                            },
                            "node_id_B":{
                                "id" : "...",
                                "prev" : ["node_id_A"],
                                "succ" : [],
                                "service_id" : "plate_recognition"
                            }
                        }
                    },
                    {
                        "dag_id":2,
                        "dag_name":"...",
                        "dag":{
                            "node_id_A":{
                                "id" : "...",
                                "prev" : [],
                                "succ" : ["node_id_B", ...],
                                "service_id" : car_detection
                            },
                            "node_id_B":{
                                "id" : "...",
                                "prev" : ["node_id_A"],
                                "succ" : [],
                                "service_id" : plate_recognition
                            }
                        }
                    }
                ],
        :return:
        """

        return self.server.dags

    async def get_all_services(self):
        """
        get all service containers

        [
        {
            "name": "face_detection",
            "description": "face detection"
        },
        {
            "name": "car_detection"，
            "description": "car detection"
        }
    ]
        :return:
        """
        self.server.parse_base_info()
        service_dict = {}
        services = self.server.services
        for service in services:
            service_view = copy.deepcopy(service)
            service_input, input_error = self.server.service_io_labels(service_view, 'input')
            service_output, output_error = self.server.service_io_labels(service_view, 'output')
            if input_error or output_error:
                raise ValueError(input_error or output_error)
            service_view['description'] = (
                f"{service_view['description']} "
                f"(in:{', '.join(service_input)}, out:{', '.join(service_output)})"
            )
            service_dict[service_view['id']] = (
                service_view if service_view['id'] not in service_dict else service_dict[service_view['id']]
            )
        return [service_dict[service_id] for service_id in service_dict]

    async def update_dag_workflows(self, data=Body(...)):
        """
        add new dag workflows
        body
        {
            "dag_name":"headup",
            "dag":{
                "_start" : [node_id_A...],//需要是一个列表 支持多个begin
                "service_id":{
                    "prev" : [],
                    "succ" : ["node_id_B", ...],
                    "service_id" : car_detection
                },
                "service_id":{
                    "prev" : ["node_id_A"],
                    "succ" : [],
                    "service_id" : plate_recognition
                }
            }
        },
        :return:
            {'state':success/fail, 'msg':'...'}
        """
        dag_name = data['dag_name']
        dag = data['dag']
        state, msg = self.server.check_dag(dag)
        if state:
            self.server.dags.append({
                'dag_id': Counter.get_count('dag_id'),
                'dag_name': dag_name,
                'dag': dag
            })

            return {'state': 'success', 'msg': 'Add new dag Successfully'}
        else:
            return {'state': 'fail', 'msg': f'Add new dag failed: {msg}'}

    async def delete_dag_workflow(self, data=Body(...)):
        """
        delete dag workflow
        body:
        {
            "dag_id":1
        }
        :return:
        {'state':success/fail, 'msg':'...'}
        """

        data = _json_object(data)
        dag_id = int(data['dag_id'])
        for index, dag in enumerate(self.server.dags):
            if dag['dag_id'] == dag_id:
                del self.server.dags[index]
                return {'state': 'success', 'msg': 'Delete dag successfully'}

        return {'state': 'fail', 'msg': 'Delete dag failed: dag not exists'}

    async def get_service_info(self, service):
        """
        get information of installed service container
        :param service:
        :return:
        [
            {
                "ip":114.212.81.11
                "hostname"
                “cpu”:
                "memory":
                "bandwidth"
                "age"
            }
        ]

        """
        try:
            if service == 'null':
                return []
            # This is an in-memory, generation-bound deep copy. Browser polls
            # never list Pods, Nodes, RuntimeServices, or metrics resources.
            telemetry = self.server.get_runtime_telemetry(
                logical_service=service,
            )
            if not telemetry.get('install_id') or not telemetry.get('directory_revision'):
                return []
            metrics = telemetry.get('runtime_metrics') or {}
            resource_data = telemetry.get('resource') or {}
            shared_bandwidth = _shared_bandwidth(
                resource_data,
                has_sample=telemetry.get('resource_sampled_at') is not None,
                stale=bool(telemetry.get('resource_stale')),
            )
            has_metrics_sample = telemetry.get('runtime_metrics_sampled_at') is not None
            info = []
            for metric in metrics.values():
                hostname = metric.get('node', '')
                resource_usage = metric.get('resource_usage') or {}
                info.append({
                    'ip': (metric.get('node_info') or {}).get('address', ''),
                    'hostname': hostname,
                    'cpu': _resource_detail(
                        resource_usage.get('cpu'),
                        'cpu',
                        has_metrics_sample,
                    ),
                    'memory': _resource_detail(
                        resource_usage.get('memory'),
                        'memory',
                        has_metrics_sample,
                    ),
                    'bandwidth': copy.deepcopy(shared_bandwidth),
                    'age': metric.get('created_at', ''),
                })
            info.sort(key=lambda item: item['hostname'])
        except Exception as e:
            LOGGER.exception(e)
            return []

        return info

    async def upload_datasource_config_file(self, request: Request):
        """
        body: file/files
        :return:
            {'state':success/fail, 'msg':'...'}
        """
        form = await request.form()
        upload_items = []
        for field_name in ("files", "file"):
            for upload in form.getlist(field_name):
                if isinstance(upload, UploadFile) or (
                    hasattr(upload, "filename") and hasattr(upload, "read")
                ):
                    upload_items.append(upload)

        if not upload_items:
            return {'state': 'fail', 'msg': 'No datasource config files uploaded', 'results': []}

        results = []
        success_count = 0

        for upload in upload_items:
            file_name = upload.filename or 'datasource_config.yaml'
            file_data = await upload.read()
            suffix = Path(file_name).suffix or '.yaml'
            temp_path = None

            try:
                with tempfile.NamedTemporaryFile(prefix='datasource_config_', suffix=suffix, delete=False) as buffer:
                    buffer.write(file_data)
                    temp_path = buffer.name

                config = self.server.check_datasource_config(temp_path)
            finally:
                if temp_path:
                    FileOps.remove_file(temp_path)

            if config:
                datasource_config = self.server.fill_datasource_config(config)
                self.server.source_configs.append(datasource_config)
                success_count += 1
                results.append({
                    'filename': file_name,
                    'state': 'success',
                    'msg': 'Datasource configured successfully',
                    'source_label': datasource_config['source_label'],
                })
            else:
                results.append({
                    'filename': file_name,
                    'state': 'fail',
                    'msg': 'Datasource configured failed, please check uploading file format',
                })

        total_count = len(upload_items)
        if success_count == total_count:
            state = 'success'
            msg = 'Datasource configured successfully' if total_count == 1 else \
                f'Datasource configured successfully for {success_count} file(s)'
        elif success_count == 0:
            state = 'fail'
            msg = 'Datasource configured failed, please check uploading file format'
        else:
            state = 'partial'
            msg = f'Datasource configured successfully for {success_count} of {total_count} file(s)'

        return {
            'state': state,
            'msg': msg,
            'results': results,
        }

    async def get_datasource_info(self):
        """
        :return:

            [
            {
                "source_label": "car"
                "source_name": "config1",
                “source_type”: "video",
                "source_mode": "http_video",
                "camera_list":[
                    {
                        "name": "camera1",
                        "url": "rtsp/114.212.81.11...",
                        "describe":""
                        “resolution”: "1080p"
                        "fps":"25fps"
                        "importance": 4

                    },
                    {}
                ]
            }
        ]
        """
        return self.server.source_configs

    async def delete_datasource_info(self, data=Body(...)):
        """
        delete dag source info
        body:
        {
            "source_label":...
        }
        :return:
        {'state':success/fail, 'msg':'...'}
        """

        data = json.loads(str(data, encoding='utf-8'))
        source_label = data['source_label']
        for index, datasource in enumerate(self.server.source_configs):
            if datasource['source_label'] == source_label:
                del self.server.source_configs[index]
                return {'state': 'success', 'msg': 'Delete datasource successfully'}

        return {'state': 'fail', 'msg': 'Delete datasource failed: datasource not exists'}

    async def install_service(self, data=Body(...)):
        """
        install system components to prepare for executing dags
        body
        {
            "dag_id": (id),
            "policy_id": (id)
        }

        content = {
            source_config_label: source_config_label,
            policy_id: policy_id,
            source: this.selectedSources,
        };

        source = [
            { id: 0, name: "s1", dag_selected: "", node_selected: [] },
            { id: 1, name: "s2", dag_selected: "", node_selected: [] }...
        ],

        :return:
        {'msg': 'service start successfully'}
        {'msg': 'Invalid service name!'}
        """

        try:
            data = _json_object(data)
            raw_install_id = data.get('install_id')
            if raw_install_id is not None and not isinstance(raw_install_id, str):
                raise TypeError('install_id must be a string')
            install_id = str(raw_install_id or '').strip()
            source_label = data['source_config_label']
            policy_id = data['policy_id']
            if not isinstance(source_label, str) or not isinstance(policy_id, str):
                raise TypeError('source_config_label and policy_id must be strings')
            source_map_list = data['source']
            if not isinstance(source_map_list, list):
                raise TypeError('source must be a list')
            if any(not isinstance(item, dict) for item in source_map_list):
                raise TypeError('every source mapping must be an object')
            dag_list = [item['dag_selected'] for item in source_map_list]
            node_set_list = [item['node_selected'] for item in source_map_list]
            if any(not isinstance(nodes, list) for nodes in node_set_list):
                raise TypeError('node_selected must be a list')
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return {
                'state': 'fail',
                'msg': 'Install services failed: invalid request body',
            }

        source_deploy = []

        policy = self.server.find_scheduler_policy_by_id(policy_id)
        if policy is None:
            return {'state': 'fail', 'msg': 'Install services failed: scheduler policy not exists'}

        source_config = self.server.find_datasource_configuration_by_label(source_label)
        if source_config is None:
            return {'state': 'fail', 'msg': 'Install services failed: datasource configuration not exists'}

        source_list = source_config.get('source_list') or []
        if not (len(source_list) == len(dag_list) == len(node_set_list)):
            return {'state': 'fail', 'msg': 'Install services failed: datasource mapping failed'}

        for source, dag_id, node_set in zip(copy.deepcopy(source_list), dag_list, node_set_list):

            dag = self.server.find_dag_by_id(dag_id)
            if dag is None:
                return {'state': 'fail', 'msg': 'Install services failed: dag not exists'}

            source.update({'source_type': source_config['source_type'], 'source_mode': source_config['source_mode']})

            source_deploy.append({'source': source, 'dag': dag, 'node_set': node_set})

        try:
            result, msg = await run_in_threadpool(
                self.server.parse_and_apply_templates,
                policy,
                source_deploy,
                source_label,
                install_id,
            )
        except Exception as e:
            LOGGER.warning(f'Parse and apply templates failed: {str(e)}')
            LOGGER.exception(e)
            result = False
            msg = 'unexpected system error, please refer to logs in backend'

        if result:
            response = {
                'state': 'success',
                'msg': 'Install services successfully',
            }
            if not self.server.inner_datasource:
                try:
                    query_result = await self.submit_query(
                        {'source_label': source_label},
                    )
                    if query_result.get('state') != 'success':
                        response['warning'] = (
                            'Runtime installed, but datasource query did not start: '
                            f"{query_result.get('msg') or 'unknown error'}"
                        )
                except Exception as exc:
                    LOGGER.warning(
                        f'Runtime installed, but datasource query did not start: {exc}'
                    )
                    LOGGER.exception(exc)
                    response['warning'] = (
                        'Runtime installed, but datasource query did not start; '
                        'start it explicitly after checking backend logs'
                    )
            return response
        else:
            return {'state': 'fail', 'msg': f'Install services failed: {msg}'}

    async def uninstall_service(self, data=Body(default=None)):
        """
        {'state':"success/fail",'msg':'...'}

        :return:
        """

        try:
            payload = _json_object(data)
            raw_install_id = payload.get('install_id')
            if raw_install_id is not None and not isinstance(raw_install_id, str):
                raise TypeError('install_id must be a string')
            expected_install_id = str(raw_install_id or '').strip()
        except (TypeError, ValueError, json.JSONDecodeError):
            return {
                'state': 'fail',
                'msg': 'Uninstall services failed: invalid request body',
            }

        try:
            # BackendCore owns query admission and cancellation as part of the
            # same lifecycle operation.  Keeping it below the threadpool
            # boundary prevents a stop/reopen race before uninstall begins.
            result, msg = await run_in_threadpool(
                self.server.parse_and_delete_templates,
                expected_install_id,
            )

        except Exception as e:
            LOGGER.warning(f'Uninstall services failed: {str(e)}')
            LOGGER.exception(e)
            result = False
            msg = 'unexpected system error, please refer to logs in backend'

        if result:
            return {'state': 'success', 'msg': msg}
        else:
            return {'state': 'fail', 'msg': f'Uninstall services failed: {msg}'}

    @staticmethod
    def _install_state_response(
            session, pending=None, local_ready=False, local_error=''):
        pending = pending if isinstance(pending, dict) else {}
        has_pending = bool(pending)
        pending_kind = str(pending.get('kind') or 'install')
        pending_install_id = str(pending.get('install_id') or '')
        pending_phase = str(pending.get('phase') or 'preparing-install')
        pending_operation_id = str(pending.get('operation_id') or '')
        install_pending = bool(
            pending_install_id and pending_kind == 'install'
        )
        if session is None:
            return {
                'state': 'uninstall',
                'phase': pending_phase if has_pending else 'uninstalled',
                'ready': False,
                'install_id': pending_install_id,
                'install_pending': install_pending,
                'operation_id': pending_operation_id,
                'updated_at': '',
                'active_directory_revision': 0,
                'active_runtime_count': 0,
                'pending_runtime_count': 0,
                'cleanup_runtime_count': 0,
                'cleanup': None,
                'retirement_revision': 0,
                'retirement_deadline': None,
                'last_error': '',
            }

        retirement = session.retirement
        same_pending_target = pending_install_id == session.install_id
        same_pending_install = same_pending_target and pending_kind == 'install'
        phase = session.phase
        operation_id = session.operation_id
        if same_pending_target and pending_kind == 'stop':
            phase = pending_phase
            operation_id = pending_operation_id
        elif same_pending_install and pending_phase == 'cancelling-install':
            phase = pending_phase
            operation_id = pending_operation_id
        elif session.phase == 'active' and local_error:
            # Keep ownership as ``install`` so exact uninstall remains
            # available, while allowing clients waiting for installation to
            # converge on a terminal failure instead of polling forever.
            phase = 'failed'
        return {
            # ``state`` remains the ownership guard used to reject a second
            # installation. ``ready``/``phase`` express whether read APIs may
            # consume the atomically published RuntimeDirectory.
            'state': 'install',
            'phase': phase,
            'ready': phase == 'active' and bool(local_ready) and not same_pending_install,
            'install_id': session.install_id,
            'install_pending': same_pending_install,
            'operation_id': operation_id,
            'updated_at': session.updated_at,
            'active_directory_revision': session.active_directory_revision,
            'active_runtime_count': len(session.active),
            'pending_runtime_count': len(session.pending),
            'cleanup_runtime_count': len(session.cleanup),
            'cleanup': _cleanup_diagnostics(session, phase),
            'retirement_revision': retirement.revision if retirement else 0,
            'retirement_deadline': retirement.deadline if retirement else None,
            'last_error': local_error or session.last_error,
        }

    async def get_install_state(self):
        """
        :return:
        {'state':'install/uninstall'}
        """

        # The first process-local snapshot load may read the Kubernetes
        # ConfigMap.  Keep even that one-off synchronous call off the event
        # loop; subsequent reads are the in-memory snapshot fast path.
        session, pending, local_ready, local_error = await run_in_threadpool(
            self.server.management_lifecycle_snapshot,
        )
        return self._install_state_response(
            session, pending, local_ready, local_error,
        )

    async def submit_query(self, data=Body(...)):
        """
        body
        {
            "source_label": "..."
        }
        :return:
        {'msg': 'Datasource open successfully'}
        {'msg': 'Invalid service name'}
        """

        return await run_in_threadpool(self._submit_query, data)

    def _submit_query(self, data):
        if isinstance(data, dict):
            parsed_data = data
        elif isinstance(data, bytes):
            parsed_data = json.loads(data.decode("utf-8"))
        elif isinstance(data, str):
            parsed_data = json.loads(data)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")
        data = parsed_data

        source_label = data['source_label']
        result, message = self.server.open_query(source_label)
        return {'state': 'success' if result else 'fail', 'msg': message}

    async def stop_query(self):
        """
        {'source_label':'...'}
        :return:
        {'state':"success/fail",'msg':'...'}
        """

        return await run_in_threadpool(self._stop_query)

    def _stop_query(self):
        result, message = self.server.close_query()
        return {'state': 'success' if result else 'fail', 'msg': message}

    async def get_query_state(self):
        """

        :return:
        {'state':'open/close','source_label':''}
        """
        snapshot = await run_in_threadpool(self.server.query_snapshot)
        if self.server.inner_datasource:
            state = 'open' if snapshot['open'] else 'close'
        else:
            state = 'disabled'

        return {'state': state,
                'source_label': snapshot['source_label']
                }

    async def get_source_list(self):
        """
        [
            {
                "id":"video_source1",
                "label":"source1"
            },
            ...
        ]
        :return:
        """
        snapshot = await run_in_threadpool(self.server.query_snapshot)
        if not snapshot['open']:
            return []

        source_config = self.server.find_datasource_configuration_by_label(
            snapshot['source_label'],
        )
        if not source_config:
            return []

        return [{'id': source['id'], 'label': source['name']} for source in source_config['source_list']]

    async def get_edge_nodes(self):
        return await run_in_threadpool(self.server.get_edge_nodes)

    async def get_task_result(self):
        """
        10 lasted results
        {
        'datasource1':[
            task_id: 12,
            data: {0:{"delay":"0.5"},1:{"image":"xxx"}}

        ],
        'datasource2':[]
        }
        :return:
        """
        session = await run_in_threadpool(
            self.server.runtime_orchestrator.current_session,
        )
        if session is None or session.phase != 'active':
            return {}
        return await run_in_threadpool(self._get_task_result)

    def _get_task_result(self):
        snapshot = self.server.query_snapshot(include_queues=True)
        if not snapshot['open']:
            return {}
        generation = snapshot['generation']
        queues = snapshot['queues']
        ans = {}
        for source_id, task_queue in queues.items():
            if task_queue is not None:
                ans[source_id] = self.server.fetch_visualization_data(
                    source_id,
                    task_queue=task_queue,
                )

        # Visualization may download and transform result artifacts.  If stop
        # or reopen won the race during that work, never publish the previous
        # generation's response to the new query.
        return ans if self.server.is_query_generation_active(generation) else {}

    async def get_system_parameters(self):
        # Rendering and append-only log maintenance are synchronous work; keep
        # them off the event loop just like the Kubernetes telemetry join.
        return await run_in_threadpool(self.server.get_system_parameters)

    async def get_result_visualization_config(self, source_id):
        """
        get visualization configuration
        """
        source_id = int(source_id)
        return self.server.get_result_visualization_config(source_id)

    async def upload_result_visualization_config(self, source_id, file: UploadFile = File(...)):
        """
        body: file
        :return:
            {'state':success/fail, 'msg':'...'}
        """
        source_id = int(source_id)
        file_data = await file.read()
        with open('result_visualization_config.yaml', 'wb') as buffer:
            buffer.write(file_data)

        config = self.server.check_visualization_config('result_visualization_config.yaml')
        FileOps.remove_file('result_visualization_config.yaml')
        if config:
            self.server.customized_source_result_visualization_configs[source_id] = copy.deepcopy(config)
            return {'state': 'success', 'msg': 'Visualization configured successfully'}
        else:
            return {'state': 'fail', 'msg': 'Visualization configured failed, please check uploading file format'}

    async def get_system_visualization_config(self):
        """
        get visualization configuration
        """
        return self.server.get_system_visualization_config()

    async def get_datasource_state(self):
        snapshot = await run_in_threadpool(self.server.query_snapshot)
        state = 'open' if snapshot['open'] else 'close'
        if state == 'close':
            return {'state': state}
        source_label = snapshot['source_label']
        config = self.server.find_datasource_configuration_by_label(source_label)
        if config is None:
            LOGGER.warning(f'Config of "{source_label}" does not exist.')
            return {'state': 'close'}
        return {'state': state, **config}

    async def reset_datasource(self):
        await run_in_threadpool(self.server.close_query)

    async def download_log(self):
        """
        :return:
        file
        """
        file_name = self.server.get_log_file_name()
        if not file_name:
            formatted_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            file_name = f'RESULT_LOG_DAYU_NAMESPACE_{self.server.namespace}_TIME_{formatted_time}'

        upstream_response = await run_in_threadpool(
            self.server.open_result_log_export_stream,
        )
        if upstream_response is None:
            raise HTTPException(status_code=503, detail='Result log export is temporarily unavailable')

        headers = {
            'Content-Disposition': f'attachment; filename="{file_name}.json.gz"',
            'Cache-Control': 'no-store',
        }
        content_length = upstream_response.headers.get('content-length')
        if content_length:
            headers['Content-Length'] = content_length

        def iter_chunks():
            try:
                for chunk in upstream_response.iter_content(chunk_size=8192):
                    if chunk:
                        yield chunk
            finally:
                upstream_response.close()

        return StreamingResponse(
            iter_chunks(),
            media_type='application/gzip',
            headers=headers
        )

    async def download_system_log(self, backtask: BackgroundTasks):
        """Download accumulated system visualization logs without clearing the store."""
        formatted_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        file_name = f'SYSTEM_LOG_DAYU_NAMESPACE_{self.server.namespace}_TIME_{formatted_time}'

        export_path = self.server.create_system_log_export_file()
        backtask.add_task(FileOps.remove_file, export_path)
        return FileResponse(
            path=export_path,
            filename=f'{file_name}.json.gz',
            media_type='application/gzip',
            background=backtask
        )
