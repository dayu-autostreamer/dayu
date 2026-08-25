import copy
import hashlib
import json
import time

from fastapi import FastAPI, Form, HTTPException
from fastapi.exception_handlers import (
    http_exception_handler,
    request_validation_exception_handler,
)
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.routing import APIRoute
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse

from core.lib.common import LOGGER
from core.lib.content import Task
from core.lib.network import NetworkAPIMethod, NetworkAPIPath
from core.lib.scheduling.deployment_plan import dag_services
from core.lib.scheduling import build_schedule_decision

from .scheduler import Scheduler
from .runtime_directory import (
    RuntimeDirectoryConflict,
    RuntimeDirectoryError,
    RuntimeDirectoryNotFound,
)
from .task_lease import TaskLeaseRetired


_API_ERROR_LOG_DETAIL_LIMIT = 1024


class SchedulerServer:
    def __init__(self):
        self.app = FastAPI(routes=[
            APIRoute(NetworkAPIPath.SCHEDULER_SCHEDULE,
                     self.generate_schedule_plan,
                     response_class=JSONResponse,
                     methods=[NetworkAPIMethod.SCHEDULER_SCHEDULE]
                     ),
            APIRoute(NetworkAPIPath.SCHEDULER_OVERHEAD,
                     self.get_schedule_overhead,
                     response_class=JSONResponse,
                     methods=[NetworkAPIMethod.SCHEDULER_OVERHEAD]
                     ),
            APIRoute(NetworkAPIPath.SCHEDULER_SCENARIO,
                     self.update_object_scenario,
                     response_class=JSONResponse,
                     methods=[NetworkAPIMethod.SCHEDULER_SCENARIO]
                     ),
            APIRoute(NetworkAPIPath.SCHEDULER_POST_RESOURCE,
                     self.update_resource_state,
                     response_class=JSONResponse,
                     methods=[NetworkAPIMethod.SCHEDULER_POST_RESOURCE]
                     ),
            APIRoute(NetworkAPIPath.SCHEDULER_GET_RESOURCE,
                     self.get_resource_state,
                     response_class=JSONResponse,
                     methods=[NetworkAPIMethod.SCHEDULER_GET_RESOURCE]
                     ),
            APIRoute(NetworkAPIPath.SCHEDULER_GET_RESOURCE_LOCK,
                     self.get_resource_lock,
                     response_class=JSONResponse,
                     methods=[NetworkAPIMethod.SCHEDULER_GET_RESOURCE_LOCK]
                     ),
            APIRoute(NetworkAPIPath.SCHEDULER_SELECT_SOURCE_NODES,
                     self.generate_source_nodes_selection_plan,
                     response_class=JSONResponse,
                     methods=[NetworkAPIMethod.SCHEDULER_SELECT_SOURCE_NODES]
                     ),
            APIRoute(NetworkAPIPath.SCHEDULER_INITIAL_DEPLOYMENT,
                     self.generate_initial_deployment_plan,
                     response_class=JSONResponse,
                     methods=[NetworkAPIMethod.SCHEDULER_INITIAL_DEPLOYMENT]),
            APIRoute(NetworkAPIPath.SCHEDULER_REDEPLOYMENT,
                     self.generate_redeployment_plan,
                     response_class=JSONResponse,
                     methods=[NetworkAPIMethod.SCHEDULER_REDEPLOYMENT]),
            APIRoute(NetworkAPIPath.SCHEDULER_GENERATION_ADMISSION,
                     self.check_generation_admission,
                     response_class=JSONResponse,
                     methods=[NetworkAPIMethod.SCHEDULER_GENERATION_ADMISSION]),
            APIRoute(NetworkAPIPath.SCHEDULER_RUNTIME_DIRECTORY,
                     self.get_runtime_directory,
                     response_class=JSONResponse,
                     methods=[NetworkAPIMethod.SCHEDULER_GET_RUNTIME_DIRECTORY]),
            APIRoute(NetworkAPIPath.SCHEDULER_RUNTIME_DIRECTORY,
                     self.put_runtime_directory,
                     response_class=JSONResponse,
                     methods=[NetworkAPIMethod.SCHEDULER_PUT_RUNTIME_DIRECTORY]),
            APIRoute(NetworkAPIPath.SCHEDULER_RUNTIME_DIRECTORY,
                     self.clear_runtime_directory,
                     response_class=JSONResponse,
                     methods=[NetworkAPIMethod.SCHEDULER_CLEAR_RUNTIME_DIRECTORY]),
            APIRoute(NetworkAPIPath.SCHEDULER_RUNTIME_DIRECTORY_PROPOSALS,
                     self.propose_runtime_directory,
                     response_class=JSONResponse,
                     methods=[NetworkAPIMethod.SCHEDULER_PROPOSE_RUNTIME_DIRECTORY]),
            APIRoute(NetworkAPIPath.SCHEDULER_RUNTIME_DIRECTORY_PROPOSAL_COMMIT,
                     self.commit_runtime_directory,
                     response_class=JSONResponse,
                     methods=[NetworkAPIMethod.SCHEDULER_COMMIT_RUNTIME_DIRECTORY]),
            APIRoute(NetworkAPIPath.SCHEDULER_RUNTIME_DIRECTORY_PROPOSAL_REJECT,
                     self.reject_runtime_directory,
                     response_class=JSONResponse,
                     methods=[NetworkAPIMethod.SCHEDULER_REJECT_RUNTIME_DIRECTORY]),
            APIRoute(NetworkAPIPath.SCHEDULER_RUNTIME_DIRECTORY_TASK_LEASES,
                     self.count_task_leases,
                     response_class=JSONResponse,
                     methods=[NetworkAPIMethod.SCHEDULER_COUNT_TASK_LEASES]),
            APIRoute(NetworkAPIPath.SCHEDULER_RUNTIME_DIRECTORY_TASK_LEASES,
                     self.acquire_task_lease,
                     response_class=JSONResponse,
                     methods=[NetworkAPIMethod.SCHEDULER_ACQUIRE_TASK_LEASE]),
            APIRoute(NetworkAPIPath.SCHEDULER_RUNTIME_DIRECTORY_TASK_LEASES,
                     self.renew_task_lease,
                     response_class=JSONResponse,
                     methods=[NetworkAPIMethod.SCHEDULER_RENEW_TASK_LEASE]),
            APIRoute(NetworkAPIPath.SCHEDULER_RUNTIME_DIRECTORY_TASK_LEASES,
                     self.release_task_lease,
                     response_class=JSONResponse,
                     methods=[NetworkAPIMethod.SCHEDULER_RELEASE_TASK_LEASE]),
            APIRoute(NetworkAPIPath.SCHEDULER_RUNTIME_DIRECTORY_TASK_LEASES,
                     self.retire_task_leases,
                     response_class=JSONResponse,
                     methods=[NetworkAPIMethod.SCHEDULER_RETIRE_TASK_LEASES]),
            APIRoute(NetworkAPIPath.SCHEDULER_RUNTIME_DIRECTORY_TASK_RESERVATIONS,
                     self.cancel_task_reservation,
                     response_class=JSONResponse,
                     methods=[NetworkAPIMethod.SCHEDULER_CANCEL_TASK_RESERVATION]),
        ])

        self.app.add_middleware(
            CORSMiddleware, allow_origins=["*"], allow_credentials=True,
            allow_methods=["*"], allow_headers=["*"],
        )
        self.app.add_exception_handler(
            StarletteHTTPException,
            self._handle_http_exception,
        )
        self.app.add_exception_handler(
            RequestValidationError,
            self._handle_request_validation_error,
        )

        self.scheduler = Scheduler()

    @staticmethod
    def _compact_log_detail(detail):
        if not isinstance(detail, str):
            detail = json.dumps(
                detail,
                ensure_ascii=False,
                default=str,
                separators=(",", ":"),
            )
        detail = " ".join(detail.split()) or "<empty>"
        if len(detail) > _API_ERROR_LOG_DETAIL_LIMIT:
            detail = f"{detail[:_API_ERROR_LOG_DETAIL_LIMIT - 3]}..."
        return detail

    @classmethod
    def _log_api_error(cls, request, status_code, detail):
        message = (
            f"[Scheduler API] method={request.method} path={request.url.path} "
            f"status={status_code} detail={cls._compact_log_detail(detail)}"
        )
        log = LOGGER.error if status_code >= 500 else LOGGER.warning
        log(message)

    @classmethod
    async def _handle_http_exception(
        cls,
        request: Request,
        exc: StarletteHTTPException,
    ):
        cls._log_api_error(request, exc.status_code, exc.detail)
        return await http_exception_handler(request, exc)

    @classmethod
    async def _handle_request_validation_error(
        cls,
        request: Request,
        exc: RequestValidationError,
    ):
        detail = [
            {
                "loc": error.get("loc", ()),
                "msg": error.get("msg", ""),
                "type": error.get("type", ""),
            }
            for error in exc.errors()
        ]
        cls._log_api_error(request, 422, detail)
        return await request_validation_exception_handler(request, exc)

    @staticmethod
    def _split_schedule_plan(plan, current_deployment_version=0):
        deployment_version = current_deployment_version
        if isinstance(plan, dict) and 'deployment_version' in plan:
            plan = plan.copy()
            deployment_version = plan.pop('deployment_version')
            if deployment_version is None:
                deployment_version = current_deployment_version
        return plan, deployment_version

    @staticmethod
    def _dag_has_runtime_targets(dag):
        if not isinstance(dag, dict):
            return False
        has_runtime_service = False
        for service_name, node in dag.items():
            if service_name in ('_start', '_end', 'start', 'end'):
                continue
            has_runtime_service = True
            service = node.get('service') if isinstance(node, dict) else None
            if not isinstance(service, dict):
                return False
            if not str(service.get('execute_device') or '').strip():
                return False
        return has_runtime_service

    def _complete_schedule_plan(self, plan, request):
        """Compose a full effective plan from a policy's partial decision.

        Policy hooks may schedule configuration, DAG offloading, deployment
        version, or any combination.  Unspecified dimensions inherit the
        current Generator state.  Only an unroutable first DAG is supplied by
        the selected startup-policy hook; the framework itself chooses no
        algorithm-specific default.
        """

        if not isinstance(plan, dict):
            raise HTTPException(
                status_code=422,
                detail='schedule policy must return an object',
            )
        request = request if isinstance(request, dict) else {}
        current_configuration = request.get('current_configuration', {})
        if current_configuration is None:
            current_configuration = {}
        if not isinstance(current_configuration, dict):
            raise HTTPException(
                status_code=422,
                detail='current_configuration must be an object',
            )

        effective = copy.deepcopy(current_configuration)
        effective.update(copy.deepcopy(plan))
        dag = effective.get('dag')
        if dag is None:
            current_dag = copy.deepcopy(request.get('dag'))
            if self._dag_has_runtime_targets(current_dag):
                dag = current_dag
            else:
                startup_plan = self.scheduler.get_startup_policy(request)
                if not isinstance(startup_plan, dict):
                    raise HTTPException(
                        status_code=422,
                        detail='startup policy must return an object',
                    )
                dag = copy.deepcopy(startup_plan.get('dag'))
        if not isinstance(dag, dict):
            raise HTTPException(
                status_code=422,
                detail='effective schedule plan must contain a dag object',
            )
        effective['dag'] = dag
        return effective

    def _runtime_state_for_plan(self, plan, source_device):
        try:
            return self.scheduler.schedule_runtime_state(
                plan,
                source_device=source_device,
            )
        except RuntimeDirectoryError as exc:
            raise HTTPException(
                status_code=503,
                detail=f'no valid runtime route for schedule plan: {exc}',
            )

    @staticmethod
    def _route_cache_key(routes):
        payload = json.dumps(
            routes,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    async def generate_schedule_plan(self, data: str = Form(...)):
        request_started = time.monotonic()
        data = json.loads(data)

        task_context = data.get('task_context')
        if task_context is not None:
            if not isinstance(task_context, dict):
                raise HTTPException(status_code=422, detail='task_context must be an object')
            if task_context.get('source_id') != data.get('source_id'):
                raise HTTPException(
                    status_code=422,
                    detail='task_context source_id must match schedule source_id',
                )

        with self.scheduler.schedule_transaction():
            state_started = time.monotonic()
            try:
                current_revision = int(
                    self.scheduler.runtime_directory_revision()
                )
            except (TypeError, ValueError):
                current_revision = 0
            if current_revision < 1:
                # Runtime components are activated before the orchestrator
                # publishes revision 1.  Do not construct or invoke any
                # scheduling extension until an immutable routing snapshot is
                # available for the resulting plan.
                raise HTTPException(
                    status_code=503,
                    detail='runtime directory is not ready for scheduling',
                )
            self.scheduler.register_schedule_table(data['source_id'])
            state_seconds = time.monotonic() - state_started
            try:
                schedule_attempt = int(data.get('schedule_request_attempt') or 0)
            except (TypeError, ValueError):
                schedule_attempt = 0
            # A fresh root UUID cannot have a persisted reservation.  Skip one
            # Redis lookup on its first request; a retry explicitly carries an
            # incremented attempt and restores the idempotent lookup path.
            reservation = None
            lookup_started = time.monotonic()
            if schedule_attempt != 1:
                reservation = self.scheduler.get_task_reservation(
                    current_revision,
                    task_context.get('root_uuid') if task_context else '',
                    task_context,
                )
            lookup_seconds = time.monotonic() - lookup_started
            agent_started = time.monotonic()
            if reservation is None:
                plan, deployment_version = self._split_schedule_plan(
                    self.scheduler.get_schedule_plan(data),
                    data.get('deployment_version', 0),
                )
                plan = self._complete_schedule_plan(plan, data)
            else:
                plan = copy.deepcopy(reservation.get('plan'))
                if not isinstance(plan, dict):
                    raise self._translate_runtime_error(
                        RuntimeDirectoryError('persisted task reservation has no valid plan'),
                        data.get('source_id'),
                    )
                deployment_version = reservation.get('deployment_version', 0)
                plan = self._complete_schedule_plan(plan, data)
            agent_seconds = time.monotonic() - agent_started

            routing_started = time.monotonic()
            runtime_state = self._runtime_state_for_plan(
                plan,
                data.get('source_device', ''),
            )

            if reservation is not None and runtime_state['revision'] != current_revision:
                reservation = None
                plan, deployment_version = self._split_schedule_plan(
                    self.scheduler.get_schedule_plan(data),
                    data.get('deployment_version', 0),
                )
                plan = self._complete_schedule_plan(plan, data)
                runtime_state = self._runtime_state_for_plan(
                    plan,
                    data.get('source_device', ''),
                )
            routing_seconds = time.monotonic() - routing_started

            response_started = time.monotonic()
            client_revision = data.get('runtime_directory_revision')
            try:
                client_revision = int(client_revision)
            except (TypeError, ValueError):
                client_revision = 0
            client_hash = str(data.get('runtime_directory_hash') or '').strip()
            directory_unchanged = (
                client_revision == runtime_state['revision']
                and client_hash == runtime_state['hash']
            )
            route_cache_key = self._route_cache_key(runtime_state['routes'])
            cached_route_keys = data.get('runtime_route_cache_keys') or []
            if not isinstance(cached_route_keys, (list, tuple, set)):
                cached_route_keys = []
            routes_cached = (
                directory_unchanged
                and route_cache_key in {str(item) for item in cached_route_keys}
            )
            response = {
                'plan': plan,
                'deployment_version': deployment_version,
                'runtime_directory_revision': runtime_state['revision'],
                'runtime_directory_hash': runtime_state['hash'],
                'runtime_directory_unchanged': directory_unchanged,
                'runtime_routes_cache_key': route_cache_key,
                'runtime_routes_cached': routes_cached,
            }
            if not directory_unchanged:
                response['deployment'] = runtime_state['deployment']
            if not routes_cached:
                response['runtime_routes'] = runtime_state['routes']
            if reservation is None:
                response['schedule_decision'] = build_schedule_decision(
                    data,
                    plan,
                    deployment_version,
                    runtime_state['revision'],
                )
            else:
                response['schedule_decision'] = {
                    key: reservation.get(key)
                    for key in (
                        'decision_id',
                        'plan_digest',
                        'source_id',
                        'task_id',
                        'root_uuid',
                    )
                }
            response_seconds = time.monotonic() - response_started
            decision = response['schedule_decision']
            staging_started = time.monotonic()
            if decision.get('root_uuid'):
                try:
                    self.scheduler.stage_task_context(
                        runtime_state['revision'],
                        decision['root_uuid'],
                        {
                            **decision,
                            'runtime_directory_revision': runtime_state['revision'],
                            'deployment_version': deployment_version,
                            'plan': copy.deepcopy(plan),
                            'metadata': copy.deepcopy(
                                (
                                    reservation.get('metadata')
                                    if reservation is not None
                                    else data.get('meta_data')
                                ) or {}
                            ),
                        },
                    )
                except (RuntimeDirectoryError, TypeError, ValueError) as exc:
                    raise self._translate_runtime_error(exc, data.get('source_id'))
            staging_seconds = time.monotonic() - staging_started

        LOGGER.info(
            "[SchedulePath] source=%s task=%s attempt=%s reused=%s "
            "directory_unchanged=%s routes_cached=%s state=%.4fs "
            "lookup=%.4fs agent=%.4fs routing=%.4fs response=%.4fs "
            "staging=%.4fs total=%.4fs",
            data.get('source_id'),
            (task_context or {}).get('task_id'),
            schedule_attempt,
            reservation is not None,
            directory_unchanged,
            routes_cached,
            state_seconds,
            lookup_seconds,
            agent_seconds,
            routing_seconds,
            response_seconds,
            staging_seconds,
            time.monotonic() - request_started,
        )
        return response

    @staticmethod
    def _translate_runtime_error(exc, source_id=None):
        detail = str(exc)
        if source_id is not None:
            detail = f"source_id={source_id}: {detail}"
        if isinstance(exc, RuntimeDirectoryNotFound):
            return HTTPException(status_code=404, detail=detail)
        if isinstance(exc, RuntimeDirectoryConflict):
            return HTTPException(status_code=409, detail=detail)
        return HTTPException(status_code=422, detail=detail)

    def get_runtime_directory(self):
        return self.scheduler.runtime_directory_snapshot()

    def put_runtime_directory(self, data: str = Form(...)):
        payload = json.loads(data)
        directory = payload.get('directory', payload)
        expected_revision = payload.get('expected_revision', payload.get('expectedRevision'))
        try:
            return self.scheduler.replace_runtime_directory(directory, expected_revision)
        except (RuntimeDirectoryError, TypeError, ValueError) as exc:
            raise self._translate_runtime_error(exc)

    def clear_runtime_directory(self, data: str = Form(...)):
        payload = json.loads(data)
        try:
            return self.scheduler.clear_runtime_directory(payload.get('install_id'))
        except (RuntimeDirectoryError, TypeError, ValueError) as exc:
            raise self._translate_runtime_error(exc)

    def propose_runtime_directory(self, data: str = Form(...)):
        payload = json.loads(data)
        try:
            return self.scheduler.propose_runtime_directory(
                payload.get('directory'),
                base_revision=payload.get('base_revision', payload.get('baseRevision')),
                proposal_id=payload.get('proposal_id', payload.get('proposalID')),
                ttl_seconds=payload.get('ttl_seconds', payload.get('ttlSeconds', 60.0)),
            )
        except (RuntimeDirectoryError, TypeError, ValueError) as exc:
            raise self._translate_runtime_error(exc)

    def commit_runtime_directory(self, proposal_id: str, data: str = Form(...)):
        payload = json.loads(data)
        try:
            return self.scheduler.commit_runtime_directory(
                proposal_id,
                payload.get('expected_revision', payload.get('expectedRevision')),
                payload.get(
                    'retirement_grace_seconds',
                    payload.get('retirementGraceSeconds'),
                ),
            )
        except (RuntimeDirectoryError, TypeError, ValueError) as exc:
            raise self._translate_runtime_error(exc)

    def reject_runtime_directory(self, proposal_id: str, data: str = Form('{}')):
        payload = json.loads(data)
        try:
            return self.scheduler.reject_runtime_directory(proposal_id, payload.get('reason', ''))
        except (RuntimeDirectoryError, TypeError, ValueError) as exc:
            raise self._translate_runtime_error(exc)

    def count_task_leases(self, revision: int):
        try:
            return self.scheduler.task_lease_status(revision)
        except (RuntimeDirectoryError, TypeError, ValueError) as exc:
            raise self._translate_runtime_error(exc)

    def retire_task_leases(self, data: str = Form(...)):
        payload = json.loads(data)
        try:
            return self.scheduler.retire_task_leases(
                payload.get('revision', payload.get('runtime_directory_revision')),
                payload.get('deadline'),
            )
        except (RuntimeDirectoryError, TypeError, ValueError) as exc:
            raise self._translate_runtime_error(exc)

    def acquire_task_lease(self, data: str = Form(...)):
        payload = json.loads(data)
        revision = payload.get('revision', payload.get('runtime_directory_revision'))
        root_uuid = payload.get('root_uuid', payload.get('rootUUID'))
        try:
            return self.scheduler.acquire_task_lease(
                revision,
                root_uuid,
                payload.get('ttl_seconds', payload.get('ttlSeconds', 60.0)),
                commitment=payload.get('commitment'),
            )
        except TaskLeaseRetired as exc:
            return self._retired_task_lease(exc, root_uuid)
        except (RuntimeDirectoryError, TypeError, ValueError) as exc:
            raise self._translate_runtime_error(exc)

    def renew_task_lease(self, data: str = Form(...)):
        payload = json.loads(data)
        revision = payload.get('revision', payload.get('runtime_directory_revision'))
        root_uuid = payload.get('root_uuid', payload.get('rootUUID'))
        try:
            return self.scheduler.renew_task_lease(
                revision,
                root_uuid,
                payload.get('ttl_seconds', payload.get('ttlSeconds', 60.0)),
            )
        except TaskLeaseRetired as exc:
            return self._retired_task_lease(exc, root_uuid)
        except (RuntimeDirectoryError, TypeError, ValueError) as exc:
            raise self._translate_runtime_error(exc)

    @staticmethod
    def _retired_task_lease(exc, root_uuid):
        return {
            'revision': exc.revision,
            'root_uuid': str(root_uuid or ''),
            'retired': True,
            'deadline': exc.deadline,
        }

    def release_task_lease(self, data: str = Form(...)):
        payload = json.loads(data)
        try:
            return self.scheduler.release_task_lease(
                payload.get('revision', payload.get('runtime_directory_revision')),
                payload.get('root_uuid', payload.get('rootUUID')),
            )
        except (RuntimeDirectoryError, TypeError, ValueError) as exc:
            raise self._translate_runtime_error(exc)

    def cancel_task_reservation(self, data: str = Form(...)):
        payload = json.loads(data)
        try:
            return self.scheduler.cancel_task_reservation(
                payload.get('revision', payload.get('runtime_directory_revision')),
                payload.get('root_uuid', payload.get('rootUUID')),
                decision_id=payload.get('decision_id', payload.get('decisionID')),
            )
        except (RuntimeDirectoryError, TypeError, ValueError) as exc:
            raise self._translate_runtime_error(exc)

    async def get_schedule_overhead(self):
        return self.scheduler.get_schedule_overhead()

    async def check_generation_admission(self, data: str = Form(...)):
        data = json.loads(data)
        source_id = int(data['source_id'])

        self.scheduler.register_schedule_table(source_id)
        return self.scheduler.should_generate(source_id, data)

    async def update_object_scenario(self, data: str = Form(...)):
        task = Task.deserialize(data)

        return {
            'accepted': bool(self.scheduler.update_scheduler_scenario(task)),
        }

    async def update_resource_state(self, data: str = Form(...)):
        data = json.loads(data)

        self.scheduler.register_resource_table(data['device'])
        self.scheduler.update_scheduler_resource(data)

    async def get_resource_state(self):
        return self.scheduler.get_scheduler_resource()

    async def get_resource_lock(self, data: str = Form(...)):
        data = json.loads(data)
        holder = await self.scheduler.get_resource_lock(data)
        return {'holder': holder}

    async def generate_source_nodes_selection_plan(self, data: str = Form(...)):
        data = json.loads(data)

        plan = {}
        for source_data in data:
            source_id = int(source_data['source']['id'])
            self.scheduler.register_schedule_table(source_id=source_id)
            try:
                source_plan = self.scheduler.get_source_node_selection_plan(source_id, source_data)
            except (RuntimeDirectoryError, ValueError) as exc:
                raise self._translate_runtime_error(exc, source_id)
            plan[source_id] = source_plan

        # LOGGER.info(f'[Source Node Selection] (all sources) Selection policy: {plan}')
        return {'plan': plan}

    async def generate_initial_deployment_plan(self, data: str = Form(...)):
        data = json.loads(data)

        plan = {}
        for source_data in data:
            source_id = source_data['source']['id']
            self.scheduler.register_schedule_table(source_id=source_id)
            try:
                source_plan = self.scheduler.get_initial_deployment_plan(source_id, source_data)
                self._merge_deployment_plan(
                    plan, source_plan, allowed_services=dag_services(source_data),
                )
            except (RuntimeDirectoryError, ValueError) as exc:
                raise self._translate_runtime_error(exc, source_id)

        return {'plan': plan}

    async def generate_redeployment_plan(self, data: str = Form(...)):
        data = json.loads(data)
        plan = {}
        # A redeployment response is one global plan assembled from all source
        # DAGs. Keep the active runtime revision immutable for the whole read so
        # a concurrent directory commit cannot mix two deployments.
        with self.scheduler.schedule_transaction():
            for source_data in data:
                source_id = source_data['source']['id']
                self.scheduler.register_schedule_table(source_id=source_id)
                try:
                    source_plan = self.scheduler.get_redeployment_plan(source_id, source_data)
                    self._merge_deployment_plan(
                        plan, source_plan, allowed_services=dag_services(source_data),
                    )
                except (RuntimeDirectoryError, ValueError) as exc:
                    raise self._translate_runtime_error(exc, source_id)

        return {'plan': plan}

    @staticmethod
    def _merge_deployment_plan(plan, source_plan, allowed_services):
        """Merge the sole deployment contract: service -> JSON node list."""
        if not isinstance(source_plan, dict):
            raise RuntimeDirectoryError("deployment policy must return an object")
        allowed_services = {str(service) for service in allowed_services}
        actual_services = {str(service) for service in source_plan}
        unknown = sorted(actual_services - allowed_services)
        missing = sorted(allowed_services - actual_services)
        if unknown:
            raise RuntimeDirectoryError(
                f"deployment policy returned services outside the current DAG: {unknown}"
            )
        if missing:
            raise RuntimeDirectoryError(
                f"deployment policy omitted current DAG services: {missing}"
            )
        for raw_service, raw_nodes in source_plan.items():
            service = str(raw_service or "").strip()
            if not service:
                raise RuntimeDirectoryError("deployment policy returned an empty service name")
            if not isinstance(raw_nodes, list):
                raise RuntimeDirectoryError(
                    f"deployment policy for service {service!r} must return a JSON node list"
                )
            nodes = [str(node or "").strip() for node in raw_nodes]
            if any(not node for node in nodes):
                raise RuntimeDirectoryError(
                    f"deployment policy for service {service!r} returned an empty node name"
                )
            if not nodes:
                raise RuntimeDirectoryError(
                    f"deployment policy for service {service!r} returned no target nodes"
                )
            plan[service] = sorted(set(plan.get(service, ())) | set(nodes))
