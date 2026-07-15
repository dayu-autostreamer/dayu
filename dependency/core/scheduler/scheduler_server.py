import json

from fastapi import FastAPI, Form, HTTPException
from fastapi.routing import APIRoute
from starlette.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from core.lib.network import NetworkAPIMethod, NetworkAPIPath
from core.lib.content import Task
from core.lib.scheduling.deployment_plan import dag_services

from .scheduler import Scheduler
from .runtime_directory import (
    RuntimeDirectoryConflict,
    RuntimeDirectoryError,
    RuntimeDirectoryNotFound,
)
from .task_lease import TaskLeaseRetired


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
        ], log_level='trace', timeout=6000)

        self.app.add_middleware(
            CORSMiddleware, allow_origins=["*"], allow_credentials=True,
            allow_methods=["*"], allow_headers=["*"],
        )

        self.scheduler = Scheduler()

    async def generate_schedule_plan(self, data: str = Form(...)):
        data = json.loads(data)

        self.scheduler.register_schedule_table(data['source_id'])
        plan = self.scheduler.get_schedule_plan(data)

        deployment_version = 0
        if isinstance(plan, dict) and 'deployment_version' in plan:
            plan = plan.copy()
            deployment_version = plan.pop('deployment_version')
            if deployment_version is None:
                deployment_version = 0

        try:
            runtime_state = self.scheduler.schedule_runtime_state(
                plan,
                source_device=data.get('source_device', ''),
            )
        except RuntimeDirectoryError as exc:
            raise HTTPException(status_code=503, detail=f'no valid runtime route for schedule plan: {exc}')

        response = {
            'plan': plan,
            'deployment': runtime_state['deployment'],
            'deployment_version': deployment_version,
            'runtime_directory_revision': runtime_state['revision'],
            'runtime_directory_hash': runtime_state['hash'],
            'runtime_routes': runtime_state['routes'],
        }

        return response

    @staticmethod
    def _translate_runtime_error(exc):
        if isinstance(exc, RuntimeDirectoryNotFound):
            return HTTPException(status_code=404, detail=str(exc))
        if isinstance(exc, RuntimeDirectoryConflict):
            return HTTPException(status_code=409, detail=str(exc))
        return HTTPException(status_code=422, detail=str(exc))

    async def get_runtime_directory(self):
        return self.scheduler.runtime_directory_snapshot()

    async def put_runtime_directory(self, data: str = Form(...)):
        payload = json.loads(data)
        directory = payload.get('directory', payload)
        expected_revision = payload.get('expected_revision', payload.get('expectedRevision'))
        try:
            return self.scheduler.replace_runtime_directory(directory, expected_revision)
        except (RuntimeDirectoryError, TypeError, ValueError) as exc:
            raise self._translate_runtime_error(exc)

    async def clear_runtime_directory(self, data: str = Form(...)):
        payload = json.loads(data)
        try:
            return self.scheduler.clear_runtime_directory(payload.get('install_id'))
        except (RuntimeDirectoryError, TypeError, ValueError) as exc:
            raise self._translate_runtime_error(exc)

    async def propose_runtime_directory(self, data: str = Form(...)):
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

    async def commit_runtime_directory(self, proposal_id: str, data: str = Form(...)):
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

    async def reject_runtime_directory(self, proposal_id: str, data: str = Form('{}')):
        payload = json.loads(data)
        try:
            return self.scheduler.reject_runtime_directory(proposal_id, payload.get('reason', ''))
        except (RuntimeDirectoryError, TypeError, ValueError) as exc:
            raise self._translate_runtime_error(exc)

    async def count_task_leases(self, revision: int):
        try:
            return self.scheduler.task_lease_status(revision)
        except (RuntimeDirectoryError, TypeError, ValueError) as exc:
            raise self._translate_runtime_error(exc)

    async def retire_task_leases(self, data: str = Form(...)):
        payload = json.loads(data)
        try:
            return self.scheduler.retire_task_leases(
                payload.get('revision', payload.get('runtime_directory_revision')),
                payload.get('deadline'),
            )
        except (RuntimeDirectoryError, TypeError, ValueError) as exc:
            raise self._translate_runtime_error(exc)

    async def acquire_task_lease(self, data: str = Form(...)):
        payload = json.loads(data)
        revision = payload.get('revision', payload.get('runtime_directory_revision'))
        root_uuid = payload.get('root_uuid', payload.get('rootUUID'))
        try:
            return self.scheduler.acquire_task_lease(
                revision,
                root_uuid,
                payload.get('ttl_seconds', payload.get('ttlSeconds', 60.0)),
            )
        except TaskLeaseRetired as exc:
            return self._retired_task_lease(exc, root_uuid)
        except (RuntimeDirectoryError, TypeError, ValueError) as exc:
            raise self._translate_runtime_error(exc)

    async def renew_task_lease(self, data: str = Form(...)):
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

    async def release_task_lease(self, data: str = Form(...)):
        payload = json.loads(data)
        try:
            return self.scheduler.release_task_lease(
                payload.get('revision', payload.get('runtime_directory_revision')),
                payload.get('root_uuid', payload.get('rootUUID')),
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
                raise self._translate_runtime_error(exc)
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
                raise self._translate_runtime_error(exc)

        return {'plan': plan}

    async def generate_redeployment_plan(self, data: str = Form(...)):
        data = json.loads(data)
        plan = {}
        for source_data in data:
            source_id = source_data['source']['id']
            self.scheduler.register_schedule_table(source_id=source_id)
            try:
                source_plan = self.scheduler.get_redeployment_plan(source_id, source_data)
                self._merge_deployment_plan(
                    plan, source_plan, allowed_services=dag_services(source_data),
                )
            except (RuntimeDirectoryError, ValueError) as exc:
                raise self._translate_runtime_error(exc)

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
