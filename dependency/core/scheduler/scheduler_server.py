import json

from fastapi import FastAPI, Form
from fastapi.routing import APIRoute
from starlette.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from core.lib.network import NetworkAPIMethod, NetworkAPIPath
from core.lib.content import Task
from core.lib.common import LOGGER

from .scheduler import Scheduler


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
            APIRoute(NetworkAPIPath.SCHEDULER_SELECT_SOURCE_NODE,
                     self.generate_source_nodes_selection_plan,
                     response_class=JSONResponse,
                     methods=[NetworkAPIMethod.SCHEDULER_SELECT_SOURCE_NODE]
                     ),
            APIRoute(NetworkAPIPath.SCHEDULER_INITIAL_DEPLOYMENT,
                     self.generate_initial_deployment_plan,
                     response_class=JSONResponse,
                     methods=[NetworkAPIMethod.SCHEDULER_INITIAL_DEPLOYMENT]),
            APIRoute(NetworkAPIPath.SCHEDULER_REDEPLOYMENT,
                     self.generate_redeployment_plan,
                     response_class=JSONResponse,
                     methods=[NetworkAPIMethod.SCHEDULER_REDEPLOYMENT]),
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

        return {'plan': plan}

    async def get_schedule_overhead(self):
        return self.scheduler.get_schedule_overhead()

    async def update_object_scenario(self, data: str = Form(...)):
        task = Task.deserialize(data)

        self.scheduler.update_scheduler_scenario(task)

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
            source_plan = self.scheduler.get_source_node_selection_plan(source_id, source_data)
            plan[source_id] = source_plan

        LOGGER.info(f'[Source Node Selection] (all sources) Selection policy: {plan}')
        return {'plan': plan}

    async def generate_initial_deployment_plan(self, data: str = Form(...)):
        data = json.loads(data)

        plan = {}
        for source_data in data:
            source_id = source_data['source']['id']
            self.scheduler.register_schedule_table(source_id=source_id)
            source_plan = self.scheduler.get_initial_deployment_plan(source_id, source_data)
            plan.update(
                {node: list(set(plan[node] + source_plan[node])) if node in plan else source_plan[node]
                 for node in source_plan}
            )

        LOGGER.info(f'[Initial Deployment] (all sources) Deploy policy: {plan}')
        return {'plan': plan}

    async def generate_redeployment_plan(self, data: str = Form(...)):
        data = json.loads(data)
        plan = {}
        for source_data in data:
            source_id = source_data['source']['id']
            self.scheduler.register_schedule_table(source_id=source_id)
            source_plan = self.scheduler.get_redeployment_plan(source_id, source_data)
            plan.update(
                {node: list(set(plan[node] + source_plan[node])) if node in plan else source_plan[node]
                 for node in source_plan}
            )

        LOGGER.info(f'[Redeployment] (all sources) Deploy policy: {plan}')
        return {'plan': plan}
