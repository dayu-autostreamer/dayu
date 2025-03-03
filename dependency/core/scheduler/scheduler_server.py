import json

from fastapi import FastAPI, Form
from fastapi.routing import APIRoute
from starlette.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from core.lib.network import NetworkAPIMethod, NetworkAPIPath
from core.lib.content import Task

from .scheduler import Scheduler


class SchedulerServer:
    def __init__(self):
        self.app = FastAPI(routes=[
            APIRoute(NetworkAPIPath.SCHEDULER_SCHEDULE,
                     self.generate_schedule_plan,
                     response_class=JSONResponse,
                     methods=[NetworkAPIMethod.SCHEDULER_SCHEDULE]
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
            APIRoute(NetworkAPIPath.SCHEDULER_SELECT_SOURCE_NODE,
                     self.get_source_nodes_selection,
                     response_class=JSONResponse,
                     methods=[NetworkAPIMethod.SCHEDULER_SELECT_SOURCE_NODE]
                     ),
            APIRoute(),
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

    async def update_object_scenario(self, data: str = Form(...)):
        task = Task.deserialize(data)

        self.scheduler.update_scheduler_scenario(task)

    async def update_resource_state(self, data: str = Form(...)):
        data = json.loads(data)

        self.scheduler.register_resource_table(data['device'])
        self.scheduler.update_scheduler_resource(data)

    async def get_resource_state(self):
        return self.scheduler.get_scheduler_resource()

    # TODO
    async def get_source_nodes_selection(self, data: str = Form(...)):
        data = json.loads(data)
        return {'plan': None}

    # TODO
    async def get_initial_deployment_plan(self, data: str = Form(...)):
        data = json.loads(data)
        return {'plan': None}
