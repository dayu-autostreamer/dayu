"""STEADY scheduling over a detection pipeline and the active deployment."""

import copy
from collections import deque
from datetime import datetime

from core.lib.common import ClassFactory, ClassType, Context
from core.lib.content import Task
from core.lib.scheduling.live_state import active_deployment_for_dag, require_active_plan
from core.lib.scheduling.pipeline import apply_pipeline_partition, pipeline_entries, pipeline_partition_index

from ..base_agent import BaseAgent
from .accuracy_calculation import AccuracyCalculation
from .configuration import constraint_combinations, constraint_values, normalize_parameters
from .context_record import ContextRecord
from .overall_scheduler import OverallScheduler

__all__ = ("SteadyAgent",)


@ClassFactory.register(ClassType.SCH_AGENT, alias="steady")
class SteadyAgent(BaseAgent):
    scheduler_mode = "steady"

    def __init__(self, system, agent_id: int, sch_param: dict, steady_param: dict):
        super().__init__(system, agent_id)
        self.agent_id = agent_id
        self.cur_resource_table = {}
        self.cur_scenario = {}
        self.cur_policy = {}
        self.cur_task = None
        self.edge_device = None
        self.service_names = None
        self.fps_list = list(system.fps_list)
        self.resolution_list = list(system.resolution_list)
        self.buffer_size_list = [size for size in system.buffer_size_list if size >= 2]
        if not self.fps_list or not self.resolution_list or not self.buffer_size_list:
            raise ValueError("STEADY requires FPS, resolution and buffer sizes of at least two frames")
        self.edge_serv_num_list = None
        self.schedule_plan_num = 0
        self.init_param = normalize_parameters(steady_param)
        self.all_cons_info_comb_list = constraint_combinations(self.init_param)
        self.cons_info_comb_idx = 0
        self.if_stop_record_in_single_cycle = bool(sch_param.get("if_stop_record_in_single_cycle", False))
        self.unit_logic_frame_num_max = self.init_param.get(
            "unit_logic_frame_num_max", sch_param.get("stop_max_frame_num", 1000)
        )
        if self.unit_logic_frame_num_max <= 0:
            raise ValueError("unit_logic_frame_num_max must be positive")
        self.unit_processed_frame_num = 0
        self.if_keep_record = True
        self.record_path_prefix = self._resolve_path(sch_param.get("record_path"))
        self.steady_record_path_prefix = self._resolve_path(self.init_param.get("steady_record_path"))
        self.correct_record_path_prefix = self._resolve_path(self.init_param.get("correct_record_path"))
        self.path_suffix = f"{agent_id}-{datetime.now():%Y-%m-%d-%H-%M-%S-%f}.json"
        self.init_param["kb_path"] = Context.get_file_path(self.init_param["kb_path"])
        self.overall_scheduler = None
        self.task_history_deque = deque(maxlen=10)
        self.acc_sample_interval = self.init_param["acc_sample_interval"]

    @staticmethod
    def _resolve_path(path):
        return Context.get_file_path(path) if path else None

    def _record_path(self, prefix, source_id, source_device):
        return f"{prefix}-source_id-{source_id}-{source_device}-{self.path_suffix}" if prefix else None

    def _constraints(self, adjusted=False):
        index = min(self.cons_info_comb_idx, len(self.all_cons_info_comb_list) - 1)
        return constraint_values(self.all_cons_info_comb_list[index], adjusted=adjusted)

    def run(self):
        # Scheduling is request-driven; the search worker starts with the first plan.
        pass

    def stop(self):
        if self.overall_scheduler is not None:
            self.overall_scheduler.stop()

    def update_scenario(self, scenario):
        self.cur_scenario = copy.deepcopy(scenario)

    def update_resource(self, device, resource):
        self.cur_resource_table[device] = copy.deepcopy(resource)

    def update_policy(self, policy):
        self.cur_policy = copy.deepcopy(policy)

    def update_task(self, task: Task):
        if task is None:
            return
        self.cur_task = copy.deepcopy(task)
        self.update_record(self.cur_task)
        if self.overall_scheduler is not None:
            context = self.get_context_info_from_task(self.cur_task)
            configuration, feedback = self.get_conf_info_task_info_from_task(self.cur_task)
            self.overall_scheduler.update_scheduler(context, configuration, feedback)

    def update_record(self, task):
        if self.if_keep_record and self.record_path_prefix:
            record = ContextRecord(
                task=task, resource_table=copy.deepcopy(self.cur_resource_table), cons_table=self._constraints()
            )
            ContextRecord.write_record(
                record, self._record_path(self.record_path_prefix, task.get_source_id(), task.get_source_device())
            )
        if not self.if_keep_record or not self.if_stop_record_in_single_cycle:
            return
        metadata, raw_metadata = task.get_metadata(), task.get_raw_metadata()
        self.unit_processed_frame_num += metadata["buffer_size"] * raw_metadata["fps"] / metadata["fps"]
        if self.unit_processed_frame_num < self.unit_logic_frame_num_max:
            return
        self.cons_info_comb_idx += 1
        self.unit_processed_frame_num = 0
        if self.cons_info_comb_idx >= len(self.all_cons_info_comb_list):
            self.if_keep_record = False
        elif self.overall_scheduler is not None:
            self.overall_scheduler.update_constraints(**self._constraints(adjusted=True))

    def _create_scheduler(self, info):
        ranges = {
            "fps": self.fps_list,
            "resolution": self.resolution_list,
            "buffer_size": self.buffer_size_list,
            "edge_serv_num": self.edge_serv_num_list,
        }
        for knob, values in ranges.items():
            if self.init_param["default_policy"][knob] not in values:
                raise ValueError(f"default_policy.{knob} must be in the configured search range")
        names = (
            "corrector_param",
            "queue_param",
            "default_policy",
            "history_length",
            "stop_threshold",
            "macro_update_interval",
            "context_analyze_type",
            "coeff_info",
            "cluster_threshold",
            "if_online_train",
        )
        options = {name: self.init_param[name] for name in names}
        options.update(self._constraints(adjusted=True))
        return OverallScheduler(
            kb_path=self.init_param["kb_path"],
            service_name_pipeline=self.service_names,
            knob_value_range_dict=ranges,
            raw_meta_data=info["meta_data"],
            context_names=("band_Mbps", "obj_num", "obj_size_norm", "obj_speed"),
            steady_record_path=self._record_path(self.steady_record_path_prefix, info["source_id"], self.edge_device),
            correct_record_path=self._record_path(self.correct_record_path_prefix, info["source_id"], self.edge_device),
            schedule_type=self.scheduler_mode,
            feedback_weight=self.init_param.get("feedback_weight"),
            **options,
        )

    def get_schedule_plan(self, info):
        services = [entry["service_name"] for entry in pipeline_entries(info["dag"])[:-1]]
        if not 1 <= len(services) <= 2 or "detection" not in services[0]:
            raise ValueError("STEADY supports detection followed by at most one classification stage")
        if len(services) == 2 and "classification" not in services[1]:
            raise ValueError("STEADY's second stage must be a classification service")
        if self.service_names is None:
            self.service_names = services
            self.edge_device = info["source_device"]
            self.edge_serv_num_list = list(range(len(services) + 1))
        elif services != self.service_names or info["source_device"] != self.edge_device:
            raise ValueError("STEADY does not support changing the pipeline topology or source device at runtime")
        if self.overall_scheduler is None:
            self.overall_scheduler = self._create_scheduler(info)
        self.schedule_plan_num += 1
        if self.schedule_plan_num % self.acc_sample_interval == 0:
            policy = {"fps": 30, "resolution": "1080p", "buffer_size": 2, "edge_serv_num": 0}
        else:
            task = self.cur_task
            configuration = self.get_conf_info_from_task(task) if task is not None else None
            context = self.get_context_info_from_task(task) if task is not None else None
            delay = self.get_delay_from_task(task) if task is not None else None
            accuracy = None
            if context is not None:
                accuracy = self.overall_scheduler.macro_search.knowledge_base.performance_predictor.acc_pre(
                    context_info=context, conf_info=configuration, if_correct=True
                )
            policy = self.overall_scheduler.get_schedule_plan(
                cur_task_id=task.get_task_id() if task is not None else None,
                cur_policy=configuration,
                context_info=context,
                real_time_delay=delay,
                real_time_acc=accuracy,
            )
        dag = apply_pipeline_partition(info["dag"], policy["edge_serv_num"], self.edge_device, self.cloud_device)
        _, deployment = active_deployment_for_dag(self.system, dag)
        require_active_plan({name: dag[name]["service"]["execute_device"] for name in services}, deployment)
        return {
            "fps": policy["fps"],
            "resolution": policy["resolution"],
            "buffer_size": policy["buffer_size"],
            "encoding": "mp4v",
            "dag": dag,
        }

    def _services(self, task):
        dag = task.get_dag()
        return [
            dag.get_node(entry["service_name"]).service
            for entry in pipeline_entries(task.get_dag_deployment_info())[:-1]
        ]

    def get_delay_from_task(self, task):
        services = self._services(task)
        transmission = next(
            (service.get_transmit_time() for service in services if service.get_execute_device() == self.cloud_device),
            0,
        )
        metadata = task.get_metadata()
        delay = (sum(service.get_execute_time() for service in services) + transmission) / metadata["buffer_size"]
        return delay * metadata["fps"] / task.get_raw_metadata()["fps"]

    def get_context_info_from_task(self, task):
        bandwidth = self.cur_resource_table.get(self.edge_device, {}).get("available_bandwidth")
        scenario = task.get_first_scenario_data() or {}
        if bandwidth is None or not all(key in scenario for key in ("obj_size", "obj_num", "obj_velocity")):
            return None
        sizes, counts = scenario["obj_size"], scenario["obj_num"]
        return {
            "band_Mbps": bandwidth,
            "obj_size_norm": sum(sizes) / len(sizes) if sizes else 0,
            "obj_num": sum(counts) / len(counts) if counts else 0,
            "obj_speed": scenario["obj_velocity"],
        }

    def get_conf_info_task_info_from_task(self, task):
        metadata = task.get_metadata()
        if self.task_history_deque and metadata["resolution"] == "1080p" and metadata["fps"] == 30:
            previous = self.task_history_deque[-1]
            if previous.get_metadata()["resolution"] != "1080p":
                accuracy = AccuracyCalculation.get_real_acc(previous, task)
                self.task_history_deque.append(task)
                feedback = {"task_id": task.get_task_id()}
                if accuracy is not None:
                    feedback["real_acc_reso"] = accuracy
                return self.get_conf_info_from_task(previous), feedback
        self.task_history_deque.append(task)
        return self.get_conf_info_from_task(task), self.get_task_info_from_task(task)

    def get_conf_info_from_task(self, task):
        metadata = task.get_metadata()
        configuration = {name: metadata[name] for name in ("resolution", "fps", "buffer_size")}
        configuration["edge_serv_num"] = pipeline_partition_index(
            task.get_dag_deployment_info(), self.edge_device, self.cloud_device
        )
        return configuration

    def get_task_info_from_task(self, task):
        services = self._services(task)
        size = task.get_metadata()["buffer_size"]
        feedback = {"task_id": task.get_task_id()}
        for name, service in zip(("detect", "classify"), services):
            execution = service.get_real_execute_time() / size
            feedback[f"real_exe_{name}"] = execution
            feedback[f"{name}_wait_delay"] = service.get_execute_time() / size - execution
        feedback["real_trans"] = next(
            (
                service.get_transmit_time() / size
                for service in services
                if service.get_execute_device() == self.cloud_device
            ),
            0,
        )
        return feedback
