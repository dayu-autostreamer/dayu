import abc
import csv
import json
import os
import threading
import time

import numpy as np

from core.lib.common import ClassFactory, ClassType, LOGGER, FileOps, Context, ConfigLoader, TaskConstant
from core.lib.estimation import OverheadEstimator

from .base_agent import BaseAgent

__all__ = ("DeepVAAgent",)


@ClassFactory.register(ClassType.SCH_AGENT, alias="deepva")
class DeepVAAgent(BaseAgent, abc.ABC):
    """Plain DRL baseline for joint deployment/offloading decisions.

    DeepVA deliberately stays simpler than Hedger: it uses a pair-state MLP DQN
    with deployment and offloading heads, no graph encoder, no QK module, no PPO
    rollout machinery, and no latency guard.
    """

    TRAIN_FIELDS = (
        "step",
        "mode",
        "reward",
        "avg_delay",
        "latency_cost",
        "slo_violation",
        "queue_cost",
        "cloud_fraction",
        "change_rate",
        "memory_overage",
        "projection_count",
        "epsilon",
        "loss",
        "q_mean",
        "target_q_mean",
        "buffer_size",
        "replica_count",
        "unique_targets",
        "state_complexity_mean",
        "state_arrival_rate",
        "state_queue_mean",
        "state_queue_max",
        "state_runtime_mean",
        "state_runtime_confidence_mean",
        "state_model_flops_mean",
        "state_model_memory_mean",
        "state_gpu_flops_mean",
        "state_memory_capacity_mean",
    )

    DECISION_FIELDS = (
        "step",
        "mode",
        "deployment_plan",
        "offloading_plan",
        "replica_count",
        "unique_targets",
        "projection_count",
        "epsilon",
        "deploy_q_mean",
        "offload_q_mean",
        "deploy_q_max",
        "offload_q_max",
    )

    def __init__(
        self,
        system,
        agent_id: int,
        mode: str = "inference",
        model_dir: str = "model",
        load_model: bool = False,
        load_model_episode: int = 0,
        redeployment_interval: int = 60,
        reward_collection_window: int = 30,
        learning_rate: float = 0.001,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.08,
        epsilon_decay: float = 0.985,
        target_update_freq: int = 10,
        hidden_dim: int = 128,
        replay_buffer_size: int = 10000,
        batch_size: int = 32,
        update_interval: int = 1,
        update_after: int = 10,
        save_interval: int = 20,
        total_steps: int = 120,
        min_replicas_per_service: int = 1,
        max_replicas_per_service: int = 2,
        replica_score_threshold: float = 0.0,
        latency_slo_s: float = 3.0,
        latency_clip: float = 3.0,
        latency_weight: float = 0.45,
        slo_weight: float = 0.65,
        queue_weight: float = 0.10,
        cloud_weight: float = 0.05,
        change_weight: float = 0.03,
        memory_weight: float = 0.45,
        normalizers=None,
        record=None,
        configuration=None,
    ):
        super().__init__(system, agent_id)

        from .deepva import DQNAgent, ReplayBuffer, StateBuffer

        if configuration is None or isinstance(configuration, dict):
            self.configuration = configuration
        elif isinstance(configuration, str):
            self.configuration = ConfigLoader.load(Context.get_file_path(configuration))
        else:
            raise TypeError(f'Input "configuration" must be of type str or dict, get type {type(configuration)}')

        self.agent_id = agent_id
        self.system = system
        self.mode = str(mode)
        self.cloud_device = getattr(system, "cloud_device", "cloud")
        self.service_names = [str(s) for s in getattr(system, "service_list", [])]
        self.device_list = [str(d) for d in getattr(system, "device_list", [])]
        self.num_services = len(self.service_names)
        self.num_devices = len(self.device_list)
        self.device_service_limits = list(getattr(system, "device_service_limits", []))
        if self.num_services <= 0 or self.num_devices <= 0:
            raise ValueError("DeepVA requires non-empty `service_list` and `device_list` in config extraction.")
        if len(self.device_service_limits) != self.num_devices:
            raise ValueError("DeepVA `device_service_limits` length must match `device_list` length.")

        self.redeployment_interval = float(redeployment_interval)
        self.reward_collection_window = float(reward_collection_window)
        self.batch_size = int(batch_size)
        self.update_interval = int(update_interval)
        self.update_after = int(update_after)
        self.save_interval = int(save_interval)
        self.total_steps = int(total_steps)
        self.latency_slo_s = max(1e-6, float(latency_slo_s))
        self.latency_clip = max(1.0, float(latency_clip))
        self.reward_weights = {
            "latency": float(latency_weight),
            "slo": float(slo_weight),
            "queue": float(queue_weight),
            "cloud": float(cloud_weight),
            "change": float(change_weight),
            "memory": float(memory_weight),
        }

        self.record_cfg = record if isinstance(record, dict) else {}
        self.record_enabled = bool(self.record_cfg.get("enabled", True))

        self.state_buffer = StateBuffer(
            self.service_names,
            self.device_list,
            cloud_device=self.cloud_device,
            delay_window_size=int(self.record_cfg.get("delay_window_size", 30)),
            runtime_alpha=float(self.record_cfg.get("runtime_alpha", 0.25)),
            normalizers=normalizers,
        )

        self.dqn_agent = DQNAgent(
            num_services=self.num_services,
            num_devices=self.num_devices,
            pair_feature_dim=self.state_buffer.feature_dim,
            device_service_limits=self.device_service_limits,
            learning_rate=learning_rate,
            gamma=gamma,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay=epsilon_decay,
            target_update_freq=target_update_freq,
            hidden_dim=hidden_dim,
            min_replicas_per_service=min_replicas_per_service,
            max_replicas_per_service=max_replicas_per_service,
            replica_score_threshold=replica_score_threshold,
        )
        self.replay_buffer = ReplayBuffer(capacity=replay_buffer_size)

        self.model_dir = Context.get_file_path(os.path.join("scheduler/deepva", model_dir, f"agent_{self.agent_id}"))
        FileOps.create_directory(self.model_dir)
        self.record_dir = Context.get_file_path(os.path.join("scheduler/deepva", model_dir, f"records_agent_{self.agent_id}"))
        FileOps.create_directory(self.record_dir)
        self.train_file = os.path.join(self.record_dir, "deepva_train.csv")
        self.decision_file = os.path.join(self.record_dir, "deepva_decisions.csv")
        if self.record_enabled:
            FileOps.remove_file(self.train_file)
            FileOps.remove_file(self.decision_file)

        self.deployment_lock = threading.Lock()
        self.current_deployment_mask = self._initial_deployment_mask()
        self.current_offloading_targets = self._targets_from_mask(self.current_deployment_mask)
        self.prev_deployment_mask = self.current_deployment_mask.copy()
        self.state_buffer.update_deployment(self.current_deployment_mask, self.current_offloading_targets)
        self.latest_offloading_policy = self._offloading_plan_from_targets(self.current_offloading_targets)
        self.global_step = 0
        self.last_projection_count = 0

        if load_model:
            self.dqn_agent.load(self.model_dir, load_model_episode)

        self.overhead_estimator = OverheadEstimator("DeepVA", "scheduler/deepva", agent_id=self.agent_id)
        LOGGER.info(
            f"[DeepVA] Initialized mode={self.mode}, services={self.service_names}, "
            f"devices={self.device_list}, limits={self.device_service_limits}"
        )

    def _initial_deployment_mask(self):
        fixed_policy = getattr(self.initial_deployment_policy, "fixed_policy", None)
        mask = np.zeros((self.num_services, self.num_devices), dtype=np.float32)
        if isinstance(fixed_policy, dict):
            for service_name, devices in fixed_policy.items():
                s_idx = self._service_idx(service_name)
                if s_idx is None:
                    continue
                for device in self._normalize_devices(devices):
                    d_idx = self._device_idx(device)
                    if d_idx is not None:
                        mask[s_idx, d_idx] = 1.0
        if np.any(mask.sum(axis=1) <= 0):
            device_counts = mask.sum(axis=0).astype(np.int64)
            for service_idx in np.flatnonzero(mask.sum(axis=1) <= 0):
                d_idx = int(np.argmin(device_counts / np.maximum(self.device_service_limits, 1)))
                mask[service_idx, d_idx] = 1.0
                device_counts[d_idx] += 1
        return self._repair_deployment_mask(mask)[0]

    @staticmethod
    def _normalize_devices(devices):
        if devices is None:
            return []
        if isinstance(devices, str):
            return [devices]
        if isinstance(devices, (list, tuple, set, frozenset)):
            return [str(device) for device in devices]
        return []

    def _service_idx(self, service_name):
        try:
            return self.service_names.index(str(service_name))
        except ValueError:
            return None

    def _device_idx(self, device):
        try:
            return self.device_list.index(str(device))
        except ValueError:
            return None

    def get_current_state(self):
        while not self.state_buffer.is_ready():
            LOGGER.info("[DeepVA] Waiting for state buffer to be ready...")
            time.sleep(1)
        return self.state_buffer.get_state()

    def reset_env(self):
        return self.get_current_state()

    def step_env(self, action):
        deployment_mask = np.asarray(action["deployment_mask"], dtype=np.float32)
        offloading_targets = np.asarray(action["offloading_targets"], dtype=np.int64)
        deployment_mask, projection_count = self._repair_deployment_mask(deployment_mask, action.get("deploy_q"))
        offloading_targets = self._repair_offloading_targets(deployment_mask, offloading_targets, action.get("offload_q"))

        with self.deployment_lock:
            old_mask = self.current_deployment_mask.copy()
            self.prev_deployment_mask = old_mask
            self.current_deployment_mask = deployment_mask
            self.current_offloading_targets = offloading_targets
            self.latest_offloading_policy = self._offloading_plan_from_targets(offloading_targets)
            self.last_projection_count = projection_count
            self.state_buffer.update_deployment(deployment_mask, offloading_targets)

        self._record_decision(action, projection_count)
        LOGGER.info(f"[DeepVA] Waiting {self.reward_collection_window}s for reward feedback...")
        time.sleep(self.reward_collection_window)

        reward, reward_info = self.calculate_reward(old_mask, deployment_mask, offloading_targets, projection_count)
        remaining_time = self.redeployment_interval - self.reward_collection_window
        if remaining_time > 0:
            time.sleep(remaining_time)
        next_state = self.get_current_state()
        return next_state, reward, False, reward_info

    def _repair_deployment_mask(self, mask, deploy_q=None):
        mask = (np.asarray(mask, dtype=np.float32) > 0).astype(np.float32)
        projection_count = 0

        device_counts = mask.sum(axis=0).astype(np.int64)
        for service_idx in range(self.num_services):
            if mask[service_idx].sum() > 0:
                continue
            d_idx = int(np.argmin(device_counts / np.maximum(self.device_service_limits, 1)))
            mask[service_idx, d_idx] = 1.0
            device_counts[d_idx] += 1
            projection_count += 1

        for device_idx, limit in enumerate(self.device_service_limits):
            while device_counts[device_idx] > int(limit):
                services = np.flatnonzero(mask[:, device_idx] > 0)
                removable = [s for s in services if mask[s].sum() > 1]
                if not removable:
                    removable = list(services)
                if deploy_q is not None:
                    service_idx = min(removable, key=lambda s: float(deploy_q[s, device_idx]))
                else:
                    service_idx = int(removable[-1])
                target = self._least_loaded_device(device_counts, exclude=device_idx, service_mask=mask[service_idx])
                if target is None:
                    break
                mask[service_idx, device_idx] = 0.0
                mask[service_idx, target] = 1.0
                device_counts[device_idx] -= 1
                device_counts[target] += 1
                projection_count += 1

        memory_projection = self._repair_memory(mask, deploy_q)
        projection_count += memory_projection
        return mask, projection_count

    def _repair_memory(self, mask, deploy_q=None):
        projection_count = 0
        with self.state_buffer.lock:
            model_memory = self.state_buffer.model_memory.copy()
            capacity = self.state_buffer.memory_capacity * (1.0 - np.clip(self.state_buffer.memory_usage, 0.0, 1.0))
        if not np.any(model_memory > 0) or not np.any(capacity > 0):
            return projection_count

        used = np.matmul(mask.T, model_memory)
        for device_idx in range(self.num_devices):
            while used[device_idx] > capacity[device_idx] > 0:
                services = np.flatnonzero(mask[:, device_idx] > 0)
                removable = [s for s in services if mask[s].sum() > 1]
                if not removable:
                    break
                if deploy_q is not None:
                    service_idx = min(removable, key=lambda s: float(deploy_q[s, device_idx]))
                else:
                    service_idx = max(removable, key=lambda s: float(model_memory[s]))
                mask[service_idx, device_idx] = 0.0
                used[device_idx] -= model_memory[service_idx]
                projection_count += 1
        return projection_count

    def _least_loaded_device(self, device_counts, exclude=None, service_mask=None):
        best_device = None
        best_load = float("inf")
        for device_idx, limit in enumerate(self.device_service_limits):
            if exclude is not None and device_idx == exclude:
                continue
            if service_mask is not None and service_mask[device_idx] > 0:
                continue
            if device_counts[device_idx] >= int(limit):
                continue
            load = device_counts[device_idx] / max(1, int(limit))
            if load < best_load:
                best_load = load
                best_device = device_idx
        return best_device

    def _repair_offloading_targets(self, deployment_mask, targets, offload_q=None):
        targets = np.asarray(targets, dtype=np.int64).copy()
        for service_idx in range(self.num_services):
            candidates = np.flatnonzero(deployment_mask[service_idx] > 0)
            if candidates.size == 0:
                candidates = np.arange(self.num_devices)
            if targets[service_idx] not in candidates:
                if offload_q is not None:
                    targets[service_idx] = int(candidates[np.argmax(offload_q[service_idx, candidates])])
                else:
                    targets[service_idx] = int(candidates[0])
        return targets

    def _targets_from_mask(self, mask):
        targets = np.zeros(self.num_services, dtype=np.int64)
        for service_idx in range(self.num_services):
            candidates = np.flatnonzero(mask[service_idx] > 0)
            targets[service_idx] = int(candidates[0]) if candidates.size else 0
        return targets

    def calculate_reward(self, old_mask, new_mask, offloading_targets, projection_count):
        avg_delay = self.state_buffer.get_average_delay()
        latency_cost = min(self.latency_clip, avg_delay / self.latency_slo_s) if avg_delay > 0 else 0.0
        slo_violation = max(0.0, avg_delay / self.latency_slo_s - 1.0) if avg_delay > 0 else 0.0
        queue_cost = min(3.0, self.state_buffer.get_selected_queue_mean(offloading_targets) / 10.0)
        cloud_fraction = self._cloud_fraction(offloading_targets)
        change_rate = float(np.mean(np.abs(new_mask - old_mask))) if old_mask.size else 0.0
        memory_overage = min(3.0, self.state_buffer.get_memory_overage(new_mask))

        reward = -(
            self.reward_weights["latency"] * latency_cost
            + self.reward_weights["slo"] * slo_violation
            + self.reward_weights["queue"] * queue_cost
            + self.reward_weights["cloud"] * cloud_fraction
            + self.reward_weights["change"] * change_rate
            + self.reward_weights["memory"] * memory_overage
        )
        info = {
            "avg_delay": avg_delay,
            "latency_cost": latency_cost,
            "slo_violation": slo_violation,
            "queue_cost": queue_cost,
            "cloud_fraction": cloud_fraction,
            "change_rate": change_rate,
            "memory_overage": memory_overage,
            "projection_count": projection_count,
        }
        return float(np.clip(reward, -10.0, 1.0)), info

    def _cloud_fraction(self, targets):
        if self.num_services <= 0:
            return 0.0
        count = 0
        for target in targets:
            device = self.device_list[int(target)]
            if str(device) == str(self.cloud_device) or "cloud" in str(device).lower():
                count += 1
        return float(count / self.num_services)

    def train_dqn_agent(self):
        LOGGER.info("[DeepVA] Start DQN training.")
        state = self.reset_env()
        for step in range(self.total_steps):
            self.global_step = step
            with self.overhead_estimator:
                action = self.dqn_agent.select_action(state, deterministic=False)
            next_state, reward, done, info = self.step_env(action)
            self.replay_buffer.add(
                state,
                self.current_deployment_mask,
                self.current_offloading_targets,
                reward,
                next_state,
                done,
            )

            train_stats = {"loss": 0.0, "epsilon": self.dqn_agent.epsilon}
            if step >= self.update_after and step % self.update_interval == 0:
                train_stats = self.dqn_agent.train(self.replay_buffer, self.batch_size)

            self._record_train(step, reward, info, train_stats)
            state = next_state
            if step > 0 and step % self.save_interval == 0:
                self.dqn_agent.save(self.model_dir, step)
                LOGGER.info(f"[DeepVA] Saved model at step {step}.")

        self.dqn_agent.save(self.model_dir, self.total_steps)
        LOGGER.info("[DeepVA] DQN training completed.")

    def inference_dqn_agent(self):
        LOGGER.info("[DeepVA] Start inference.")
        state = self.reset_env()
        step = 0
        while True:
            self.global_step = step
            with self.overhead_estimator:
                action = self.dqn_agent.select_action(state, deterministic=True)
            next_state, reward, _, info = self.step_env(action)
            self._record_train(step, reward, info, {"loss": 0.0, "epsilon": self.dqn_agent.epsilon})
            state = next_state
            step += 1

    def update_scenario(self, scenario):
        try:
            self.state_buffer.update_scenario(scenario)
        except Exception as exc:
            LOGGER.warning(f"[DeepVA] Error updating scenario: {exc}")

    def update_resource(self, device, resource):
        try:
            self.state_buffer.update_resource(device, resource)
        except Exception as exc:
            LOGGER.warning(f"[DeepVA] Error updating resource for {device}: {exc}")

    def update_policy(self, policy):
        pass

    def update_task(self, task):
        try:
            dag = task.get_dag()
            if dag is None:
                return
            metadata = task.get_metadata() or {}
            buffer_size = max(1.0, float(metadata.get("buffer_size", 1.0)))
            try:
                self.state_buffer.add_task_delay(task.get_real_end_to_end_time() / buffer_size)
            except Exception:
                pass
            for service_name in dag.nodes:
                if service_name in (TaskConstant.START.value, TaskConstant.END.value):
                    continue
                if service_name not in self.service_names:
                    continue
                service = task.get_service(service_name)
                complexity = self._extract_complexity(service.get_scenario_data())
                self.state_buffer.add_task_feedback(
                    service_name,
                    service.get_execute_device(),
                    service.get_real_execute_time(),
                    complexity=complexity,
                )
        except Exception as exc:
            LOGGER.warning(f"[DeepVA] Error updating task feedback: {exc}")

    @staticmethod
    def _extract_complexity(scenario):
        if not isinstance(scenario, dict):
            return 0.0
        obj_num = scenario.get("obj_num")
        if obj_num is None:
            return 0.0
        try:
            arr = np.asarray(obj_num, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            return 0.0
        return float(np.mean(arr)) if arr.size else 0.0

    def get_schedule_plan(self, info):
        with self.deployment_lock:
            targets = self.current_offloading_targets.copy()

        dag = info.get("dag", {})
        if not dag:
            return None

        source_device = info.get("source_device")
        offloading_policy = {}
        for service_name in dag:
            if service_name == TaskConstant.START.value:
                if source_device and "service" in dag[service_name]:
                    dag[service_name]["service"]["execute_device"] = source_device
                continue
            if service_name == TaskConstant.END.value:
                if "service" in dag[service_name]:
                    dag[service_name]["service"]["execute_device"] = self.cloud_device
                continue
            s_idx = self._service_idx(service_name)
            if s_idx is None:
                continue
            target_idx = int(targets[s_idx])
            device_name = self.device_list[target_idx]
            if "service" in dag[service_name]:
                dag[service_name]["service"]["execute_device"] = device_name
            offloading_policy[service_name] = device_name

        self.latest_offloading_policy = dict(offloading_policy)
        policy = {"dag": dag}
        if self.configuration is not None:
            policy.update(self.configuration.copy())
        return policy

    def _deployment_plan_from_mask(self, mask):
        plan = {}
        for s_idx, service_name in enumerate(self.service_names):
            devices = [self.device_list[d_idx] for d_idx in np.flatnonzero(mask[s_idx] > 0)]
            plan[service_name] = devices or [self.device_list[int(np.argmax(mask[s_idx]))]]
        return plan

    def _offloading_plan_from_targets(self, targets):
        plan = {}
        for s_idx, service_name in enumerate(self.service_names):
            target_idx = int(targets[s_idx])
            if 0 <= target_idx < self.num_devices:
                plan[service_name] = self.device_list[target_idx]
        return plan

    def get_current_deployment(self):
        with self.deployment_lock:
            return self._deployment_plan_from_mask(self.current_deployment_mask)

    def get_latest_offloading_policy(self):
        with self.deployment_lock:
            return dict(self.latest_offloading_policy)

    def get_schedule_overhead(self):
        return self.overhead_estimator.get_latest_overhead()

    def _record_decision(self, action, projection_count):
        if not self.record_enabled:
            return
        deploy_q = action.get("deploy_q")
        offload_q = action.get("offload_q")
        row = {
            "step": self.global_step,
            "mode": self.mode,
            "deployment_plan": json.dumps(self._deployment_plan_from_mask(self.current_deployment_mask), sort_keys=True),
            "offloading_plan": json.dumps(self._offloading_plan_from_targets(self.current_offloading_targets), sort_keys=True),
            "replica_count": int(self.current_deployment_mask.sum()),
            "unique_targets": int(len(set(map(int, self.current_offloading_targets.tolist())))),
            "projection_count": int(projection_count),
            "epsilon": float(self.dqn_agent.epsilon),
            "deploy_q_mean": self._array_stat(deploy_q, np.mean),
            "offload_q_mean": self._array_stat(offload_q, np.mean),
            "deploy_q_max": self._array_stat(deploy_q, np.max),
            "offload_q_max": self._array_stat(offload_q, np.max),
        }
        self._append_csv(self.decision_file, self.DECISION_FIELDS, row)

    def _record_train(self, step, reward, info, train_stats):
        if not self.record_enabled:
            return
        summary = self.state_buffer.get_summary()
        row = {
            "step": step,
            "mode": self.mode,
            "reward": float(reward),
            "avg_delay": info.get("avg_delay", 0.0),
            "latency_cost": info.get("latency_cost", 0.0),
            "slo_violation": info.get("slo_violation", 0.0),
            "queue_cost": info.get("queue_cost", 0.0),
            "cloud_fraction": info.get("cloud_fraction", 0.0),
            "change_rate": info.get("change_rate", 0.0),
            "memory_overage": info.get("memory_overage", 0.0),
            "projection_count": info.get("projection_count", 0),
            "epsilon": train_stats.get("epsilon", self.dqn_agent.epsilon),
            "loss": train_stats.get("loss", 0.0),
            "q_mean": train_stats.get("q_mean", 0.0),
            "target_q_mean": train_stats.get("target_q_mean", 0.0),
            "buffer_size": len(self.replay_buffer),
            "replica_count": int(self.current_deployment_mask.sum()),
            "unique_targets": int(len(set(map(int, self.current_offloading_targets.tolist())))),
            "state_complexity_mean": summary["complexity_mean"],
            "state_arrival_rate": summary["arrival_rate"],
            "state_queue_mean": summary["queue_mean"],
            "state_queue_max": summary["queue_max"],
            "state_runtime_mean": summary["runtime_mean"],
            "state_runtime_confidence_mean": summary["runtime_confidence_mean"],
            "state_model_flops_mean": summary["model_flops_mean"],
            "state_model_memory_mean": summary["model_memory_mean"],
            "state_gpu_flops_mean": summary["gpu_flops_mean"],
            "state_memory_capacity_mean": summary["memory_capacity_mean"],
        }
        self._append_csv(self.train_file, self.TRAIN_FIELDS, row)

    @staticmethod
    def _array_stat(value, fn):
        if value is None:
            return 0.0
        arr = np.asarray(value, dtype=float)
        return float(fn(arr)) if arr.size else 0.0

    @staticmethod
    def _append_csv(path, fieldnames, row):
        exists = os.path.exists(path)
        with open(path, "a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            if not exists:
                writer.writeheader()
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    def run(self):
        if self.mode == "train":
            self.train_dqn_agent()
        elif self.mode == "inference":
            self.inference_dqn_agent()
        else:
            raise ValueError(f'Invalid DeepVA mode: {self.mode}. Expected "train" or "inference".')
