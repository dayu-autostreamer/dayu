import threading
import types

import pytest
import torch

from core.lib.algorithms.shared.hedger.hedger import Hedger
from core.lib.algorithms.shared.hedger.hedger import HedgerCheckpointLoadCfg
from core.lib.algorithms.shared.hedger.hedger import HedgerTrainingStageCfg
from core.lib.algorithms.shared.hedger.hedger import HedgerCheckpointSaveCfg
from core.lib.algorithms.shared.hedger.hedger import HedgerCheckpointCfg
from core.lib.algorithms.shared.hedger.hedger import HedgerTrainingCfg
from core.lib.algorithms.schedule_agent.hedger_agent import HedgerAgent
from core.lib.algorithms.shared.hedger.hedger_config import (
    DeploymentConstraintCfg,
    OffloadingConstraintCfg,
    LogicalTopology,
    PhysicalTopology,
    from_partial_dict,
)
from core.lib.algorithms.shared.hedger.ppo_agent import (
    HedgerDeploymentPPO,
    HedgerOffloadingPPO,
)
from core.lib.algorithms.shared.hedger.state_buffer import BufferWaitCfg, StateBuffer
from core.lib.algorithms.schedule_initial_deployment_policy.hedger_initial_deployment_policy import (
    HedgerInitialDeploymentPolicy,
)
from core.lib.content.task import Task


class DummyEncoder:
    def __init__(self, service_emb: torch.Tensor, device_emb: torch.Tensor):
        self.service_emb = service_emb
        self.device_emb = device_emb

    def parameters(self):
        return []

    def encode(self, logic_edge_index, logic_feats, phys_edge_index, phys_feats):
        return self.service_emb, self.device_emb


class TinyCheckpointAgent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(1, 1)
        self.actor_opt = torch.optim.Adam(self.layer.parameters(), lr=1e-3)
        self.critic_opt = torch.optim.Adam(self.layer.parameters(), lr=1e-3)


def build_test_topologies():
    dag = Task.extract_dag_from_dict(
        {
            "svc-a": {"service": {"service_name": "svc-a"}, "next_nodes": ["svc-b"]},
            "svc-b": {"service": {"service_name": "svc-b"}, "next_nodes": []},
        }
    )
    logical_topology = LogicalTopology(dag)
    physical_topology = PhysicalTopology(["edge-a", "edge-b"], "edge-a")
    return logical_topology, physical_topology


def build_training_cfg(stage: str, total_updates: int = 20) -> HedgerTrainingCfg:
    return HedgerTrainingCfg(
        stage=stage,
        total_updates=total_updates,
        ppo_epochs=4,
        deployment_rollout_len=8,
        offloading_rollout_len=32,
        deployment_batch_size=4,
        offloading_batch_size=16,
    )


def build_checkpoint_cfg(root_dir: str, *, load=None, save=None) -> HedgerCheckpointCfg:
    return HedgerCheckpointCfg(
        root_dir=root_dir,
        load=load or HedgerCheckpointLoadCfg(),
        save=save or HedgerCheckpointSaveCfg(interval_updates=20),
    )


@pytest.mark.unit
def test_from_partial_dict_ignores_removed_hedger_constraint_keys():
    off_cfg = from_partial_dict(
        OffloadingConstraintCfg,
        {
            "allow_stay": False,
            "forbid_return": True,
            "cloud_sticky": False,
            "use_monotone_metric": True,
            "metric_non_decreasing": False,
            "penalty_switch": 1.5,
            "penalty_relax": 2.5,
        },
    )
    dep_cfg = from_partial_dict(
        DeploymentConstraintCfg,
        {
            "enforce_capacity": False,
            "min_edge_replicas": 3,
            "penalty_capacity_relax": 4.0,
        },
    )

    assert isinstance(off_cfg, OffloadingConstraintCfg)
    assert off_cfg.penalty_switch == pytest.approx(1.5)
    assert off_cfg.penalty_relax == pytest.approx(2.5)
    assert not hasattr(off_cfg, "allow_stay")
    assert not hasattr(off_cfg, "forbid_return")
    assert not hasattr(off_cfg, "cloud_sticky")
    assert not hasattr(off_cfg, "use_monotone_metric")
    assert not hasattr(off_cfg, "metric_non_decreasing")

    assert isinstance(dep_cfg, DeploymentConstraintCfg)
    assert dep_cfg.penalty_capacity_relax == pytest.approx(4.0)
    assert not hasattr(dep_cfg, "enforce_capacity")
    assert not hasattr(dep_cfg, "min_edge_replicas")


@pytest.mark.unit
def test_deployment_policy_always_projects_to_capacity(monkeypatch):
    encoder = DummyEncoder(
        service_emb=torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32),
        device_emb=torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32),
    )
    agent = HedgerDeploymentPPO(
        encoder=encoder,
        d_model=2,
        update_encoder=False,
        cloud_node_idx=1,
        constraint_cfg=from_partial_dict(DeploymentConstraintCfg, {"enforce_capacity": False}),
    )

    def fake_forward(self, h_s, h_p, mask=None):
        probs = torch.tensor([[0.9, 0.05], [0.8, 0.05]], dtype=torch.float32, device=h_s.device)
        if mask is not None:
            probs = torch.where(mask, probs, torch.zeros_like(probs))
        return probs

    monkeypatch.setattr(agent.actor, "forward", types.MethodType(fake_forward, agent.actor))
    monkeypatch.setattr(
        torch,
        "rand_like",
        lambda x: torch.full_like(x, 0.1),
    )

    logic_edge_index = torch.empty((2, 0), dtype=torch.long)
    phys_edge_index = torch.empty((2, 0), dtype=torch.long)
    logic_feats = {"model_mem": torch.tensor([1.0, 1.0], dtype=torch.float32)}
    phys_feats = {
        "mem_capacity": torch.tensor([1.0, 100.0], dtype=torch.float32),
        "mem_util_seq": torch.zeros((2, 1), dtype=torch.float32),
    }

    deploy_mask, _, _, _, aux = agent.policy(
        logic_edge_index=logic_edge_index,
        logic_feats=logic_feats,
        phys_edge_index=phys_edge_index,
        phys_feats=phys_feats,
    )

    assert deploy_mask[:, 1].tolist() == [True, True]
    assert deploy_mask[:, 0].tolist() == [True, False]
    assert aux["capacity_relax_cnt"] == 1


@pytest.mark.unit
def test_offloading_policy_samples_then_promotes_descendants_to_cloud():
    encoder = DummyEncoder(
        service_emb=torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32),
        device_emb=torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32),
    )
    agent = HedgerOffloadingPPO(
        encoder=encoder,
        d_model=2,
        update_encoder=False,
        source_node_idx=0,
        cloud_node_idx=1,
        constraint_cfg=OffloadingConstraintCfg(penalty_relax=2.0),
    )
    agent._adapt_embeddings = lambda h_s, h_p: (h_s, h_p)

    def fake_forward(self, h_s, h_p, mask=None):
        service_id = int(round(float(h_s[0, 0].item())))
        if service_id == 0:
            probs = torch.tensor([0.0, 1.0], dtype=torch.float32, device=h_s.device)
        else:
            probs = torch.tensor([1.0, 0.0], dtype=torch.float32, device=h_s.device)
        if mask is not None:
            probs = torch.where(mask[0], probs, torch.zeros_like(probs))
        probs = probs / probs.sum()
        return probs.unsqueeze(0)

    agent.actor.forward = types.MethodType(fake_forward, agent.actor)

    logic_edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    phys_edge_index = torch.empty((2, 0), dtype=torch.long)
    static_mask = torch.tensor([[True, True], [True, True]], dtype=torch.bool)

    actions, _, _, _, aux = agent.policy(
        logic_edge_index=logic_edge_index,
        logic_feats={},
        phys_edge_index=phys_edge_index,
        phys_feats={},
        static_mask=static_mask,
    )

    assert actions.tolist() == [1, 1]
    assert aux["correction_cnt"] == 1
    assert aux["aux_cost"] == pytest.approx(2.0)


@pytest.mark.unit
def test_logical_topology_excludes_start_and_end_from_sizes():
    logical_topology, _ = build_test_topologies()

    assert len(logical_topology) == 2
    assert logical_topology.node_num == 2
    assert logical_topology.service_list == ["svc-a", "svc-b"]


@pytest.mark.unit
def test_state_buffer_supports_incremental_service_and_device_updates():
    logical_topology, physical_topology = build_test_topologies()
    cloud_name = physical_topology[physical_topology.cloud_idx]
    buffer = StateBuffer(
        16,
        logical_topology=logical_topology,
        physical_topology=physical_topology,
        fixed_lan_bandwidth_mbps=100.0,
    )

    buffer.add_model_flops("svc-a", 10.0)
    buffer.add_model_flops("svc-b", 20.0)
    buffer.add_model_memory("svc-a", 1.0)
    buffer.add_model_memory("svc-b", 2.0)

    buffer.add_gpu_flops("edge-a", 100.0)
    buffer.add_gpu_flops("edge-b", 80.0)
    buffer.add_gpu_flops(cloud_name, 1000.0)
    buffer.add_memory_capacity("edge-a", 8.0)
    buffer.add_memory_capacity("edge-b", 6.0)
    buffer.add_memory_capacity(cloud_name, 64.0)

    for _ in range(2):
        buffer.add_bandwidths(10.0)
        buffer.add_gpu_utilization("edge-a", 0.2)
        buffer.add_gpu_utilization("edge-b", 0.3)
        buffer.add_gpu_utilization(cloud_name, 0.1)
        buffer.add_memory_utilization("edge-a", 0.4)
        buffer.add_memory_utilization("edge-b", 0.5)
        buffer.add_memory_utilization(cloud_name, 0.2)
        buffer.add_task_complexity("svc-a", 3.0)
        buffer.add_task_complexity("svc-b", 5.0)
        buffer.add_task_latency("svc-a", 1.0)
        buffer.add_task_latency("svc-b", 2.0)

    # Unknown sentinel nodes from the task DAG should be ignored gracefully.
    buffer.add_task_complexity("_start", 99.0)
    buffer.add_task_latency("_end", 99.0)

    logic_feats, phys_feats = buffer.get_state(
        seq_len=4,
        wait_cfg=BufferWaitCfg(min_dynamic_len=1, require_full_seq=False, timeout_s=0.0),
    )

    assert buffer._static_logic_ready is True
    assert buffer._static_phys_ready is True
    assert len(logic_feats["model_flops"]) == 2
    assert len(logic_feats["task_complexity_seq"]) == 2
    assert len(logic_feats["task_complexity_seq"][0]) == 4
    assert len(phys_feats["gpu_flops"]) == 3
    assert len(phys_feats["bandwidth_seq"]) == 3
    assert len(phys_feats["bandwidth_seq"][0]) == 4
    assert phys_feats["bandwidth_seq"][0] == pytest.approx([100.0, 100.0, 100.0, 100.0])
    assert phys_feats["bandwidth_seq"][1] == pytest.approx([100.0, 100.0, 100.0, 100.0])
    assert phys_feats["bandwidth_seq"][2] == pytest.approx([10.0, 10.0, 10.0, 10.0])


@pytest.mark.unit
def test_hedger_collect_state_builds_metrics_from_buffer():
    logical_topology, physical_topology = build_test_topologies()
    cloud_name = physical_topology[physical_topology.cloud_idx]
    buffer = StateBuffer(
        16,
        logical_topology=logical_topology,
        physical_topology=physical_topology,
        fixed_lan_bandwidth_mbps=100.0,
    )

    buffer.add_model_flops({"svc-a": 10.0, "svc-b": 20.0})
    buffer.add_model_memory({"svc-a": 1.0, "svc-b": 2.0})
    buffer.add_gpu_flops({"edge-a": 100.0, "edge-b": 80.0, cloud_name: 1000.0})
    buffer.add_memory_capacity({"edge-a": 8.0, "edge-b": 6.0, cloud_name: 64.0})

    for _ in range(2):
        buffer.add_bandwidths(10.0)
        buffer.add_gpu_utilization("edge-a", 0.2)
        buffer.add_gpu_utilization("edge-b", 0.3)
        buffer.add_gpu_utilization(cloud_name, 0.1)
        buffer.add_memory_utilization("edge-a", 0.4)
        buffer.add_memory_utilization("edge-b", 0.5)
        buffer.add_memory_utilization(cloud_name, 0.2)

    buffer.add_task_complexity("svc-a", 3.0)
    buffer.add_task_complexity("svc-b", 5.0)
    buffer.add_task_latency("svc-a", 1.0)
    buffer.add_task_latency("svc-b", 3.0)
    buffer.add_offloading_reward(1.0)
    buffer.add_offloading_reward(3.0)

    hedger = Hedger.__new__(Hedger)
    hedger.state_buffer = buffer
    hedger.logical_topology = logical_topology
    hedger.physical_topology = physical_topology
    hedger.state_cfg = types.SimpleNamespace(
        offloading_seq_len=4,
        deployment_seq_len=4,
        min_dynamic_len=1,
        wait_timeout_s=0.0,
        require_full_seq=False,
        latency_slo=2.0,
        deployment_reward_window=2,
    )
    hedger.offloading_plan = {"svc-a": "edge-a", "svc-b": cloud_name}
    hedger.cur_deploy_mask = torch.tensor(
        [[True, False, True], [False, True, True]],
        dtype=torch.bool,
    )

    prev_deploy_mask = torch.tensor(
        [[False, False, True], [False, True, True]],
        dtype=torch.bool,
    )

    logic_feats, phys_feats, off_metrics, done = hedger._collect_offloading_state()
    assert logic_feats["task_complexity_seq"].shape == (2, 4)
    assert phys_feats["bandwidth_seq"].shape == (3, 4)
    assert phys_feats["bandwidth_seq"][0].tolist() == pytest.approx([100.0, 100.0, 100.0, 100.0])
    assert phys_feats["bandwidth_seq"][2].tolist() == pytest.approx([10.0, 10.0, 10.0, 10.0])
    assert off_metrics["latency"] == pytest.approx(2.0)
    assert off_metrics["slo_violation"] == pytest.approx(0.5)
    assert off_metrics["cloud_fraction"] == pytest.approx(0.5)
    assert done is False

    _, _, dep_metrics, done = hedger._collect_deployment_state(prev_deploy_mask=prev_deploy_mask)
    assert dep_metrics["avg_offloading_reward"] == pytest.approx(2.0)
    assert dep_metrics["deploy_change_cost"] == pytest.approx(1.0)
    assert done is False


@pytest.mark.unit
def test_hedger_build_edge_index_handles_empty_graph():
    hedger = Hedger.__new__(Hedger)
    hedger.device = torch.device("cpu")

    edge_index = hedger._build_edge_index([])

    assert edge_index.shape == (2, 0)
    assert edge_index.dtype == torch.long


@pytest.mark.unit
def test_build_training_stage_cfg_accepts_only_explicit_stage_names():
    hedger = Hedger.__new__(Hedger)

    warmup = hedger._build_training_stage_cfg("offloading_warmup")
    adaptation = hedger._build_training_stage_cfg("deployment_adaptation")
    finetune = hedger._build_training_stage_cfg("joint_finetune")

    assert warmup == HedgerTrainingStageCfg(
        name="offloading_warmup",
        run_deployment_worker=False,
        update_deployment_policy=False,
        run_offloading_worker=True,
        update_offloading_policy=True,
        use_frozen_offloading_rollout=False,
    )
    with pytest.raises(ValueError):
        hedger._build_training_stage_cfg("stage1")
    assert adaptation == HedgerTrainingStageCfg(
        name="deployment_adaptation",
        run_deployment_worker=True,
        update_deployment_policy=True,
        run_offloading_worker=True,
        update_offloading_policy=False,
        use_frozen_offloading_rollout=True,
    )
    assert finetune == HedgerTrainingStageCfg(
        name="joint_finetune",
        run_deployment_worker=True,
        update_deployment_policy=True,
        run_offloading_worker=True,
        update_offloading_policy=True,
        use_frozen_offloading_rollout=False,
    )


@pytest.mark.unit
def test_inference_hedger_initializes_runtime_and_starts_worker_threads(monkeypatch):
    logical_topology, physical_topology = build_test_topologies()
    cloud_name = physical_topology[physical_topology.cloud_idx]

    hedger = Hedger.__new__(Hedger)
    hedger.logical_topology = logical_topology
    hedger.physical_topology = physical_topology
    hedger.device = torch.device("cpu")
    hedger.seed = 0
    hedger.shared_topology_encoder = torch.nn.Identity()
    hedger.deployment_agent = torch.nn.Identity()
    hedger.offloading_agent = torch.nn.Identity()
    hedger.deployment_thread_stop_event = threading.Event()
    hedger.offloading_thread_stop_event = threading.Event()
    hedger.deployment_plan = None
    hedger.initial_deployment_plan = {"svc-a": ["edge-a"], "svc-b": [cloud_name]}
    hedger.cur_deploy_mask = None

    created_threads = []

    class FakeThread:
        def __init__(self, target=None, daemon=None, **kwargs):
            self.target = target
            self.daemon = daemon
            self.started = False
            created_threads.append(self)

        def start(self):
            self.started = True

        def is_alive(self):
            return True

    def fake_sleep(_):
        hedger.deployment_thread_stop_event.set()
        hedger.offloading_thread_stop_event.set()

    monkeypatch.setattr(
        "core.lib.algorithms.shared.hedger.hedger.threading.Thread",
        FakeThread,
    )
    monkeypatch.setattr(
        "core.lib.algorithms.shared.hedger.hedger.time.sleep",
        fake_sleep,
    )

    hedger.inference_hedger()

    assert hedger.shared_topology_encoder.training is False
    assert hedger.deployment_agent.training is False
    assert hedger.offloading_agent.training is False
    assert hedger.deployment_plan == hedger.initial_deployment_plan
    assert hedger.cur_deploy_mask.tolist() == [
        [True, False, True],
        [False, False, True],
    ]
    assert len(created_threads) == 2
    assert created_threads[0].target == hedger.inference_deployment_agent
    assert created_threads[1].target == hedger.inference_offloading_agent
    assert all(t.daemon is True for t in created_threads)
    assert all(t.started is True for t in created_threads)


@pytest.mark.unit
def test_train_hedger_deployment_adaptation_keeps_frozen_offloading_worker(monkeypatch):
    logical_topology, physical_topology = build_test_topologies()
    cloud_name = physical_topology[physical_topology.cloud_idx]

    hedger = Hedger.__new__(Hedger)
    hedger.logical_topology = logical_topology
    hedger.physical_topology = physical_topology
    hedger.device = torch.device("cpu")
    hedger.mode = "train"
    hedger.seed = 0
    hedger.deployment_interval = 10.0
    hedger.offloading_interval = 1.0
    hedger.training_cfg = build_training_cfg("deployment_adaptation", total_updates=0)
    hedger.checkpoint_cfg = build_checkpoint_cfg(
        "/tmp/hedger-test",
        save=HedgerCheckpointSaveCfg(interval_updates=10),
    )
    hedger.state_cfg = types.SimpleNamespace(deployment_seq_len=8, offloading_seq_len=8)
    hedger.shared_topology_encoder = torch.nn.Identity()
    hedger.deployment_agent = torch.nn.Identity()
    hedger.offloading_agent = torch.nn.Identity()
    hedger.stage_cfg = HedgerTrainingStageCfg(
        name="deployment_adaptation",
        run_deployment_worker=True,
        update_deployment_policy=True,
        run_offloading_worker=True,
        update_offloading_policy=False,
        use_frozen_offloading_rollout=True,
    )
    hedger.deployment_thread_stop_event = threading.Event()
    hedger.offloading_thread_stop_event = threading.Event()
    hedger.cur_deploy_mask = torch.tensor(
        [[True, False, True], [False, False, True]],
        dtype=torch.bool,
    )
    hedger.deployment_plan = {"svc-a": ["edge-a", cloud_name], "svc-b": [cloud_name]}
    hedger.initial_deployment_plan = hedger.deployment_plan
    hedger.offloading_plan = None
    hedger.deployment_transitions = []
    hedger.offloading_transitions = []
    hedger._deployment_update_steps = 0
    hedger._offloading_update_steps = 0
    hedger._epoch = 0
    hedger._global_update_step = 0
    hedger.save_checkpoint = lambda *args, **kwargs: None

    created_threads = []

    class FakeThread:
        def __init__(self, target=None, daemon=None, **kwargs):
            self.target = target
            self.daemon = daemon
            self.started = False
            created_threads.append(self)

        def start(self):
            self.started = True

        def is_alive(self):
            return False

    monkeypatch.setattr(
        "core.lib.algorithms.shared.hedger.hedger.threading.Thread",
        FakeThread,
    )

    hedger.train_hedger()

    assert hedger._frozen_offloading_agent is not None
    assert hedger._frozen_offloading_agent is not hedger.offloading_agent
    assert len(created_threads) == 2
    assert created_threads[0].target == hedger.train_deployment_agent
    assert created_threads[1].target == hedger.train_offloading_agent
    assert all(t.daemon is True for t in created_threads)
    assert all(t.started is True for t in created_threads)


@pytest.mark.unit
def test_stage_aware_checkpoint_saves_latest_final_and_prunes_history(tmp_path):
    hedger = Hedger.__new__(Hedger)
    hedger.device = torch.device("cpu")
    hedger.seed = 7
    hedger.mode = "train"
    hedger.training_cfg = build_training_cfg("deployment_adaptation")
    hedger.checkpoint_cfg = build_checkpoint_cfg(
        str(tmp_path),
        save=HedgerCheckpointSaveCfg(
            interval_updates=20,
            save_latest=True,
            save_final=True,
            save_history=True,
            keep_last_snapshots=2,
        ),
    )
    hedger.shared_topology_encoder = torch.nn.Linear(1, 1)
    hedger.deployment_agent = TinyCheckpointAgent()
    hedger.offloading_agent = TinyCheckpointAgent()
    hedger._deployment_update_steps = 2
    hedger._offloading_update_steps = 5
    hedger._epoch = 0
    hedger._global_update_step = 0
    hedger._loaded_checkpoint_path = "/tmp/parent/latest.pt"

    hedger._epoch = 4
    hedger._global_update_step = 10
    hedger.save_checkpoint(stage_step=4, is_final=False)
    hedger._epoch = 8
    hedger._global_update_step = 14
    hedger.save_checkpoint(stage_step=8, is_final=False)
    hedger._epoch = 12
    hedger._global_update_step = 18
    hedger.save_checkpoint(stage_step=12, is_final=True)

    stage_dir = tmp_path / "deployment_adaptation"
    latest_path = stage_dir / "latest.pt"
    final_path = stage_dir / "final.pt"
    snapshot_dir = stage_dir / "snapshots"
    snapshot_paths = sorted(snapshot_dir.glob("step_*.pt"))

    assert latest_path.exists()
    assert final_path.exists()
    assert [path.name for path in snapshot_paths] == ["step_00000008.pt", "step_00000012.pt"]

    latest_ckpt = torch.load(latest_path, map_location="cpu")
    final_ckpt = torch.load(final_path, map_location="cpu")
    assert latest_ckpt["meta"]["training_stage"] == "deployment_adaptation"
    assert latest_ckpt["meta"]["stage_step"] == 12
    assert latest_ckpt["meta"]["global_step"] == 18
    assert latest_ckpt["meta"]["source_checkpoint"] == "/tmp/parent/latest.pt"
    assert final_ckpt["meta"]["stage_step"] == 12


@pytest.mark.unit
def test_load_checkpoint_resumes_same_stage_counters(tmp_path):
    writer = Hedger.__new__(Hedger)
    writer.device = torch.device("cpu")
    writer.seed = 3
    writer.mode = "train"
    writer.training_cfg = build_training_cfg("offloading_warmup")
    writer.checkpoint_cfg = build_checkpoint_cfg(str(tmp_path))
    writer.shared_topology_encoder = torch.nn.Linear(1, 1)
    writer.deployment_agent = TinyCheckpointAgent()
    writer.offloading_agent = TinyCheckpointAgent()
    writer._deployment_update_steps = 1
    writer._offloading_update_steps = 9
    writer._epoch = 6
    writer._global_update_step = 15
    writer._loaded_checkpoint_path = None
    writer.save_checkpoint(stage_step=6, is_final=False)

    reader = Hedger.__new__(Hedger)
    reader.device = torch.device("cpu")
    reader.seed = 3
    reader.mode = "train"
    reader.training_cfg = build_training_cfg("offloading_warmup")
    reader.checkpoint_cfg = build_checkpoint_cfg(
        str(tmp_path),
        load=HedgerCheckpointLoadCfg(
            enabled=True,
            from_stage="offloading_warmup",
            which="latest",
            restore_encoder=True,
            restore_deployment_agent=True,
            restore_offloading_agent=True,
            restore_optimizer=True,
            reset_stage_counters=False,
        ),
    )
    reader.shared_topology_encoder = torch.nn.Linear(1, 1)
    reader.deployment_agent = TinyCheckpointAgent()
    reader.offloading_agent = TinyCheckpointAgent()
    reader._deployment_update_steps = 0
    reader._offloading_update_steps = 0
    reader._epoch = 0
    reader._global_update_step = 0
    reader._loaded_checkpoint_path = None

    reader.load_checkpoint()

    assert reader._epoch == 6
    assert reader._global_update_step == 15
    assert reader._deployment_update_steps == 1
    assert reader._offloading_update_steps == 9
    assert reader._loaded_checkpoint_path.endswith("offloading_warmup/latest.pt")


@pytest.mark.unit
def test_load_checkpoint_from_previous_stage_resets_stage_local_counters(tmp_path):
    writer = Hedger.__new__(Hedger)
    writer.device = torch.device("cpu")
    writer.seed = 11
    writer.mode = "train"
    writer.training_cfg = build_training_cfg("deployment_adaptation")
    writer.checkpoint_cfg = build_checkpoint_cfg(str(tmp_path))
    writer.shared_topology_encoder = torch.nn.Linear(1, 1)
    writer.deployment_agent = TinyCheckpointAgent()
    writer.offloading_agent = TinyCheckpointAgent()
    writer._deployment_update_steps = 4
    writer._offloading_update_steps = 7
    writer._epoch = 5
    writer._global_update_step = 23
    writer._loaded_checkpoint_path = None
    writer.save_checkpoint(stage_step=5, is_final=True)

    reader = Hedger.__new__(Hedger)
    reader.device = torch.device("cpu")
    reader.seed = 11
    reader.mode = "train"
    reader.training_cfg = build_training_cfg("joint_finetune")
    reader.checkpoint_cfg = build_checkpoint_cfg(
        str(tmp_path),
        load=HedgerCheckpointLoadCfg(
            enabled=True,
            from_stage="deployment_adaptation",
            which="final",
            restore_encoder=True,
            restore_deployment_agent=True,
            restore_offloading_agent=True,
            restore_optimizer=True,
            reset_stage_counters=False,
        ),
    )
    reader.shared_topology_encoder = torch.nn.Linear(1, 1)
    reader.deployment_agent = TinyCheckpointAgent()
    reader.offloading_agent = TinyCheckpointAgent()
    reader._deployment_update_steps = 0
    reader._offloading_update_steps = 0
    reader._epoch = 0
    reader._global_update_step = 0
    reader._loaded_checkpoint_path = None

    reader.load_checkpoint()

    assert reader._epoch == 0
    assert reader._global_update_step == 23
    assert reader._deployment_update_steps == 0
    assert reader._offloading_update_steps == 0
    assert reader._loaded_checkpoint_path.endswith("deployment_adaptation/final.pt")


@pytest.mark.unit
def test_register_physical_topology_syncs_agent_indices():
    hedger = Hedger.__new__(Hedger)
    hedger.physical_topology = None
    hedger.deployment_agent = types.SimpleNamespace(cloud_idx=-1)
    hedger.offloading_agent = types.SimpleNamespace(source=-1, cloud_idx=-1)

    hedger.register_physical_topology(["edge-a", "edge-b"], "edge-b")

    assert hedger.physical_topology.source_idx == 0
    assert hedger.physical_topology.cloud_idx == 2
    assert hedger.deployment_agent.cloud_idx == 2
    assert hedger.offloading_agent.source == 0
    assert hedger.offloading_agent.cloud_idx == 2


@pytest.mark.unit
def test_map_deployment_plan_to_mask_ignores_unknown_services_and_devices():
    logical_topology, physical_topology = build_test_topologies()
    cloud_name = physical_topology[physical_topology.cloud_idx]

    hedger = Hedger.__new__(Hedger)
    hedger.logical_topology = logical_topology
    hedger.physical_topology = physical_topology
    hedger.device = torch.device("cpu")

    deploy_mask = hedger._map_deployment_plan_to_deployment_mask(
        {
            "svc-a": ["edge-a", "missing-edge"],
            "missing-service": ["edge-b"],
            "svc-b": cloud_name,
        }
    )

    assert deploy_mask.tolist() == [
        [True, False, True],
        [False, False, True],
    ]


@pytest.mark.unit
def test_hedger_agent_extract_task_complexity_reduces_nested_scenario_data():
    service = types.SimpleNamespace(
        get_scenario_data=lambda: {"obj_num": [1, 3], "size": {"w": 2, "h": 4}},
    )

    complexity = HedgerAgent._extract_task_complexity(service)

    assert complexity == pytest.approx(2.0)


@pytest.mark.unit
def test_hedger_agent_extract_task_complexity_ignores_non_obj_num_fields():
    service = types.SimpleNamespace(
        get_scenario_data=lambda: {"obj_size": [10, 20], "delay": 5.0},
    )

    complexity = HedgerAgent._extract_task_complexity(service)

    assert complexity == pytest.approx(0.0)


@pytest.mark.unit
def test_hedger_agent_get_schedule_plan_tolerates_missing_default_mappings(monkeypatch):
    dag = {
        "svc-a": {"service": {"service_name": "svc-a"}, "next_nodes": ["svc-b"]},
        "svc-b": {"service": {"service_name": "svc-b"}, "next_nodes": []},
    }

    hedger = types.SimpleNamespace(
        register_logical_topology=lambda dag: None,
        register_physical_topology=lambda edge_nodes, source_device: None,
        register_state_buffer=lambda: None,
        get_offloading_plan=lambda: None,
    )
    agent = HedgerAgent.__new__(HedgerAgent)
    agent.cloud_device = "cloud-x"
    agent.default_configuration = None
    agent.default_offloading = None
    agent.hedger = hedger

    monkeypatch.setattr(
        "core.lib.algorithms.schedule_agent.hedger_agent.KubeConfig.get_service_nodes_dict",
        lambda: {"svc-a": {}, "svc-b": {}},
    )

    policy = agent.get_schedule_plan(
        {
            "source": {"id": 1},
            "source_device": "edge-a",
            "all_edge_devices": ["edge-a", "edge-b"],
            "dag": dag,
        }
    )

    assert policy["dag"]["svc-a"]["service"]["execute_device"] == "cloud-x"
    assert policy["dag"]["svc-b"]["service"]["execute_device"] == "cloud-x"


@pytest.mark.unit
def test_hedger_agent_update_resource_tolerates_partial_resource_updates():
    calls = []
    buffer = types.SimpleNamespace(
        add_bandwidths=lambda value: calls.append(("bandwidth", value)),
        add_gpu_flops=lambda device, value: calls.append(("gpu_flops", device, value)),
        add_memory_capacity=lambda device, value: calls.append(("memory_capacity", device, value)),
        add_gpu_utilization=lambda device, value: calls.append(("gpu_usage", device, value)),
        add_memory_utilization=lambda device, value: calls.append(("memory_usage", device, value)),
        add_model_flops=lambda service, value: calls.append(("model_flops", service, value)),
        add_model_memory=lambda service, value: calls.append(("model_memory", service, value)),
    )
    agent = HedgerAgent.__new__(HedgerAgent)
    agent.hedger = types.SimpleNamespace(state_buffer=buffer)

    agent.update_resource(
        "edge-a",
        {
            "available_bandwidth": 12.5,
            "gpu_usage": 0.3,
            "model_flops": {"svc-a": 10.0},
        },
    )

    assert calls == [
        ("bandwidth", 12.5),
        ("gpu_usage", "edge-a", 0.3),
        ("model_flops", "svc-a", 10.0),
    ]


@pytest.mark.unit
def test_state_buffer_bandwidth_model_is_device_agnostic():
    logical_topology, physical_topology = build_test_topologies()
    buffer = StateBuffer(
        16,
        logical_topology=logical_topology,
        physical_topology=physical_topology,
        fixed_lan_bandwidth_mbps=100.0,
    )

    buffer.add_model_flops({"svc-a": 10.0, "svc-b": 20.0})
    buffer.add_model_memory({"svc-a": 1.0, "svc-b": 2.0})
    buffer.add_gpu_flops({"edge-a": 100.0, "edge-b": 80.0, physical_topology[physical_topology.cloud_idx]: 1000.0})
    buffer.add_memory_capacity({"edge-a": 8.0, "edge-b": 6.0, physical_topology[physical_topology.cloud_idx]: 64.0})
    buffer.add_gpu_utilization("edge-a", 0.2)
    buffer.add_gpu_utilization("edge-b", 0.3)
    buffer.add_gpu_utilization(physical_topology[physical_topology.cloud_idx], 0.1)
    buffer.add_memory_utilization("edge-a", 0.4)
    buffer.add_memory_utilization("edge-b", 0.5)
    buffer.add_memory_utilization(physical_topology[physical_topology.cloud_idx], 0.2)
    buffer.add_task_complexity("svc-a", 3.0)
    buffer.add_task_complexity("svc-b", 5.0)
    buffer.add_task_latency("svc-a", 1.0)
    buffer.add_task_latency("svc-b", 2.0)

    buffer.add_bandwidths(12.5)

    _, phys_feats = buffer.get_state(
        seq_len=1,
        wait_cfg=BufferWaitCfg(min_dynamic_len=1, require_full_seq=False, timeout_s=0.0),
    )

    assert phys_feats["bandwidth_seq"][0] == pytest.approx([100.0])
    assert phys_feats["bandwidth_seq"][1] == pytest.approx([100.0])
    assert phys_feats["bandwidth_seq"][2] == pytest.approx([12.5])


@pytest.mark.unit
def test_initial_deployment_policy_accepts_single_node_strings():
    policy = HedgerInitialDeploymentPolicy.__new__(HedgerInitialDeploymentPolicy)
    policy.default_deployment = {"svc-a": "edge-a"}
    policy.hedger = types.SimpleNamespace(
        register_logical_topology=lambda dag: None,
        register_physical_topology=lambda edge_nodes, source_device: None,
        register_state_buffer=lambda: None,
        register_initial_deployment=lambda deployment_plan: None,
        get_initial_deployment_plan=lambda: {"svc-a": "edge-a"},
    )

    deploy_plan = policy(
        {
            "source": {"id": 1, "source_device": "edge-a"},
            "dag": {
                "svc-a": {"service": {"service_name": "svc-a"}, "next_nodes": []},
                "svc-b": {"service": {"service_name": "svc-b"}, "next_nodes": []},
            },
            "node_set": ["edge-a", "edge-b"],
        }
    )

    assert deploy_plan["svc-a"] == ["edge-a"]
    assert deploy_plan["svc-b"] == ["edge-a", "edge-b"]
