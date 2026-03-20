import importlib
import importlib.util
import pickle
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest


def build_fake_torch_modules():
    class FakeTensor:
        def __init__(self, data):
            self.data = np.asarray(data)

        def unsqueeze(self, _axis):
            return self

    class DummyModule:
        def __call__(self, *args, **kwargs):
            if hasattr(self, "forward"):
                return self.forward(*args, **kwargs)
            return FakeTensor([[0.0, 0.0]])

        def parameters(self):
            return []

        def state_dict(self):
            return {}

        def load_state_dict(self, _state):
            return None

    class DummyLinear:
        def __init__(self, *_args, **_kwargs):
            pass

    class DummyReLU:
        def __call__(self, value):
            return value

    class DummySequential:
        def __init__(self, *_layers):
            pass

        def __call__(self, _value):
            return FakeTensor([[0.2, 0.8]])

    class DummyOptimizer:
        def zero_grad(self):
            return None

        def step(self):
            return None

    class DummyNoGrad:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    fake_nn = ModuleType("torch.nn")
    fake_nn.Module = DummyModule
    fake_nn.Sequential = DummySequential
    fake_nn.Linear = DummyLinear
    fake_nn.ReLU = DummyReLU

    fake_optim = ModuleType("torch.optim")
    fake_optim.Adam = lambda params, lr=0.001: DummyOptimizer()

    fake_torch = ModuleType("torch")
    fake_torch.nn = fake_nn
    fake_torch.optim = fake_optim
    fake_torch.FloatTensor = lambda data: FakeTensor(data)
    fake_torch.LongTensor = lambda data: FakeTensor(data)
    fake_torch.no_grad = lambda: DummyNoGrad()
    fake_torch.argmax = lambda tensor: SimpleNamespace(item=lambda: int(np.argmax(tensor.data)))

    return fake_torch, fake_nn, fake_optim


def load_frame_compress_module(module_name, monkeypatch, *, with_fake_torch=False):
    package_dir = (
        Path(__file__).resolve().parents[4]
        / "dependency"
        / "core"
        / "lib"
        / "algorithms"
        / "frame_compress"
    )
    package_name = f"_test_frame_compress_{uuid4().hex}"
    package_module = ModuleType(package_name)
    package_module.__path__ = [str(package_dir)]
    sys.modules[package_name] = package_module

    class_factory_module = importlib.import_module("core.lib.common.class_factory")
    monkeypatch.setattr(
        class_factory_module.ClassFactory,
        "register",
        staticmethod(lambda *args, **kwargs: (lambda cls: cls)),
    )

    if with_fake_torch:
        fake_torch, fake_nn, fake_optim = build_fake_torch_modules()
        monkeypatch.setitem(sys.modules, "torch", fake_torch)
        monkeypatch.setitem(sys.modules, "torch.nn", fake_nn)
        monkeypatch.setitem(sys.modules, "torch.optim", fake_optim)

    base_spec = importlib.util.spec_from_file_location(
        f"{package_name}.base_compress",
        package_dir / "base_compress.py",
    )
    base_module = importlib.util.module_from_spec(base_spec)
    sys.modules[f"{package_name}.base_compress"] = base_module
    base_spec.loader.exec_module(base_module)

    spec = importlib.util.spec_from_file_location(
        f"{package_name}.{module_name}",
        package_dir / f"{module_name}.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"{package_name}.{module_name}"] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.unit
def test_frame_compress_package_loader_exports_only_public_symbols(monkeypatch):
    package_dir = (
        Path(__file__).resolve().parents[4]
        / "dependency"
        / "core"
        / "lib"
        / "algorithms"
        / "frame_compress"
    )
    package_name = f"_test_frame_compress_pkg_{uuid4().hex}"
    fake_simple = ModuleType(f"{package_name}.simple_compress")
    fake_simple.__all__ = ["SimpleCompress"]
    fake_simple.SimpleCompress = object()

    fake_hidden = ModuleType(f"{package_name}.hidden")
    fake_hidden.internal = object()

    monkeypatch.setattr(
        importlib.import_module("pkgutil"),
        "iter_modules",
        lambda path: [(None, "simple_compress", False), (None, "hidden", False)],
    )

    def fake_import_module(name, package=None):
        if name == ".simple_compress":
            return fake_simple
        if name == ".hidden":
            return fake_hidden
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    spec = importlib.util.spec_from_file_location(
        package_name,
        package_dir / "__init__.py",
        submodule_search_locations=[str(package_dir)],
    )
    package_module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = package_module
    spec.loader.exec_module(package_module)

    assert package_module.__all__ == ["SimpleCompress"]
    assert package_module.SimpleCompress is fake_simple.SimpleCompress


@pytest.mark.unit
def test_adaptive_compress_initializes_model_and_exposes_helper_paths(monkeypatch, tmp_path):
    adaptive_module = load_frame_compress_module("adaptive_compress", monkeypatch, with_fake_torch=True)

    model_file = tmp_path / "model.pkl"
    model_file.write_bytes(pickle.dumps(SimpleNamespace(epsilon=5)))

    loaded_agent = adaptive_module.AdaptiveCompress.load_model(str(model_file))
    assert loaded_agent.epsilon == 5

    monkeypatch.setattr(adaptive_module.Context, "get_file_path", staticmethod(lambda _name: str(model_file)))
    monkeypatch.setattr(
        adaptive_module.AdaptiveCompress,
        "load_model",
        staticmethod(lambda _filename="agent_model.pkl": SimpleNamespace(epsilon=7)),
    )

    compressor = adaptive_module.AdaptiveCompress()
    assert compressor.agent_file == str(model_file)
    assert compressor.agent.epsilon == 0
    assert adaptive_module.AdaptiveCompress.generate_yuv_temp_path(2, 9) == "video_source_2_task_9_tmp.yuv"
    assert adaptive_module.AdaptiveCompress.generate_file_path(2, 9) == "video_source_2_task_9.h264"
    assert adaptive_module.AdaptiveCompress.generate_roi_path(2, 9) == "roi_2_task_9.txt"

    state = adaptive_module.State(1, 2, 3, 4, 5, 6)
    assert np.array_equal(state.to_array(), np.array([1, 2, 3, 4, 5, 6]))

    performance_gt = {
        20: {"file_size": 50, "accuracy": 0.6},
        30: {"file_size": 20, "accuracy": 0.8},
    }
    assert adaptive_module.AdaptiveCompress.estimate_performace_with_qp(
        performance_gt, 30, -10, -5, 0.5, 0.5
    ) > 0


@pytest.mark.unit
def test_adaptive_compress_analysis_qp_selection_and_call_cleanup(monkeypatch, tmp_path):
    adaptive_module = load_frame_compress_module("adaptive_compress", monkeypatch, with_fake_torch=True)
    compressor = adaptive_module.AdaptiveCompress.__new__(adaptive_module.AdaptiveCompress)
    compressor.past_acc = 0
    compressor.past_latency = 0
    compressor.past_qp = 45

    monkeypatch.setattr(compressor, "calculate_edge_density", lambda frame, roi=None: 10 if roi is None else 5)
    monkeypatch.setattr(
        compressor,
        "calculate_texture_complexity",
        lambda frame, roi=None: 6 if roi is None else 2,
    )
    frames = [np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(3)]
    rois = [[(0, 0, 1, 1)] for _ in range(3)]
    total_complexity, roi_complexity = compressor.analyze_packet_content(frames, rois)
    assert total_complexity == pytest.approx(24.0)
    assert roi_complexity == pytest.approx(12.6)

    yuv_path = tmp_path / "frames.yuv"
    adaptive_module.AdaptiveCompress.init_yuv_temp_path(str(yuv_path), frames[:2])
    assert yuv_path.stat().st_size == 12

    performance_gt = {
        20: {"file_size": 80, "accuracy": 0.61},
        30: {"file_size": 32, "accuracy": 0.83},
    }
    qp = compressor.adjust_qp(
        performance_gt,
        total_complexity=24.0,
        roi_complexity=12.0,
        agent=SimpleNamespace(choose_action=lambda state: 0),
        past_qp=45,
    )
    assert qp == 30
    assert compressor.past_qp == 30
    assert compressor.past_acc == 0.83
    assert compressor.past_latency > 0

    errors = []
    monkeypatch.setattr(adaptive_module.LOGGER, "error", lambda message: errors.append(message))
    compressor.past_qp = 45
    assert compressor.adjust_qp(
        performance_gt,
        total_complexity=24.0,
        roi_complexity=12.0,
        agent=SimpleNamespace(
            choose_action=lambda state: (_ for _ in ()).throw(ValueError("bad state"))
        ),
        past_qp=45,
    ) == 45
    assert any("bad state" in message for message in errors)

    runtime_compressor = adaptive_module.AdaptiveCompress.__new__(adaptive_module.AdaptiveCompress)
    runtime_compressor.performace_gt = performance_gt
    runtime_compressor.agent = SimpleNamespace()
    runtime_compressor.past_qp = 45

    init_calls = []
    removed = []
    commands = []
    monkeypatch.setattr(
        runtime_compressor,
        "init_yuv_temp_path",
        lambda path, frame_buffer: init_calls.append((path, len(frame_buffer))),
    )
    monkeypatch.setattr(runtime_compressor, "analyze_packet_content", lambda frame_set, roi_set: (18.0, 6.0))
    monkeypatch.setattr(runtime_compressor, "adjust_qp", lambda *args, **kwargs: 37)
    monkeypatch.setattr(adaptive_module.FileOps, "remove_file", lambda file_path: removed.append(file_path))
    monkeypatch.setattr(adaptive_module.LOGGER, "debug", lambda message: None)

    class DummyProcess:
        def wait(self):
            commands.append("wait")

    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda command, shell=True: commands.append(command) or DummyProcess(),
    )

    frame_buffer = [(np.zeros((4, 6, 3), dtype=np.uint8), [(0, 0, 2, 2)])]
    output_path = runtime_compressor(SimpleNamespace(), frame_buffer, source_id=2, task_id=9)
    assert output_path == "video_source_2_task_9.h264"
    assert init_calls == [("video_source_2_task_9_tmp.yuv", 1)]
    assert any("--econstqp -qpi 37 37 37" in command for command in commands if isinstance(command, str))
    assert removed == ["roi_2_task_9.txt", "video_source_2_task_9_tmp.yuv"]

    with pytest.raises(AssertionError, match="frame buffer is empty"):
        runtime_compressor(SimpleNamespace(), [], source_id=1, task_id=1)


@pytest.mark.unit
def test_adaptive_compress_dqn_agent_supports_explore_exploit_and_buffer_updates(monkeypatch):
    adaptive_module = load_frame_compress_module("adaptive_compress", monkeypatch, with_fake_torch=True)

    exploring_agent = adaptive_module.DQNAgent(state_dim=6, action_dim=2, epsilon=1.0)
    monkeypatch.setattr(adaptive_module.random, "random", lambda: 0.0)
    monkeypatch.setattr(adaptive_module.random, "randint", lambda start, end: end)
    assert exploring_agent.choose_action([1, 2, 3, 4, 5, 6]) == 1

    exploiting_agent = adaptive_module.DQNAgent(state_dim=6, action_dim=2, epsilon=0.0)
    monkeypatch.setattr(adaptive_module.random, "random", lambda: 1.0)
    assert exploiting_agent.choose_action([1, 2, 3, 4, 5, 6]) == 1

    exploiting_agent.store_transition([1, 2, 3, 4, 5, 6], 0, 1.0, [1, 2, 3, 4, 5, 7], False)
    assert len(exploiting_agent.memory) == 1
    exploiting_agent.update()
    assert exploiting_agent.steps == 0
