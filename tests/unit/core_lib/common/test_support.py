import importlib
from pathlib import Path

import imagehash
import numpy as np
import pytest
import yaml
from PIL import Image


counter_module = importlib.import_module("core.lib.common.counter")
encode_module = importlib.import_module("core.lib.common.encode_ops")
hash_module = importlib.import_module("core.lib.common.hash_ops")
name_module = importlib.import_module("core.lib.common.name")
video_module = importlib.import_module("core.lib.common.video_ops")
yaml_module = importlib.import_module("core.lib.common.yaml_ops")


Counter = counter_module.Counter
EncodeOps = encode_module.EncodeOps
HashOps = hash_module.HashOps
NameMaintainer = name_module.NameMaintainer
VideoOps = video_module.VideoOps
YamlOps = yaml_module.YamlOps


class DummyNameTask:
    def get_source_id(self):
        return 3

    def get_task_id(self):
        return 11

    def get_root_uuid(self):
        return "root-uuid"


@pytest.mark.unit
def test_counter_tracks_named_sequences_and_blocks_instantiation():
    Counter.reset_all_counts()

    assert Counter.get_count("alpha") == 0
    assert Counter.get_count("alpha") == 1
    assert Counter.get_count("beta") == 0

    snapshot = Counter.get_all_counts()
    snapshot["alpha"] = 99
    assert Counter.get_all_counts() == {"alpha": 1, "beta": 0}

    Counter.reset_count("alpha")
    assert Counter.get_all_counts() == {"beta": 0}

    with pytest.raises(RuntimeError, match="cannot be instantiated"):
        Counter()


@pytest.mark.unit
def test_encode_hash_name_and_video_helpers_cover_success_and_validation(monkeypatch):
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    frame[:, :, 1] = 255

    encoded = EncodeOps.encode_image(frame)
    decoded = EncodeOps.decode_image(encoded)

    assert encoded.startswith("data:image/jpg;base64,")
    assert decoded.shape == frame.shape
    assert decoded.dtype == frame.dtype

    with pytest.raises(ValueError, match="numpy array"):
        EncodeOps.encode_image("not-an-image")

    import cv2

    monkeypatch.setattr(cv2, "imencode", lambda ext, image: (False, None))
    with pytest.raises(ValueError, match="encoding failed"):
        EncodeOps.encode_image(frame)

    with pytest.raises(ValueError, match="must be a string"):
        EncodeOps.decode_image(123)

    with pytest.raises(ValueError, match="Invalid base64"):
        EncodeOps.decode_image("data:image/jpg;base64,not-base64")

    raw_base64 = encoded.split(",", 1)[1]
    roundtrip_from_raw = EncodeOps.decode_image(raw_base64)
    assert roundtrip_from_raw.shape == frame.shape

    monkeypatch.setattr(cv2, "imdecode", lambda data, mode: None)
    with pytest.raises(ValueError, match="Invalid base64 string or incompatible image data"):
        EncodeOps.decode_image(raw_base64)

    assert str(HashOps.get_frame_hash(frame)) == str(imagehash.phash(Image.fromarray(frame)))

    task = DummyNameTask()
    assert NameMaintainer.get_time_ticket_tag_prefix(task) == "dayu:source-3-task-11:root-uuid"
    assert NameMaintainer.get_task_data_file_name("cam-a", 9, "mp4") == "data_of_source_cam-a_task_9.mp4"
    assert NameMaintainer.standardize_device_name("Edge-X_1.A") == "edgex1a"
    assert NameMaintainer.standardize_service_name("Face_Detection") == "face-detection"

    assert VideoOps.text2resolution("720p") == (1280, 720)
    assert VideoOps.resolution2text((640, 480)) == "480p"

    with pytest.raises(AssertionError, match="Invalid resolution"):
        VideoOps.text2resolution("2k")

    with pytest.raises(AssertionError, match="Invalid resolution"):
        VideoOps.resolution2text((1, 2))


@pytest.mark.unit
def test_yaml_ops_support_include_multi_document_and_validation(tmp_path):
    included_yaml = tmp_path / "included.yaml"
    included_yaml.write_text("name: detector\nreplicas: 2\n", encoding="utf-8")

    main_yaml = tmp_path / "main.yaml"
    main_yaml.write_text("service: !include included.yaml\nmode: test\n", encoding="utf-8")

    assert YamlOps.read_yaml(str(main_yaml)) == {
        "service": {"name": "detector", "replicas": 2},
        "mode": "test",
    }

    single_yaml = tmp_path / "single.yaml"
    YamlOps.write_yaml({"alpha": 1, "items": ["x"]}, str(single_yaml))
    assert yaml.safe_load(single_yaml.read_text(encoding="utf-8")) == {"alpha": 1, "items": ["x"]}
    assert YamlOps.is_yaml_file(str(single_yaml)) is True

    multi_yaml = tmp_path / "multi.yaml"
    YamlOps.write_all_yaml([{"first": 1}, {"second": 2}], str(multi_yaml))
    assert YamlOps.read_all_yaml(str(multi_yaml)) == [{"first": 1}, {"second": 2}]

    YamlOps.clean_yaml(str(multi_yaml))
    assert multi_yaml.read_text(encoding="utf-8") == ""

    broken_yaml = tmp_path / "broken.yaml"
    broken_yaml.write_text("alpha: [1, 2\n", encoding="utf-8")
    assert YamlOps.is_yaml_file(str(broken_yaml)) is False
    assert YamlOps.is_yaml_file(str(Path(tmp_path) / "missing.yaml")) is False
