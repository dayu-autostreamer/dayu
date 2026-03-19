import importlib
import types

import pytest


runtime_security_module = importlib.import_module("core.lib.common.runtime_security_backports")


@pytest.mark.unit
def test_runtime_security_backports_version_helpers_and_range_parsing():
    parse_range_owner = type(
        "ParseRangeOwner",
        (),
        {
            "_parse_ranges": classmethod(runtime_security_module._backported_parse_ranges_method),
        },
    )

    assert runtime_security_module._version_tuple("0.27.0") == (0, 27, 0)
    assert runtime_security_module._version_lt("0.27.0", (0, 49, 1)) is True
    assert runtime_security_module._backported_parse_ranges("0-3, 5-6", 10) == [(0, 4), (5, 7)]
    assert runtime_security_module._backported_parse_range_header(parse_range_owner, "bytes=0-1,2-3", 10) == [(0, 4)]

    with pytest.raises(runtime_security_module.MalformedRangeHeader):
        runtime_security_module._backported_parse_range_header(parse_range_owner, "items=0-1", 10)
    with pytest.raises(runtime_security_module.MalformedRangeHeader):
        runtime_security_module._backported_parse_range_header(parse_range_owner, "bytes=", 10)
    with pytest.raises(runtime_security_module.RangeNotSatisfiable):
        runtime_security_module._backported_parse_range_header(parse_range_owner, "bytes=20-30", 10)


@pytest.mark.unit
def test_runtime_security_backports_patch_starlette_and_upload_behaviour(monkeypatch):
    module = runtime_security_module
    monkeypatch.setattr(module, "_STARLETTE_PATCH_APPLIED", False)
    monkeypatch.setattr(module.starlette, "__version__", "0.27.0")
    monkeypatch.delattr(module.FileResponse, "_parse_ranges", raising=False)
    monkeypatch.delattr(module.FileResponse, "_parse_range_header", raising=False)

    module.apply_starlette_backports()
    module.apply_starlette_backports()

    assert hasattr(module.FileResponse, "_parse_ranges")
    assert hasattr(module.FileResponse, "_parse_range_header")
    assert module._STARLETTE_PATCH_APPLIED is True

    class DummyFile:
        _max_size = 8

        def __init__(self):
            self.buffer = []
            self.offset = 3

        def tell(self):
            return self.offset

        def write(self, data):
            self.buffer.append(data)

    class DummyUpload:
        def __init__(self):
            self.file = None
            self.size = 0
            self._in_memory = True

    def original_init(self, file, size=None, filename=None, headers=None):
        self.file = file
        self.size = size

    patched_init = module._backported_upload_init(original_init)
    upload = DummyUpload()
    patched_init(upload, DummyFile(), size=1)
    assert upload._max_mem_size == 8
    assert module._upload_will_roll(upload, 10) is True


@pytest.mark.unit
def test_runtime_security_backports_harden_pillow_psd_registry(monkeypatch):
    module = runtime_security_module
    monkeypatch.setattr(module, "_PILLOW_HARDENING_APPLIED", False)

    fake_image = types.SimpleNamespace(
        OPEN={"PSD": object(), "PNG": object()},
        SAVE={"PSD": object(), "PNG": object()},
        MIME={"PSD": "image/psd", "PNG": "image/png"},
        EXTENSION={".psd": "PSD", ".png": "PNG"},
        ID=["PSD", "PNG"],
        init=lambda: None,
    )
    monkeypatch.setitem(__import__("sys").modules, "PIL", types.SimpleNamespace(Image=fake_image))

    module.apply_pillow_psd_hardening()

    assert "PSD" not in fake_image.OPEN
    assert "PSD" not in fake_image.SAVE
    assert "PSD" not in fake_image.MIME
    assert ".psd" not in fake_image.EXTENSION
    assert fake_image.ID == ["PNG"]
