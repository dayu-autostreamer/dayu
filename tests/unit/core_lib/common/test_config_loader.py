import builtins
import importlib

import pytest


config_module = importlib.import_module("core.lib.common.config")
ConfigLoader = config_module.ConfigLoader


@pytest.mark.unit
def test_config_loader_internal_helpers_cover_extension_sorting_and_io_errors(monkeypatch, tmp_path):
    assert ConfigLoader._get_file_extension("settings.JSON") == "json"
    assert ConfigLoader._get_file_extension("settings") is None

    ordered = ConfigLoader._order_parsers_by_extension("yml")
    assert [parser["name"] for parser in ordered[:1]] == ["YAML"]
    assert ordered is not ConfigLoader._PARSERS

    def failing_open(*args, **kwargs):
        raise IOError("permission denied")

    monkeypatch.setattr(builtins, "open", failing_open)
    with pytest.raises(ValueError, match="Failed to read file"):
        ConfigLoader._read_file_content(str(tmp_path / "config.json"))


@pytest.mark.unit
def test_config_loader_load_skips_empty_results_and_generic_parser_failures(monkeypatch, tmp_path):
    config_path = tmp_path / "config.cfg"
    config_path.write_text("ignored", encoding="utf-8")

    monkeypatch.setattr(
        ConfigLoader,
        "_PARSERS",
        [
            {
                "name": "NoneParser",
                "extensions": ["cfg"],
                "load": lambda content: None,
                "exceptions": (ValueError,),
                "required": False,
            },
            {
                "name": "BrokenParser",
                "extensions": ["cfg"],
                "load": lambda content: (_ for _ in ()).throw(RuntimeError("boom")),
                "exceptions": (ValueError,),
                "required": False,
            },
            {
                "name": "SuccessParser",
                "extensions": ["cfg"],
                "load": lambda content: {"loaded": True},
                "exceptions": (ValueError,),
                "required": False,
            },
        ],
    )

    assert ConfigLoader.load(str(config_path)) == {"loaded": True}
