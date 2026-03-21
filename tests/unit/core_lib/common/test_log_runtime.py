import importlib

import pytest


@pytest.mark.unit
def test_logger_uses_default_name_and_configures_real_logger(monkeypatch):
    log_module = importlib.import_module("core.lib.common.log")

    monkeypatch.setattr(log_module.Context, "get_parameter", staticmethod(lambda key, default=None: "WARNING"))

    logger_wrapper = log_module.Logger()
    logger = logger_wrapper.logger

    assert logger.name == log_module.SystemConstant.DEFAULT.value
    assert logger.level == log_module.logging.WARNING
    assert logger.propagate is False
    assert logger.handlers

    added_handler = logger.handlers[-1]
    assert getattr(added_handler, "formatter", None) is not None
    logger.removeHandler(added_handler)
    added_handler.close()


@pytest.mark.unit
def test_logger_preserves_explicit_name(monkeypatch):
    log_module = importlib.import_module("core.lib.common.log")

    monkeypatch.setattr(log_module.Context, "get_parameter", staticmethod(lambda key, default=None: "INFO"))

    logger_wrapper = log_module.Logger("custom-monitor")
    logger = logger_wrapper.logger

    assert logger.name == "custom-monitor"
    added_handler = logger.handlers[-1]
    logger.removeHandler(added_handler)
    added_handler.close()
