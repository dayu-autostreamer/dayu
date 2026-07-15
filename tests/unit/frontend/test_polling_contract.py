from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_system_parameters_polling_is_settle_then_schedule_and_abortable():
    source = (REPO_ROOT / "frontend/src/stores/systemParameters.ts").read_text(
        encoding="utf-8"
    )

    assert "setInterval(" not in source
    assert "requestInFlight" in source
    assert "new AbortController()" in source
    assert "await this.fetchLatest(controller.signal)" in source
    assert "this.requestController?.abort()" in source
    assert "this.pollingTimer = setTimeout(poll, 2000)" in source
    assert "state.status === 'install' && state.phase === 'active'" in source


def test_service_query_polling_and_detail_requests_cancel_stale_work():
    source = (REPO_ROOT / "frontend/src/views/install/SvcQuery.vue").read_text(
        encoding="utf-8"
    )

    assert "setInterval(" not in source
    assert "stateController?.abort()" in source
    assert "stateTimer = window.setTimeout(pollInstallState, 3000)" in source
    assert "this.serviceListController?.abort()" in source
    assert "this.serviceInfoController?.abort()" in source
    assert "this.selected_service !== service" in source
    assert "this.serviceInfoTimer = window.setTimeout" in source
    assert "void this.sendRequest(service)" in source
    assert "this.stopServiceInfoPolling()" in source
    assert "if (!this.runtimeReady)" in source
    assert "this.install_state.setPhase('uninstalling')" in source


def test_service_detail_uses_normalized_resource_meters_and_shared_bandwidth():
    source = (REPO_ROOT / "frontend/src/views/install/SvcQuery.vue").read_text(
        encoding="utf-8"
    )

    assert "Pod usage vs node resources" in source
    assert 'role="progressbar"' in source
    assert ':aria-valuenow="meterWidth(item.cpu)"' in source
    assert "Math.min(100, Math.max(0, metric.utilization_percent))" in source
    assert "usage_millicores" in source
    assert "usage_bytes" in source
    assert "node_allocatable" in source
    assert "node_capacity" in source
    assert "Collecting metrics" in source
    assert "Metrics unavailable" in source
    assert "Last known sample" in source
    assert "Shared Edge → Cloud" in source
    assert "Measured by ${metric.probe_node}" in source
    assert "Multiple active probe values" in source


def test_install_state_separates_session_ownership_from_runtime_readiness():
    source = (REPO_ROOT / "frontend/src/stores/installState.ts").read_text(
        encoding="utf-8"
    )

    assert "status: 'uninstall'" in source
    assert "phase: 'uninstalled'" in source
    assert "state.status === 'install' && state.phase === 'active'" in source
    assert "install(phase = 'active')" in source


def test_result_polling_is_single_flight_and_cancels_stale_fetches():
    source = (REPO_ROOT / "frontend/src/views/result/index.vue").read_text(
        encoding="utf-8"
    )

    assert "setInterval(" not in source
    assert "new AbortController()" in source
    assert "await this.getLatestResultData(controller.signal)" in source
    assert "this.pollingTimer = window.setTimeout(poll, 2000)" in source
    assert "this.resultController?.abort()" in source
    assert "if (this.pollingActive || this.componentUnmounted) return" in source
    assert "emitter.off('force-update-charts', this.forceUpdateHandler)" in source
