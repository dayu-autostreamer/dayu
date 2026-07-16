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
    assert "installStore.isReady" in source
    assert "runtimeInstallId" in source
    assert "syncRuntimeGeneration(installStore.installId" in source
    assert "this.bufferedTaskCache.splice(0, this.bufferedTaskCache.length)" in source
    assert "localStorage" not in source
    assert "fetch('/api/install_state')" not in source


def test_install_state_polling_is_centralized_single_flight_and_abortable():
    store = (REPO_ROOT / "frontend/src/stores/installState.ts").read_text(
        encoding="utf-8"
    )
    install = (REPO_ROOT / "frontend/src/views/install/SvcInstall.vue").read_text(
        encoding="utf-8"
    )
    query = (REPO_ROOT / "frontend/src/views/install/SvcQuery.vue").read_text(
        encoding="utf-8"
    )

    assert store.count("'/api/install_state'") == 1
    assert "fetchJsonWithTimeout<InstallStateSnapshot>(" in store
    assert "STATE_REQUEST_TIMEOUT_MS" in store
    assert "stateRequest" in store
    assert "new AbortController()" in store
    assert "await refresh()" in store
    assert "pollingTimer = window.setTimeout(poll, interval)" in store
    assert "stateController?.abort()" in store
    assert "fetch('/api/install_state'" not in install
    assert "fetch('/api/install_state'" not in query
    assert "dayu-install-changed" not in install
    assert "dayu-install-changed" not in query
    assert "updated_at" not in store
    assert "retiredInstallIds" not in store
    assert "confirmInstall" not in store
    assert "function sync(" not in store


def test_service_query_detail_and_uninstall_requests_follow_lifecycle_state():
    source = (REPO_ROOT / "frontend/src/views/install/SvcQuery.vue").read_text(
        encoding="utf-8"
    )

    assert "setInterval(" not in source
    assert "this.serviceListController?.abort()" in source
    assert "this.serviceInfoController?.abort()" in source
    assert "this.selected_service !== service" in source
    assert "this.serviceInfoTimer = window.setTimeout" in source
    assert "void this.sendRequest(service)" in source
    assert "this.stopServiceInfoPolling()" in source
    assert "if (!this.runtimeReady)" in source
    assert ':loading="install_state.isUninstalling"' in source
    assert ':disabled="!install_state.canUninstall"' in source
    assert "this.install_state.beginUninstall()" in source
    assert "JSON.stringify({ install_id: targetInstallId })" in source
    assert "classifyStopResponse" in source
    assert "while (!commandObserved && !this.componentUnmounted)" in source
    assert "STOP_RETRY_INTERVAL_MS" in source
    assert "waitUntilUninstallCommandObserved" in source
    assert "markUninstallCommandObserved" in source
    assert "this.uninstallCommandController?.abort()" in source
    assert "phase === 'cancelling-install'" not in source
    assert "await this.install_state.waitUntilUninstallCompletes(actionId)" in source
    assert "this.install_state.finishUninstall(actionId)" in source
    assert 'v-if="install_state.cleanup"' in source
    assert "Cleanup is taking longer than expected" in source
    assert "Installation remains unavailable until cleanup is complete" in source
    assert "install_state.cleanup.blocking_objects" in source
    assert "this.install_state.detachUninstallWaiter()" in source
    assert "'Installation cancelled successfully'" in source
    assert "'Uninstall services successfully'" in source
    assert "setPhase('uninstalling')" not in source
    assert "runtimeGeneration" in source
    assert "deriveRuntimeDetailTransition(previousGeneration, generation)" in source


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
    lifecycle = (REPO_ROOT / "frontend/src/stores/installLifecycle.ts").read_text(
        encoding="utf-8"
    )
    store = (REPO_ROOT / "frontend/src/stores/installState.ts").read_text(
        encoding="utf-8"
    )
    install = (REPO_ROOT / "frontend/src/views/install/SvcInstall.vue").read_text(
        encoding="utf-8"
    )

    assert "const hasSession = input.status === 'install'" in lifecycle
    assert "input.ready && input.phase === 'active'" in lifecycle
    assert "(hasSession || input.installPending) && !isUninstalling" in lifecycle
    assert "canInstall: input.hydrated" in lifecycle
    assert "INITIAL_INSTALL_PHASES" in lifecycle
    assert "UNINSTALL_PHASES" in lifecycle
    assert "cancelling-install" in lifecycle
    assert "preparing-uninstall" in lifecycle
    assert "localInstallWaitingForAdmission" in lifecycle
    assert "classifyStopResponse" in lifecycle
    assert "operationId" in store
    assert "installId" in store
    assert "createInstallId" in store
    assert "hasInstallIdentity" in store
    assert "serverInstallPending" in store
    assert "lastError" in store
    assert "cleanupDelayed" in store
    assert "snapshot.cleanup" in store
    assert "normalizeCleanup" in store
    assert "truncated_count" in store
    assert "installCancelRequested" in store
    assert "reconcileInstallAcceptance" in store
    assert "reconcileUninstallCommand" in store
    assert ':loading="install_state.isInstalling"' in install
    assert ':disabled="!install_state.canInstall"' in install
    assert "Promise.race([commandResult, lifecycleResult])" in install
    assert "await this.waitForInstallCommandTail(commandResult)" in install
    assert "INSTALL_RESPONSE_SETTLE_MS" in install
    assert "await this.install_state.refresh({ fresh: true })" in install
    assert "data.warning" in install
    assert "showInstallWarning" in install
    assert "confirmInstall" not in install
    assert "data.install_state" not in install
    assert "submittedConfig" in install
    assert "install_id: actionId" in install
    assert "install_state.install()" not in install


def test_install_form_persists_a_namespace_scoped_semantic_draft():
    source = (REPO_ROOT / "frontend/src/views/install/SvcInstall.vue").read_text(
        encoding="utf-8"
    )
    draft = (
        REPO_ROOT / "frontend/src/views/install/installFormDraft.ts"
    ).read_text(encoding="utf-8")

    assert "writeInstallFormDraft" in source
    assert "install_state.namespace" in source
    assert "globalThis.localStorage" in draft
    assert "policyId" in draft
    assert "datasourceLabel" in draft
    assert "sourceId" in draft
    assert "savedInstallConfig" not in source
    assert "INSTALL_STATE_KEY" not in source
    assert "sessionStorage" not in source
    assert "sessionStorage" not in draft
    clear_method = source.split("handleClear()", 1)[1].split("},", 1)[0]
    assert "this.clearSelections()" in clear_method
    complete_method = source.split("completeInstall(message)", 1)[1].split(
        "},", 1
    )[0]
    assert "clearInstallFormDraft" not in complete_method


def test_install_lifecycle_is_initialized_once_before_mount():
    main = (REPO_ROOT / "frontend/src/main.ts").read_text(encoding="utf-8")
    install = (REPO_ROOT / "frontend/src/views/install/SvcInstall.vue").read_text(
        encoding="utf-8"
    )
    query = (REPO_ROOT / "frontend/src/views/install/SvcQuery.vue").read_text(
        encoding="utf-8"
    )
    system = (
        REPO_ROOT / "frontend/src/stores/systemParameters.ts"
    ).read_text(encoding="utf-8")

    assert main.count("installState.init()") == 1
    assert main.index("installState.init()") < main.index("app.mount('#app')")
    assert "install_state.init()" not in install
    assert "install_state.init()" not in query
    assert "installStore.init()" not in system


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
