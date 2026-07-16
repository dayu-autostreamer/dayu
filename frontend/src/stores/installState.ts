import { computed, ref, watch } from 'vue';
import { defineStore } from 'pinia';
import {
	deriveInstallLifecycle,
	isUninstallCommandObserved,
	isTargetInstallGone,
	UNINSTALL_PHASES,
	type InstallStatus,
} from '/@/stores/installLifecycle';
import { fetchJsonWithTimeout } from '/@/utils/fetchWithTimeout';

type InstallStateSnapshot = {
	state: InstallStatus;
	phase?: string;
	ready?: boolean;
	install_id?: string;
	install_pending?: boolean;
	operation_id?: string;
	last_error?: string;
	cleanup?: unknown;
};

export type CleanupObject = {
	kind: string;
	name: string;
	uid: string;
	node?: string;
	deletion_timestamp?: string;
	finalizers?: string[];
};

export type CleanupDiagnostics = {
	status: 'progressing' | 'delayed';
	started_at: string;
	last_progress_at: string;
	seconds_without_progress: number;
	warning_after_seconds: number;
	remaining_count: number;
	remaining_by_kind: Record<string, number>;
	affected_nodes: string[];
	blocking_objects: CleanupObject[];
	truncated_count: number;
};

export type InstallCompletion = 'active' | 'cancelled' | 'failed' | 'unknown';

const ACTIVE_POLL_INTERVAL_MS = 1000;
const IDLE_POLL_INTERVAL_MS = 3000;
const STATE_REQUEST_TIMEOUT_MS = 10000;
const INSTALL_ACCEPTANCE_WAIT_MS = 15000;
const INSTALL_ACCEPTANCE_RETRY_MS = 1000;
const INSTALL_SETTLE_TIMEOUT_MS = 930000;
const UNINSTALL_ACCEPTANCE_WAIT_MS = 5000;
const UNINSTALL_ACCEPTANCE_RETRY_MS = 250;

function isAbortError(error: unknown) {
	return error instanceof Error && error.name === 'AbortError';
}

function nonNegativeInteger(value: unknown, fallback = 0) {
	return typeof value === 'number' && Number.isFinite(value) && value >= 0 ? Math.floor(value) : fallback;
}

function stringArray(value: unknown) {
	return Array.isArray(value) ? value.map((item) => String(item || '')).filter(Boolean) : [];
}

function normalizeCleanup(value: unknown): CleanupDiagnostics | null {
	if (!value || typeof value !== 'object' || Array.isArray(value)) return null;
	const raw = value as Record<string, unknown>;
	const remainingByKind: Record<string, number> = {};
	if (raw.remaining_by_kind && typeof raw.remaining_by_kind === 'object' && !Array.isArray(raw.remaining_by_kind)) {
		for (const [kind, count] of Object.entries(raw.remaining_by_kind)) {
			const normalized = nonNegativeInteger(count, -1);
			if (kind && normalized >= 0) remainingByKind[kind] = normalized;
		}
	}
	const blockingObjects = Array.isArray(raw.blocking_objects)
		? raw.blocking_objects.flatMap((value) => {
				if (!value || typeof value !== 'object' || Array.isArray(value)) return [];
				const item = value as Record<string, unknown>;
				const kind = String(item.kind || '');
				const name = String(item.name || '');
				const uid = String(item.uid || '');
				if (!kind || !name || !uid) return [];
				return [
					{
						kind,
						name,
						uid,
						node: String(item.node || ''),
						deletion_timestamp: String(item.deletion_timestamp || ''),
						finalizers: stringArray(item.finalizers),
					},
				];
			})
		: [];
	const truncatedCount = nonNegativeInteger(raw.truncated_count);
	return {
		status: raw.status === 'delayed' ? 'delayed' : 'progressing',
		started_at: String(raw.started_at || ''),
		last_progress_at: String(raw.last_progress_at || ''),
		seconds_without_progress: nonNegativeInteger(raw.seconds_without_progress),
		warning_after_seconds: nonNegativeInteger(raw.warning_after_seconds),
		remaining_count: nonNegativeInteger(raw.remaining_count, blockingObjects.length + truncatedCount),
		remaining_by_kind: remainingByKind,
		affected_nodes: stringArray(raw.affected_nodes),
		blocking_objects: blockingObjects,
		truncated_count: truncatedCount,
	};
}

export function createInstallId() {
	const cryptoApi = globalThis.crypto;
	if (typeof cryptoApi?.randomUUID === 'function') return cryptoApi.randomUUID();
	if (typeof cryptoApi?.getRandomValues !== 'function') {
		throw new Error('A cryptographic random source is required to install services');
	}
	const bytes = cryptoApi.getRandomValues(new Uint8Array(16));
	bytes[6] = (bytes[6] & 0x0f) | 0x40;
	bytes[8] = (bytes[8] & 0x3f) | 0x80;
	const hex = Array.from(bytes, (value) => value.toString(16).padStart(2, '0')).join('');
	return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
}

export const useInstallStateStore = defineStore('install_state', () => {
	const status = ref<InstallStatus>('uninstall');
	const phase = ref('uninstalled');
	const ready = ref(false);
	const hydrated = ref(false);
	const installId = ref('');
	const serverInstallPending = ref(false);
	const operationId = ref('');
	const lastError = ref('');
	const cleanup = ref<CleanupDiagnostics | null>(null);
	const uninstallRequested = ref(false);
	const installCancelRequested = ref(false);
	const uninstallCancelsInstall = ref(false);
	const initialized = ref(false);
	const activeInstallAction = ref('');
	const activeUninstallAction = ref(0);
	const lifecycleSnapshotSequence = ref(0);
	const uninstallCommandObserved = ref(false);
	const installRequestPending = computed(() => Boolean(activeInstallAction.value));

	const lifecycle = computed(() =>
		deriveInstallLifecycle({
			hydrated: hydrated.value,
			status: status.value,
			phase: phase.value,
			ready: ready.value,
			lastError: lastError.value,
			installPending: serverInstallPending.value,
			installRequestPending: installRequestPending.value,
			uninstallRequested: uninstallRequested.value,
		})
	);
	const hasSession = computed(() => lifecycle.value.hasSession);
	const isUninstalled = computed(() => lifecycle.value.isUninstalled);
	const isInstalling = computed(() => lifecycle.value.isInstalling);
	const isUninstalling = computed(() => lifecycle.value.isUninstalling);
	const isCancellingInstall = computed(() => lifecycle.value.isCancellingInstall);
	const isReady = computed(() => lifecycle.value.isReady);
	const canInstall = computed(() => lifecycle.value.canInstall);
	const canUninstall = computed(() => lifecycle.value.canUninstall);
	const hasTerminalInstallFailure = computed(() => lifecycle.value.hasTerminalInstallFailure);
	const cleanupDelayed = computed(() => cleanup.value?.status === 'delayed');

	let uninstallActionSequence = 0;
	let pollingActive = false;
	let pollingTimer: number | null = null;
	let stateController: AbortController | null = null;
	let stateRequest: Promise<InstallStateSnapshot> | null = null;
	const uninstallTargetInstallId = ref('');
	let uninstallBaselineOperationId = '';
	let uninstallCommandObservedAt = 0;
	let uninstallCommandOperationId = '';
	let activeInstallIdentityObserved = false;

	function applySnapshot(snapshot: InstallStateSnapshot) {
		if (snapshot?.state !== 'install' && snapshot?.state !== 'uninstall') {
			throw new Error('Install state response contains an invalid state');
		}
		const nextInstallId = String(snapshot.install_id || '');
		if ((snapshot.state === 'install' || snapshot.install_pending) && !nextInstallId) {
			throw new Error('Owned install state response contains no install_id');
		}
		status.value = snapshot.state;
		phase.value = snapshot.phase || (snapshot.state === 'install' ? 'unknown' : 'uninstalled');
		ready.value = snapshot.state === 'install' && Boolean(snapshot.ready) && phase.value === 'active';
		installId.value = nextInstallId;
		serverInstallPending.value = Boolean(snapshot.install_pending);
		operationId.value = String(snapshot.operation_id || '');
		lastError.value = String(snapshot.last_error || '');
		cleanup.value = UNINSTALL_PHASES.has(phase.value) ? normalizeCleanup(snapshot.cleanup) : null;
		hydrated.value = true;
		lifecycleSnapshotSequence.value += 1;

		const actionId = activeInstallAction.value;
		if (!actionId) return;
		if (nextInstallId === actionId) {
			activeInstallIdentityObserved = true;
			// A stop issued by any window supersedes the initiating window's
			// unresolved install response. The server phase is authoritative.
			if (UNINSTALL_PHASES.has(phase.value)) {
				activeInstallAction.value = '';
				activeInstallIdentityObserved = false;
			}
			return;
		}
		if (nextInstallId || activeInstallIdentityObserved) {
			// Another installation won, or an admitted target disappeared. Do not
			// let the old local POST keep this window in a divergent state.
			activeInstallAction.value = '';
			activeInstallIdentityObserved = false;
		}
	}

	async function refresh(options: { fresh?: boolean } = {}): Promise<InstallStateSnapshot> {
		if (stateRequest) {
			if (!options.fresh) return stateRequest;
			try {
				await stateRequest;
			} catch {
				// A fresh read is still required after an older request fails.
			}
			// Multiple callers that were waiting for the same older read share the
			// first post-boundary request instead of starting parallel refreshes.
			if (stateRequest) return stateRequest;
		}
		const controller = new AbortController();
		stateController = controller;
		const request = (async () => {
			const { response, data: snapshot } = await fetchJsonWithTimeout<InstallStateSnapshot>(
				'/api/install_state',
				{},
				STATE_REQUEST_TIMEOUT_MS,
				controller
			);
			if (!response.ok) throw new Error(`Install state request failed: ${response.status}`);
			applySnapshot(snapshot);
			return snapshot;
		})();
		stateRequest = request;
		try {
			return await request;
		} finally {
			if (stateRequest === request) stateRequest = null;
			if (stateController === controller) stateController = null;
		}
	}

	async function poll() {
		if (!pollingActive) return;
		try {
			await refresh();
		} catch (error) {
			if (!isAbortError(error)) console.error('Fail to refresh install state', error);
		}
		if (!pollingActive) return;
		const interval = isInstalling.value || isUninstalling.value ? ACTIVE_POLL_INTERVAL_MS : IDLE_POLL_INTERVAL_MS;
		pollingTimer = window.setTimeout(poll, interval);
	}

	function init() {
		if (initialized.value) return;
		initialized.value = true;
		pollingActive = true;
		void poll();
	}

	function stopPolling() {
		pollingActive = false;
		if (pollingTimer !== null) {
			window.clearTimeout(pollingTimer);
			pollingTimer = null;
		}
		stateController?.abort();
		stateController = null;
		activeInstallAction.value = '';
		activeUninstallAction.value = 0;
		uninstallRequested.value = false;
		installCancelRequested.value = false;
		uninstallCancelsInstall.value = false;
		uninstallTargetInstallId.value = '';
		uninstallBaselineOperationId = '';
		uninstallCommandObserved.value = false;
		uninstallCommandObservedAt = 0;
		uninstallCommandOperationId = '';
		activeInstallIdentityObserved = false;
		cleanup.value = null;
		initialized.value = false;
	}

	function beginInstall(): string | null {
		if (!canInstall.value) return null;
		init();
		activeInstallAction.value = createInstallId();
		activeInstallIdentityObserved = false;
		installCancelRequested.value = false;
		return activeInstallAction.value;
	}

	function isCurrentInstallAction(actionId: string) {
		return Boolean(actionId) && actionId === activeInstallAction.value;
	}

	function shouldIgnoreInstallResult(actionId: string) {
		return !isCurrentInstallAction(actionId) || installCancelRequested.value;
	}

	function finishInstall(actionId: string) {
		if (!isCurrentInstallAction(actionId)) return;
		activeInstallAction.value = '';
		activeInstallIdentityObserved = false;
		installCancelRequested.value = false;
	}

	function hasInstallIdentity(actionId: string) {
		return isCurrentInstallAction(actionId) && installId.value === actionId;
	}

	async function reconcileInstallAcceptance(actionId: string) {
		const deadline = Date.now() + INSTALL_ACCEPTANCE_WAIT_MS;
		while (isCurrentInstallAction(actionId)) {
			try {
				await refresh({ fresh: true });
			} catch (error) {
				if (!isAbortError(error)) console.error('Fail to reconcile install acceptance', error);
			}
			if (hasInstallIdentity(actionId) && (hasSession.value || serverInstallPending.value)) return true;
			if (hydrated.value && installId.value && installId.value !== actionId) return false;
			if (Date.now() >= deadline) return false;
			await new Promise((resolve) => window.setTimeout(resolve, INSTALL_ACCEPTANCE_RETRY_MS));
		}
		return false;
	}

	function beginUninstall(): number | null {
		if (!canUninstall.value) return null;
		init();
		activeUninstallAction.value = ++uninstallActionSequence;
		uninstallTargetInstallId.value = installId.value;
		uninstallBaselineOperationId = operationId.value;
		uninstallCommandObserved.value = false;
		uninstallCommandObservedAt = 0;
		uninstallCommandOperationId = '';
		uninstallCancelsInstall.value = isInstalling.value;
		if (activeInstallAction.value === uninstallTargetInstallId.value) installCancelRequested.value = true;
		uninstallRequested.value = true;
		return activeUninstallAction.value;
	}

	function isCurrentUninstallAction(actionId: number) {
		return actionId !== 0 && actionId === activeUninstallAction.value;
	}

	function rejectUninstall(actionId: number) {
		if (!isCurrentUninstallAction(actionId)) return;
		activeUninstallAction.value = 0;
		uninstallTargetInstallId.value = '';
		uninstallBaselineOperationId = '';
		uninstallCommandObserved.value = false;
		uninstallCommandObservedAt = 0;
		uninstallCommandOperationId = '';
		uninstallRequested.value = false;
		uninstallCancelsInstall.value = false;
		installCancelRequested.value = false;
	}

	function detachUninstallWaiter() {
		if (activeUninstallAction.value) rejectUninstall(activeUninstallAction.value);
	}

	function hasUninstallCompleted(actionId: number) {
		return isCurrentUninstallAction(actionId) && isTargetInstallGone(installId.value, uninstallTargetInstallId.value);
	}

	function finishUninstall(actionId: number) {
		if (!hasUninstallCompleted(actionId)) return;
		activeUninstallAction.value = 0;
		const completedInstallId = uninstallTargetInstallId.value;
		uninstallTargetInstallId.value = '';
		uninstallBaselineOperationId = '';
		uninstallCommandObserved.value = false;
		uninstallCommandObservedAt = 0;
		uninstallCommandOperationId = '';
		uninstallRequested.value = false;
		uninstallCancelsInstall.value = false;
		// Target identity disappearance proves the server-side transaction has
		// yielded to stop, even if another client already installed a replacement.
		// Any eventual response from the stopped install request is now stale.
		if (activeInstallAction.value === completedInstallId) activeInstallAction.value = '';
		activeInstallIdentityObserved = false;
		installCancelRequested.value = false;
	}

	function waitUntilInstallSettles(actionId: string): Promise<InstallCompletion> {
		const completion = (): InstallCompletion | null => {
			if (shouldIgnoreInstallResult(actionId)) return 'cancelled';
			if (hasInstallIdentity(actionId) && isReady.value) return 'active';
			if (hasInstallIdentity(actionId) && hasTerminalInstallFailure.value) return 'failed';
			return null;
		};
		const current = completion();
		if (current) return Promise.resolve(current);
		return new Promise((resolve) => {
			const timer = window.setTimeout(() => {
				stop();
				resolve('unknown');
			}, INSTALL_SETTLE_TIMEOUT_MS);
			const stop = watch(
				[
					isReady,
					hasTerminalInstallFailure,
					installId,
					serverInstallPending,
					installCancelRequested,
					activeInstallAction,
				],
				() => {
					const result = completion();
					if (!result) return;
					window.clearTimeout(timer);
					stop();
					resolve(result);
				}
			);
		});
	}

	function markUninstallCommandObserved(actionId: number) {
		if (!isCurrentUninstallAction(actionId)) return false;
		if (!uninstallCommandObserved.value) {
			uninstallCommandObserved.value = true;
			uninstallCommandObservedAt = lifecycleSnapshotSequence.value;
			uninstallCommandOperationId = operationId.value;
		}
		return true;
	}

	function hasObservedUninstallCommand(actionId: number) {
		if (!isCurrentUninstallAction(actionId)) return false;
		if (uninstallCommandObserved.value) return true;
		const observed = isUninstallCommandObserved(
			phase.value,
			installId.value,
			uninstallTargetInstallId.value,
			operationId.value,
			uninstallBaselineOperationId
		);
		return observed ? markUninstallCommandObserved(actionId) : false;
	}

	function waitUntilUninstallCommandObserved(actionId: number): Promise<boolean> {
		if (!isCurrentUninstallAction(actionId)) return Promise.resolve(false);
		if (hasObservedUninstallCommand(actionId)) return Promise.resolve(true);
		return new Promise((resolve) => {
			const stop = watch([lifecycleSnapshotSequence, activeUninstallAction], () => {
				if (!isCurrentUninstallAction(actionId) || hasObservedUninstallCommand(actionId)) {
					const observed = hasObservedUninstallCommand(actionId);
					stop();
					resolve(observed);
				}
			});
		});
	}

	async function reconcileUninstallCommand(actionId: number) {
		const deadline = Date.now() + UNINSTALL_ACCEPTANCE_WAIT_MS;
		while (isCurrentUninstallAction(actionId)) {
			try {
				await refresh({ fresh: true });
			} catch (error) {
				if (!isAbortError(error)) console.error('Fail to reconcile uninstall command delivery', error);
			}
			if (hasObservedUninstallCommand(actionId)) return true;
			if (Date.now() >= deadline) return false;
			await new Promise((resolve) => window.setTimeout(resolve, UNINSTALL_ACCEPTANCE_RETRY_MS));
		}
		return false;
	}

	function waitUntilUninstallCompletes(actionId: number): Promise<boolean> {
		const outcome = (): boolean | null => {
			if (!isCurrentUninstallAction(actionId)) return false;
			if (hasUninstallCompleted(actionId)) return true;
			const commandReverted =
				uninstallCommandObserved.value &&
				lifecycleSnapshotSequence.value > uninstallCommandObservedAt &&
				installId.value === uninstallTargetInstallId.value &&
				!UNINSTALL_PHASES.has(phase.value) &&
				(phase.value === 'failed' ||
					Boolean(lastError.value) ||
					(Boolean(uninstallCommandOperationId) && operationId.value !== uninstallCommandOperationId));
			return commandReverted ? false : null;
		};
		const current = outcome();
		if (current !== null) return Promise.resolve(current);
		return new Promise((resolve) => {
			const stop = watch([lifecycleSnapshotSequence, activeUninstallAction], () => {
				const completed = outcome();
				if (completed === null) return;
				stop();
				resolve(completed);
			});
		});
	}

	return {
		status,
		phase,
		ready,
		hydrated,
		installId,
		uninstallTargetInstallId,
		serverInstallPending,
		operationId,
		lastError,
		cleanup,
		cleanupDelayed,
		installRequestPending,
		uninstallRequested,
		uninstallCommandObserved,
		installCancelRequested,
		uninstallCancelsInstall,
		initialized,
		hasSession,
		isUninstalled,
		isInstalling,
		isUninstalling,
		isCancellingInstall,
		isReady,
		canInstall,
		canUninstall,
		hasTerminalInstallFailure,
		refresh,
		init,
		stopPolling,
		beginInstall,
		shouldIgnoreInstallResult,
		finishInstall,
		hasInstallIdentity,
		reconcileInstallAcceptance,
		beginUninstall,
		markUninstallCommandObserved,
		rejectUninstall,
		detachUninstallWaiter,
		finishUninstall,
		waitUntilInstallSettles,
		waitUntilUninstallCommandObserved,
		reconcileUninstallCommand,
		waitUntilUninstallCompletes,
	};
});
