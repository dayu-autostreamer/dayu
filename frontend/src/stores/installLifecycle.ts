export const INITIAL_INSTALL_PHASES = new Set(['activating-scheduler', 'activating-runtime', 'publishing']);
export const UNINSTALL_PHASES = new Set([
	'preparing-uninstall',
	'cancelling-install',
	'uninstalling',
	'finalizing-uninstall',
]);

export type InstallStatus = 'install' | 'uninstall';

export type InstallLifecycleInput = {
	hydrated: boolean;
	status: InstallStatus;
	phase: string;
	ready: boolean;
	lastError: string;
	installPending: boolean;
	installRequestPending: boolean;
	uninstallRequested: boolean;
};

export type InstallLifecycle = {
	hasSession: boolean;
	hasServerUninstallIntent: boolean;
	isUninstalled: boolean;
	isInstalling: boolean;
	isUninstalling: boolean;
	isCancellingInstall: boolean;
	isReady: boolean;
	canInstall: boolean;
	canUninstall: boolean;
	hasTerminalInstallFailure: boolean;
};

export function isUninstallCommandObserved(
	phase: string,
	installId: string,
	targetInstallId: string,
	operationId: string,
	baselineOperationId: string
) {
	if (isTargetInstallGone(installId, targetInstallId)) return true;
	if (installId === targetInstallId && phase === 'cancelling-install') return true;
	return (
		installId === targetInstallId &&
		UNINSTALL_PHASES.has(phase) &&
		Boolean(operationId) &&
		operationId !== baselineOperationId
	);
}

export function isTargetInstallGone(installId: string, targetInstallId: string) {
	return Boolean(targetInstallId) && installId !== targetInstallId;
}

export type StopResponseDisposition = 'accepted' | 'rejected' | 'unknown';

export function classifyStopResponse(responseOk: boolean, state: unknown, status = 0): StopResponseDisposition {
	if (responseOk && state === 'success') return 'accepted';
	if (responseOk && state === 'fail') return 'rejected';
	// A gateway/authentication rejection will not become successful by replaying
	// the same command. Timeouts, conflicts, rate limits and server failures are
	// uncertain and remain target-bound retry cases.
	if (status >= 400 && status < 500 && ![408, 409, 425, 429].includes(status)) return 'rejected';
	return 'unknown';
}

export function deriveRuntimeDetailTransition(previousInstallId: string, nextInstallId: string) {
	const changed = previousInstallId !== nextInstallId;
	return {
		clear: changed,
		load: changed && Boolean(nextInstallId),
	};
}

export function deriveInstallLifecycle(input: InstallLifecycleInput): InstallLifecycle {
	const hasSession = input.status === 'install';
	// Stop admission is itself a namespace-wide server fact.  In the narrow
	// gap after a cancelled install has released its admission but before
	// begin_uninstall() publishes (or confirms the absence of) a Session, the
	// Backend legitimately reports ``uninstall/preparing-uninstall`` without
	// install_pending.  Every browser must still project the same spinner.
	const serverUninstalling = UNINSTALL_PHASES.has(input.phase);
	const isUninstalling = input.uninstallRequested || serverUninstalling;
	const isCancellingInstall = input.phase === 'cancelling-install';
	const hasTerminalInstallFailure =
		hasSession && !serverUninstalling && (input.phase === 'failed' || (!input.ready && Boolean(input.lastError)));
	const serverInstalling =
		!serverUninstalling &&
		!hasTerminalInstallFailure &&
		(input.installPending || (hasSession && INITIAL_INSTALL_PHASES.has(input.phase)));
	// A local POST may provide immediate feedback only before the Backend has
	// exposed any owned lifecycle object. Once a server identity exists, its
	// phase is the sole source of global install/uninstall semantics.
	const localInstallWaitingForAdmission =
		input.installRequestPending &&
		input.status === 'uninstall' &&
		input.phase === 'uninstalled' &&
		!input.installPending;
	const isInstalling = !isUninstalling && (serverInstalling || localInstallWaitingForAdmission);
	const isReady = !isUninstalling && !input.installPending && hasSession && input.ready && input.phase === 'active';
	const isUninstalled = input.status === 'uninstall' && input.phase === 'uninstalled' && !input.installPending;

	return {
		hasSession,
		hasServerUninstallIntent: serverUninstalling,
		isUninstalled,
		isInstalling,
		isUninstalling,
		isCancellingInstall,
		isReady,
		canInstall: input.hydrated && isUninstalled && !input.installRequestPending && !input.uninstallRequested,
		// Cancellation becomes safe as soon as the Backend exposes its
		// identity-bound admission token; a merely local POST is not enough.
		canUninstall: input.hydrated && (hasSession || input.installPending) && !isUninstalling,
		hasTerminalInstallFailure,
	};
}
