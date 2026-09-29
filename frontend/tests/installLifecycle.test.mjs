import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import ts from 'typescript';

const sourceUrl = new URL('../src/stores/installLifecycle.ts', import.meta.url);
const source = await readFile(sourceUrl, 'utf8');
const compiled = ts.transpileModule(source, {
	compilerOptions: {
		module: ts.ModuleKind.ES2022,
		target: ts.ScriptTarget.ES2022,
	},
}).outputText;
const moduleUrl = `data:text/javascript;base64,${Buffer.from(compiled).toString('base64')}`;
const {
	classifyStopResponse,
	deriveInstallLifecycle,
	deriveRuntimeDetailTransition,
	isTargetInstallGone,
	isUninstallCommandObserved,
} = await import(moduleUrl);

const lifecycle = (overrides = {}) =>
	deriveInstallLifecycle({
		hydrated: true,
		status: 'uninstall',
		phase: 'uninstalled',
		ready: false,
		lastError: '',
		installPending: false,
		installRequestPending: false,
		uninstallRequested: false,
		...overrides,
	});

test('actions remain disabled until the first authoritative snapshot', () => {
	const state = lifecycle({ hydrated: false });
	assert.equal(state.canInstall, false);
	assert.equal(state.canUninstall, false);
});

test('uninstalled runtime permits only install', () => {
	const state = lifecycle();
	assert.equal(state.canInstall, true);
	assert.equal(state.canUninstall, false);
	assert.equal(state.isInstalling, false);
	assert.equal(state.isUninstalling, false);
});

test('local install request spins without exposing a premature cancel race', () => {
	const state = lifecycle({ installRequestPending: true });
	assert.equal(state.isInstalling, true);
	assert.equal(state.canInstall, false);
	assert.equal(state.canUninstall, false);
});

test('an unresolved local response cannot override an authoritative active session', () => {
	const state = lifecycle({
		status: 'install',
		phase: 'active',
		ready: true,
		installRequestPending: true,
	});
	assert.equal(state.isInstalling, false);
	assert.equal(state.isReady, true);
	assert.equal(state.canUninstall, true);
});

test('identity-bound server admission survives reload and permits safe cancellation', () => {
	const state = lifecycle({ phase: 'preparing-install', installPending: true });
	assert.equal(state.isInstalling, true);
	assert.equal(state.isUninstalled, false);
	assert.equal(state.canInstall, false);
	assert.equal(state.canUninstall, true);
});

test('an active directory is not exposed before backend install admission finalizes', () => {
	const state = lifecycle({
		status: 'install',
		phase: 'active',
		ready: true,
		installPending: true,
	});
	assert.equal(state.isInstalling, true);
	assert.equal(state.isReady, false);
	assert.equal(state.canUninstall, true);
});

for (const phase of ['activating-scheduler', 'activating-runtime', 'publishing']) {
	test(`${phase} permits cancellation after session ownership is durable`, () => {
		const state = lifecycle({ status: 'install', phase, installRequestPending: true });
		assert.equal(state.hasSession, true);
		assert.equal(state.isInstalling, true);
		assert.equal(state.canUninstall, true);
		assert.equal(state.isReady, false);
	});
}

test('cancel request switches loading ownership from install to uninstall', () => {
	const state = lifecycle({
		status: 'install',
		phase: 'activating-runtime',
		installRequestPending: true,
		uninstallRequested: true,
	});
	assert.equal(state.isInstalling, false);
	assert.equal(state.isUninstalling, true);
	assert.equal(state.canUninstall, false);
});

test('only active and ready session exposes runtime reads', () => {
	const state = lifecycle({ status: 'install', phase: 'active', ready: true });
	assert.equal(state.isReady, true);
	assert.equal(state.canInstall, false);
	assert.equal(state.canUninstall, true);
});

test('failed install stops install spinner but remains cleanable', () => {
	const state = lifecycle({ status: 'install', phase: 'failed', lastError: 'activation failed' });
	assert.equal(state.hasTerminalInstallFailure, true);
	assert.equal(state.isInstalling, false);
	assert.equal(state.canUninstall, true);
});

test('a non-ready owned session with an error is terminal even if its phase still says active', () => {
	const state = lifecycle({
		status: 'install',
		phase: 'active',
		ready: false,
		lastError: 'final activation failed',
		installRequestPending: true,
	});
	assert.equal(state.hasTerminalInstallFailure, true);
	assert.equal(state.isInstalling, false);
	assert.equal(state.isReady, false);
	assert.equal(state.canUninstall, true);
});

test('publication error is a cleanable terminal install failure', () => {
	const state = lifecycle({ status: 'install', phase: 'publishing', lastError: 'commit failed' });
	assert.equal(state.hasTerminalInstallFailure, true);
	assert.equal(state.isInstalling, false);
	assert.equal(state.canUninstall, true);
});

for (const phase of ['uninstalling', 'finalizing-uninstall']) {
	test(`${phase} keeps uninstall spinner active despite retry errors`, () => {
		const state = lifecycle({ status: 'install', phase, lastError: 'temporary API failure' });
		assert.equal(state.hasServerUninstallIntent, true);
		assert.equal(state.isUninstalling, true);
		assert.equal(state.canInstall, false);
		assert.equal(state.canUninstall, false);
	});
}

test('pre-session cancellation is a shared server uninstall intent', () => {
	const state = lifecycle({
		phase: 'cancelling-install',
		installPending: true,
		installRequestPending: true,
	});
	assert.equal(state.hasSession, false);
	assert.equal(state.hasServerUninstallIntent, true);
	assert.equal(state.isInstalling, false);
	assert.equal(state.isUninstalling, true);
	assert.equal(state.isCancellingInstall, true);
	assert.equal(state.canUninstall, false);
});

test('preparing-uninstall spins globally and stops duplicate command delivery', () => {
	const state = lifecycle({
		status: 'install',
		phase: 'preparing-uninstall',
		ready: false,
	});
	assert.equal(state.hasServerUninstallIntent, true);
	assert.equal(state.isUninstalling, true);
	assert.equal(state.canUninstall, false);
	assert.equal(
		isUninstallCommandObserved('preparing-uninstall', 'install-a', 'install-a', 'stop-op', 'install-op'),
		true
	);
});

test('sessionless stop admission still spins in every browser', () => {
	const state = lifecycle({
		status: 'uninstall',
		phase: 'preparing-uninstall',
		installPending: false,
	});
	assert.equal(state.hasSession, false);
	assert.equal(state.hasServerUninstallIntent, true);
	assert.equal(state.isUninstalling, true);
	assert.equal(state.canInstall, false);
	assert.equal(state.canUninstall, false);
});

test('automatic rollout is not presented as initial installation', () => {
	const state = lifecycle({ status: 'install', phase: 'publishing-rollout' });
	assert.equal(state.isInstalling, false);
	assert.equal(state.canUninstall, true);
});

test('accepted uninstall remains pending through server cleanup and unlocks only after finalization', () => {
	const accepted = lifecycle({ status: 'install', phase: 'uninstalling', uninstallRequested: true });
	assert.equal(accepted.isUninstalled, false);
	assert.equal(accepted.isUninstalling, true);
	assert.equal(accepted.canInstall, false);

	const finalizedLocally = lifecycle({ uninstallRequested: true });
	assert.equal(finalizedLocally.isUninstalled, true);
	assert.equal(finalizedLocally.isUninstalling, true);
	assert.equal(finalizedLocally.canInstall, false);

	const settled = lifecycle();
	assert.equal(settled.isUninstalled, true);
	assert.equal(settled.isUninstalling, false);
	assert.equal(settled.canInstall, true);
});

test('uninstall command observation rejects pre-command snapshots and accepts a new operation generation', () => {
	assert.equal(isUninstallCommandObserved('active', 'install-a', 'install-a', 'install-op', 'install-op'), false);
	assert.equal(isUninstallCommandObserved('uninstalling', 'install-a', 'install-a', 'install-op', 'install-op'), false);
	assert.equal(isUninstallCommandObserved('uninstalling', 'install-a', 'install-a', 'stop-op', 'install-op'), true);
	assert.equal(isUninstallCommandObserved('uninstalled', '', 'install-a', '', 'install-op'), true);
});

test('cancelling-install accepts the same target without requiring a Session operation id', () => {
	assert.equal(isUninstallCommandObserved('cancelling-install', 'install-a', 'install-a', '', ''), true);
});

test('Backend failures and permanent gateway 4xx reject stop; uncertain responses remain retryable', () => {
	assert.equal(classifyStopResponse(true, 'success'), 'accepted');
	assert.equal(classifyStopResponse(true, 'fail'), 'rejected');
	assert.equal(classifyStopResponse(false, 'fail', 503), 'unknown');
	assert.equal(classifyStopResponse(false, undefined), 'unknown');
	assert.equal(classifyStopResponse(true, 'unexpected'), 'unknown');
	for (const status of [400, 401, 403, 404, 405, 422]) {
		assert.equal(classifyStopResponse(false, undefined, status), 'rejected');
	}
	for (const status of [408, 409, 425, 429, 500, 503]) {
		assert.equal(classifyStopResponse(false, undefined, status), 'unknown');
	}
});

test('an active A to active B replacement reloads service details despite readiness staying true', () => {
	assert.deepEqual(deriveRuntimeDetailTransition('install-a', 'install-b'), { clear: true, load: true });
	assert.deepEqual(deriveRuntimeDetailTransition('install-a', ''), { clear: true, load: false });
	assert.deepEqual(deriveRuntimeDetailTransition('install-b', 'install-b'), { clear: false, load: false });
});

test('uninstall completion remains bound to the target install across an immediate replacement', () => {
	assert.equal(isTargetInstallGone('install-a', 'install-a'), false);
	assert.equal(isTargetInstallGone('', 'install-a'), true);
	assert.equal(isTargetInstallGone('install-b', 'install-a'), true);
});
