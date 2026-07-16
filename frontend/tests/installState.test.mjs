import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import ts from 'typescript';
import { createPinia, setActivePinia } from 'pinia';

const compile = (source) =>
	ts.transpileModule(source, {
		compilerOptions: {
			module: ts.ModuleKind.ES2022,
			target: ts.ScriptTarget.ES2022,
		},
	}).outputText;
const dataUrl = (source) => `data:text/javascript;base64,${Buffer.from(source).toString('base64')}`;

const lifecycleSource = await readFile(new URL('../src/stores/installLifecycle.ts', import.meta.url), 'utf8');
const timeoutSource = await readFile(new URL('../src/utils/fetchWithTimeout.ts', import.meta.url), 'utf8');
const lifecycleUrl = dataUrl(compile(lifecycleSource));
const timeoutUrl = dataUrl(compile(timeoutSource));
let storeSource = compile(await readFile(new URL('../src/stores/installState.ts', import.meta.url), 'utf8'));
storeSource = storeSource
	.replace("from 'vue'", `from '${import.meta.resolve('vue')}'`)
	.replace("from 'pinia'", `from '${import.meta.resolve('pinia')}'`)
	.replace("from '/@/stores/installLifecycle'", `from '${lifecycleUrl}'`)
	.replace("from '/@/utils/fetchWithTimeout'", `from '${timeoutUrl}'`);
const { createInstallId, useInstallStateStore } = await import(dataUrl(storeSource));

globalThis.window = globalThis;
const INSTALL_A = '11111111-1111-4111-8111-111111111111';
const INSTALL_B = '22222222-2222-4222-8222-222222222222';

const snapshot = (overrides = {}) => ({
	state: 'uninstall',
	phase: 'uninstalled',
	ready: false,
	install_id: '',
	install_pending: false,
	operation_id: '',
	last_error: '',
	...overrides,
});

const createStore = () => {
	setActivePinia(createPinia());
	return useInstallStateStore();
};

const response = (data) => ({
	ok: true,
	status: 200,
	json: async () => data,
});

function useSnapshots(context, values) {
	const originalFetch = globalThis.fetch;
	let index = 0;
	context.after(() => {
		globalThis.fetch = originalFetch;
	});
	globalThis.fetch = async () => {
		assert.ok(index < values.length, `unexpected install-state request ${index + 1}`);
		const value = values[index++];
		return response(typeof value === 'function' ? value() : value);
	};
	return () => index;
}

async function hydrate(store) {
	await store.refresh({ fresh: true });
	// Unit tests drive snapshots explicitly; suppress the background poll that
	// beginInstall/beginUninstall would otherwise start.
	store.initialized = true;
}

test('an authoritative active snapshot ends global install loading before the POST returns', async (context) => {
	let actionId = '';
	useSnapshots(context, [
		snapshot(),
		() =>
			snapshot({
				state: 'install',
				phase: 'active',
				ready: true,
				install_id: actionId,
				operation_id: 'install-op',
			}),
	]);
	const store = createStore();
	await hydrate(store);
	actionId = store.beginInstall();
	const completion = store.waitUntilInstallSettles(actionId);

	await store.refresh({ fresh: true });
	assert.equal(store.installRequestPending, true);
	assert.equal(store.isInstalling, false);
	assert.equal(store.isReady, true);
	assert.equal(await completion, 'active');
	store.finishInstall(actionId);
});

test('UUID generation falls back to getRandomValues on an HTTP deployment', (context) => {
	const originalDescriptor = Object.getOwnPropertyDescriptor(globalThis, 'crypto');
	context.after(() => {
		if (originalDescriptor) Object.defineProperty(globalThis, 'crypto', originalDescriptor);
		else delete globalThis.crypto;
	});
	Object.defineProperty(globalThis, 'crypto', {
		configurable: true,
		value: {
			getRandomValues(values) {
				values.fill(0);
				return values;
			},
		},
	});

	assert.equal(createInstallId(), '00000000-0000-4000-8000-000000000000');
});

test('another window installation replaces local request feedback with server state', async (context) => {
	useSnapshots(context, [
		snapshot(),
		snapshot({
			phase: 'preparing-install',
			install_id: INSTALL_B,
			install_pending: true,
		}),
	]);
	const store = createStore();
	await hydrate(store);
	const actionId = store.beginInstall();
	const completion = store.waitUntilInstallSettles(actionId);

	await store.refresh({ fresh: true });
	assert.equal(store.installId, INSTALL_B);
	assert.equal(store.installRequestPending, false);
	assert.equal(store.isInstalling, true);
	assert.equal(await completion, 'cancelled');
});

test('a projected failure ends an unresolved install action immediately', async (context) => {
	let actionId = '';
	useSnapshots(context, [
		snapshot(),
		() =>
			snapshot({
				state: 'install',
				phase: 'failed',
				install_id: actionId,
				last_error: 'activation failed',
			}),
	]);
	const store = createStore();
	await hydrate(store);
	actionId = store.beginInstall();
	const completion = store.waitUntilInstallSettles(actionId);

	await store.refresh({ fresh: true });
	assert.equal(store.isInstalling, false);
	assert.equal(store.hasTerminalInstallFailure, true);
	assert.equal(await completion, 'failed');
	store.finishInstall(actionId);
});

test('ready=false plus last_error is terminal even when the phase is active', async (context) => {
	let actionId = '';
	useSnapshots(context, [
		snapshot(),
		() =>
			snapshot({
				state: 'install',
				phase: 'active',
				ready: false,
				install_id: actionId,
				last_error: 'finalization failed',
			}),
	]);
	const store = createStore();
	await hydrate(store);
	actionId = store.beginInstall();
	const completion = store.waitUntilInstallSettles(actionId);

	await store.refresh({ fresh: true });
	assert.equal(await completion, 'failed');
	assert.equal(store.canUninstall, true);
	store.finishInstall(actionId);
});

test('another window cancellation releases the initiating window and is shared globally', async (context) => {
	let actionId = '';
	useSnapshots(context, [
		snapshot(),
		() => snapshot({ phase: 'preparing-install', install_id: actionId, install_pending: true }),
		() => snapshot({ phase: 'cancelling-install', install_id: actionId, install_pending: true }),
		snapshot(),
	]);
	const store = createStore();
	await hydrate(store);
	actionId = store.beginInstall();
	const completion = store.waitUntilInstallSettles(actionId);

	await store.refresh({ fresh: true });
	assert.equal(store.isInstalling, true);
	await store.refresh({ fresh: true });
	assert.equal(store.installRequestPending, false);
	assert.equal(store.isInstalling, false);
	assert.equal(store.isUninstalling, true);
	assert.equal(store.isCancellingInstall, true);
	assert.equal(await completion, 'cancelled');

	await store.refresh({ fresh: true });
	assert.equal(store.isUninstalled, true);
	assert.equal(store.canInstall, true);
});

test('cancelling-install confirms stop acceptance without an operation id', async (context) => {
	useSnapshots(context, [
		snapshot({ phase: 'preparing-install', install_id: INSTALL_A, install_pending: true }),
		snapshot({ phase: 'cancelling-install', install_id: INSTALL_A, install_pending: true }),
	]);
	const store = createStore();
	await hydrate(store);
	const actionId = store.beginUninstall();
	const observed = store.waitUntilUninstallCommandObserved(actionId);

	await store.refresh({ fresh: true });
	assert.equal(await observed, true);
});

test('preparing-uninstall stops duplicate command delivery before durable cleanup', async (context) => {
	useSnapshots(context, [
		snapshot({
			state: 'install',
			phase: 'active',
			ready: true,
			install_id: INSTALL_A,
			operation_id: 'install-op',
		}),
		snapshot({
			state: 'install',
			phase: 'preparing-uninstall',
			install_id: INSTALL_A,
			operation_id: 'stop-op',
		}),
		snapshot({
			state: 'install',
			phase: 'uninstalling',
			install_id: INSTALL_A,
			operation_id: 'stop-op',
		}),
	]);
	const store = createStore();
	await hydrate(store);
	const actionId = store.beginUninstall();
	let resolved = false;
	const observed = store.waitUntilUninstallCommandObserved(actionId).then((value) => {
		resolved = true;
		return value;
	});

	await store.refresh({ fresh: true });
	await new Promise((resolve) => setImmediate(resolve));
	assert.equal(store.isUninstalling, true);
	assert.equal(store.canUninstall, false);
	assert.equal(resolved, true);
	assert.equal(await observed, true);
});

test('a stop admission that reverts to a failed target releases the uninstall action', async (context) => {
	useSnapshots(context, [
		snapshot({
			state: 'install',
			phase: 'active',
			ready: true,
			install_id: INSTALL_A,
			operation_id: 'install-op',
		}),
		snapshot({
			state: 'install',
			phase: 'preparing-uninstall',
			install_id: INSTALL_A,
			operation_id: 'stop-op',
		}),
		snapshot({
			state: 'install',
			phase: 'failed',
			install_id: INSTALL_A,
			operation_id: 'install-op',
			last_error: 'session CAS failed',
		}),
	]);
	const store = createStore();
	await hydrate(store);
	const actionId = store.beginUninstall();
	const observed = store.waitUntilUninstallCommandObserved(actionId);
	const completion = store.waitUntilUninstallCompletes(actionId);

	await store.refresh({ fresh: true });
	assert.equal(await observed, true);
	await store.refresh({ fresh: true });
	assert.equal(await completion, false);
	store.rejectUninstall(actionId);
	assert.equal(store.canUninstall, true);
});

test('uninstall completion follows the target across an immediate replacement session', async (context) => {
	useSnapshots(context, [
		snapshot({
			state: 'install',
			phase: 'active',
			ready: true,
			install_id: INSTALL_A,
			operation_id: 'install-op',
		}),
		snapshot({
			state: 'install',
			phase: 'active',
			ready: true,
			install_id: INSTALL_B,
			operation_id: 'replacement-op',
		}),
	]);
	const store = createStore();
	await hydrate(store);
	const actionId = store.beginUninstall();
	const completion = store.waitUntilUninstallCompletes(actionId);

	await store.refresh({ fresh: true });
	assert.equal(await completion, true);
	store.finishUninstall(actionId);
	assert.equal(store.isUninstalling, false);
	assert.equal(store.isReady, true);
});

test('cancel completion is not delayed by a replacement pending install', async (context) => {
	useSnapshots(context, [
		snapshot({
			state: 'install',
			phase: 'activating-runtime',
			install_id: INSTALL_A,
			install_pending: true,
			operation_id: 'install-op',
		}),
		snapshot({ phase: 'preparing-install', install_id: INSTALL_B, install_pending: true }),
	]);
	const store = createStore();
	await hydrate(store);
	const actionId = store.beginUninstall();
	const completion = store.waitUntilUninstallCompletes(actionId);

	await store.refresh({ fresh: true });
	assert.equal(await completion, true);
	store.finishUninstall(actionId);
	assert.equal(store.isInstalling, true);
});

test('cancel install waits for both session cleanup and install-handler exit', async (context) => {
	useSnapshots(context, [
		snapshot({
			state: 'install',
			phase: 'activating-runtime',
			install_id: INSTALL_A,
			install_pending: true,
			operation_id: 'install-op',
		}),
		snapshot({ phase: 'preparing-install', install_id: INSTALL_A, install_pending: true }),
		snapshot(),
	]);
	const store = createStore();
	await hydrate(store);
	const actionId = store.beginUninstall();
	let resolved = false;
	const completion = store.waitUntilUninstallCompletes(actionId).then((value) => {
		resolved = true;
		return value;
	});

	await store.refresh({ fresh: true });
	await new Promise((resolve) => setImmediate(resolve));
	assert.equal(resolved, false);

	await store.refresh({ fresh: true });
	assert.equal(await completion, true);
	store.finishUninstall(actionId);
	assert.equal(store.isUninstalling, false);
});

test('invalidating an install action releases its lifecycle waiter', async (context) => {
	useSnapshots(context, [snapshot()]);
	const store = createStore();
	await hydrate(store);
	const actionId = store.beginInstall();
	const completion = store.waitUntilInstallSettles(actionId);

	store.stopPolling();
	assert.equal(await completion, 'cancelled');
});

test('fresh refresh waits for the older single-flight request before reading again', async (context) => {
	const store = createStore();
	const originalFetch = globalThis.fetch;
	const pending = [];
	context.after(() => {
		globalThis.fetch = originalFetch;
		store.stopPolling();
	});
	globalThis.fetch = (_input, init) =>
		new Promise((resolve, reject) => {
			pending.push({ resolve });
			init.signal.addEventListener(
				'abort',
				() => {
					const error = new Error('aborted');
					error.name = 'AbortError';
					reject(error);
				},
				{ once: true }
			);
		});

	const older = store.refresh();
	const fresh = store.refresh({ fresh: true });
	assert.equal(pending.length, 1);
	pending[0].resolve(
		response(
			snapshot({
				state: 'install',
				phase: 'active',
				ready: true,
				install_id: INSTALL_A,
				operation_id: 'install-op',
			})
		)
	);
	await older;
	for (let attempt = 0; attempt < 10 && pending.length < 2; attempt += 1) {
		await new Promise((resolve) => setImmediate(resolve));
	}
	assert.equal(pending.length, 2);
	pending[1].resolve(response(snapshot()));
	await fresh;

	assert.equal(store.isUninstalled, true);
	assert.equal(store.installId, '');
});
