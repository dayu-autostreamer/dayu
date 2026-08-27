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

const installStoreStub = dataUrl(`
	export const useInstallStateStore = () => ({
		installId: '',
		isReady: false,
		$subscribe() {},
	});
`);
let source = compile(await readFile(new URL('../src/stores/systemParameters.ts', import.meta.url), 'utf8'));
source = source
	.replace("from 'pinia'", `from '${import.meta.resolve('pinia')}'`)
	.replace("from 'vue'", `from '${import.meta.resolve('vue')}'`)
	.replace("from '/@/stores/installState'", `from '${installStoreStub}'`);
const { useSystemParametersStore } = await import(dataUrl(source));

globalThis.window = globalThis;

const createStore = () => {
	setActivePinia(createPinia());
	return useSystemParametersStore();
};

test('losing runtime readiness clears history even before the install id disappears', () => {
	const store = createStore();
	store.runtimeInstallId = 'install-a';
	store.bufferedTaskCache.push({ timestamp: 1, data: [] });

	store.syncRuntimeGeneration('install-a', false);
	assert.deepEqual(store.bufferedTaskCache, []);

	store.bufferedTaskCache.push({ timestamp: 2, data: [] });
	store.syncRuntimeGeneration('install-b', false);
	assert.equal(store.runtimeInstallId, 'install-b');
	assert.deepEqual(store.bufferedTaskCache, []);
});

test('uninstall clears the previous runtime generation buffer', () => {
	const store = createStore();
	store.runtimeInstallId = 'install-a';
	store.bufferedTaskCache.push({ timestamp: 1, data: [] });

	store.syncRuntimeGeneration('', false);
	assert.equal(store.runtimeInstallId, '');
	assert.deepEqual(store.bufferedTaskCache, []);
});

test('runtime replacement fences an in-flight sample from the old install id', () => {
	const store = createStore();
	let aborted = false;
	store.runtimeInstallId = 'install-a';
	store.pollingActive = true;
	store.pollingGeneration = 4;
	store.requestController = {
		abort() {
			aborted = true;
		},
	};

	store.syncRuntimeGeneration('install-b', false);
	assert.equal(aborted, true);
	assert.equal(store.pollingActive, false);
	assert.equal(store.pollingGeneration, 5);
	assert.equal(store.requestController, null);
});
