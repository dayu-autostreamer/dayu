import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import ts from 'typescript';

const source = await readFile(new URL('../src/views/install/installFormDraft.ts', import.meta.url), 'utf8');
const compiled = ts.transpileModule(source, {
	compilerOptions: {
		module: ts.ModuleKind.ES2022,
		target: ts.ScriptTarget.ES2022,
	},
}).outputText;
const moduleUrl = `data:text/javascript;base64,${Buffer.from(compiled).toString('base64')}`;
const {
	clearInstallFormDraft,
	createInstallFormDraft,
	installFormDraftKey,
	readInstallFormDraft,
	restoreInstallFormDraft,
	writeInstallFormDraft,
} = await import(moduleUrl);

class MemoryStorage {
	items = new Map();

	get length() {
		return this.items.size;
	}

	clear() {
		this.items.clear();
	}

	getItem(key) {
		return this.items.has(key) ? this.items.get(key) : null;
	}

	key(index) {
		return [...this.items.keys()][index] ?? null;
	}

	removeItem(key) {
		this.items.delete(key);
	}

	setItem(key, value) {
		this.items.set(key, String(value));
	}
}

const catalogs = (overrides = {}) => ({
	policies: [],
	datasources: [],
	dags: [],
	nodes: [],
	...overrides,
});

test('the reusable form draft is namespace-scoped and stores only semantic identifiers', () => {
	const storage = new MemoryStorage();
	const draft = createInstallFormDraft(
		{ policy_id: 'fixed', name: 'Fixed Policy' },
		{ source_label: 'street-cameras', secret: 'not-for-storage' },
		[
			{
				id: 7,
				name: 'camera-7',
				url: 'https://example.invalid/private-stream',
				dag_selected: 3,
				node_selected: ['edge-a', 'edge-a', 'edge-b'],
			},
		]
	);

	assert.deepEqual(draft, {
		version: 1,
		policyId: 'fixed',
		datasourceLabel: 'street-cameras',
		mappings: [{ sourceId: 7, dagId: 3, nodeNames: ['edge-a', 'edge-b'] }],
	});
	assert.equal(writeInstallFormDraft('dayu/a', draft, storage), true);
	assert.deepEqual(readInstallFormDraft('dayu/a', storage), draft);
	assert.equal(readInstallFormDraft('dayu/b', storage), null);
	assert.equal(storage.getItem(installFormDraftKey('dayu/a')).includes('private-stream'), false);
	assert.equal(storage.getItem(installFormDraftKey('dayu/a')).includes('not-for-storage'), false);

	writeInstallFormDraft('dayu/b', { ...draft, policyId: 'hedger' }, storage);
	assert.equal(clearInstallFormDraft('dayu/a', storage), true);
	assert.equal(readInstallFormDraft('dayu/a', storage), null);
	assert.equal(readInstallFormDraft('dayu/b', storage).policyId, 'hedger');
});

test('restoration follows stable ids after every catalog is reordered', () => {
	const draft = {
		version: 1,
		policyId: 'fixed',
		datasourceLabel: 'street-cameras',
		mappings: [
			{ sourceId: 1, dagId: 10, nodeNames: ['edge-a', 'removed-node'] },
			{ sourceId: 2, dagId: 20, nodeNames: ['edge-b'] },
			{ sourceId: 99, dagId: 10, nodeNames: ['edge-a'] },
		],
	};
	const restored = restoreInstallFormDraft(
		draft,
		catalogs({
			policies: [{ policy_id: 'hedger' }, { policy_id: 'fixed' }],
			datasources: [
				{ source_label: 'other', source_list: [] },
				{
					source_label: 'street-cameras',
					source_list: [
						{ id: 2, name: 'camera-2-current', url: 'http://current/2' },
						{ id: 1, name: 'camera-1-current', url: 'http://current/1' },
					],
				},
			],
			dags: [{ dag_id: 20 }, { dag_id: 10 }],
			nodes: [{ name: 'edge-b' }, { name: 'edge-a' }],
		})
	);

	assert.equal(restored.policyIndex, 1);
	assert.equal(restored.datasourceIndex, 1);
	assert.deepEqual(restored.sources, [
		{
			id: 2,
			name: 'camera-2-current',
			url: 'http://current/2',
			dag_selected: 20,
			node_selected: ['edge-b'],
		},
		{
			id: 1,
			name: 'camera-1-current',
			url: 'http://current/1',
			dag_selected: 10,
			node_selected: ['edge-a'],
		},
	]);
});

test('restoration prunes deleted choices and initializes newly added sources', () => {
	const draft = {
		version: 1,
		policyId: 'deleted-policy',
		datasourceLabel: 'street-cameras',
		mappings: [{ sourceId: 1, dagId: 10, nodeNames: ['removed-node'] }],
	};
	const currentCatalogs = catalogs({
		policies: [{ policy_id: 'fixed' }],
		datasources: [
			{
				source_label: 'street-cameras',
				source_list: [{ id: 1 }, { id: 2, name: 'new-camera' }],
			},
		],
		dags: [{ dag_id: 20 }],
		nodes: [{ name: 'edge-a' }],
	});

	assert.deepEqual(restoreInstallFormDraft(draft, currentCatalogs), {
		policyIndex: null,
		datasourceIndex: 0,
		sources: [
			{ id: 1, dag_selected: '', node_selected: [] },
			{ id: 2, name: 'new-camera', dag_selected: '', node_selected: [] },
		],
	});
	assert.deepEqual(restoreInstallFormDraft({ ...draft, datasourceLabel: 'deleted-datasource' }, currentCatalogs), {
		policyIndex: null,
		datasourceIndex: null,
		sources: [],
	});
});

test('corrupt, incompatible, and unavailable browser storage degrades to an empty draft', () => {
	const storage = new MemoryStorage();
	const key = installFormDraftKey('dayu-test');
	storage.setItem(key, '{broken');
	assert.equal(readInstallFormDraft('dayu-test', storage), null);
	assert.equal(storage.getItem(key), null);

	storage.setItem(key, JSON.stringify({ version: 2, policyId: 'fixed', datasourceLabel: '', mappings: [] }));
	assert.equal(readInstallFormDraft('dayu-test', storage), null);
	assert.equal(storage.getItem(key), null);

	const unavailable = {
		getItem() {
			throw new Error('storage denied');
		},
		removeItem() {
			throw new Error('storage denied');
		},
		setItem() {
			throw new Error('storage denied');
		},
	};
	assert.equal(readInstallFormDraft('dayu-test', unavailable), null);
	assert.equal(
		writeInstallFormDraft('dayu-test', { version: 1, policyId: '', datasourceLabel: '', mappings: [] }, unavailable),
		false
	);
});

test('an entirely empty form removes the stored preference', () => {
	const storage = new MemoryStorage();
	const draft = { version: 1, policyId: 'fixed', datasourceLabel: '', mappings: [] };
	writeInstallFormDraft('dayu-test', draft, storage);
	assert.equal(createInstallFormDraft(null, null, []), null);
	writeInstallFormDraft('dayu-test', createInstallFormDraft(null, null, []), storage);
	assert.equal(readInstallFormDraft('dayu-test', storage), null);
});
