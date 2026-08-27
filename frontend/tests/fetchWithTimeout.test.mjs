import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import ts from 'typescript';

const sourceUrl = new URL('../src/utils/fetchWithTimeout.ts', import.meta.url);
const source = await readFile(sourceUrl, 'utf8');
const compiled = ts.transpileModule(source, {
	compilerOptions: {
		module: ts.ModuleKind.ES2022,
		target: ts.ScriptTarget.ES2022,
	},
}).outputText;
const moduleUrl = `data:text/javascript;base64,${Buffer.from(compiled).toString('base64')}`;
const { fetchJsonWithTimeout } = await import(moduleUrl);

globalThis.window = globalThis;

test('request timeout aborts a fetch that never returns headers', async (context) => {
	const originalFetch = globalThis.fetch;
	context.after(() => {
		globalThis.fetch = originalFetch;
	});
	globalThis.fetch = (_input, init) =>
		new Promise((_resolve, reject) => {
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

	await assert.rejects(fetchJsonWithTimeout('/slow-request', {}, 5), { name: 'FetchTimeoutError' });
});

test('request timeout includes a response body that never finishes', async (context) => {
	const originalFetch = globalThis.fetch;
	context.after(() => {
		globalThis.fetch = originalFetch;
	});
	globalThis.fetch = async (_input, init) => ({
		ok: true,
		status: 200,
		json: () =>
			new Promise((_resolve, reject) => {
				init.signal.addEventListener(
					'abort',
					() => {
						const error = new Error('aborted');
						error.name = 'AbortError';
						reject(error);
					},
					{ once: true }
				);
			}),
	});

	await assert.rejects(fetchJsonWithTimeout('/slow-json', {}, 5), { name: 'FetchTimeoutError' });
});

test('successful JSON response preserves status and parsed data', async (context) => {
	const originalFetch = globalThis.fetch;
	context.after(() => {
		globalThis.fetch = originalFetch;
	});
	const response = { ok: true, status: 200, json: async () => ({ state: 'success' }) };
	globalThis.fetch = async () => response;

	const result = await fetchJsonWithTimeout('/success', {}, 100);
	assert.equal(result.response, response);
	assert.deepEqual(result.data, { state: 'success' });
});

test('invalid JSON retains the HTTP status for permanent-error classification', async (context) => {
	const originalFetch = globalThis.fetch;
	context.after(() => {
		globalThis.fetch = originalFetch;
	});
	globalThis.fetch = async () => ({
		ok: false,
		status: 401,
		json: async () => {
			throw new SyntaxError('not json');
		},
	});

	await assert.rejects(fetchJsonWithTimeout('/unauthorized', {}, 100), {
		name: 'FetchJsonResponseError',
		status: 401,
	});
});
