export class FetchTimeoutError extends Error {
	constructor(timeoutMs: number) {
		super(`Request timed out after ${timeoutMs}ms`);
		this.name = 'FetchTimeoutError';
	}
}

export class FetchJsonResponseError extends Error {
	readonly status: number;

	constructor(status: number) {
		super(`Response body is not valid JSON (HTTP ${status})`);
		this.name = 'FetchJsonResponseError';
		this.status = status;
	}
}

export async function fetchJsonWithTimeout<T = unknown>(
	input: RequestInfo | URL,
	init: RequestInit,
	timeoutMs: number,
	controller = new AbortController()
): Promise<{ response: Response; data: T }> {
	let timedOut = false;
	let response: Response | null = null;
	const timer = window.setTimeout(() => {
		timedOut = true;
		controller.abort();
	}, timeoutMs);
	try {
		response = await fetch(input, { ...init, signal: controller.signal });
		const data = (await response.json()) as T;
		return { response, data };
	} catch (error) {
		if (timedOut) throw new FetchTimeoutError(timeoutMs);
		if (response && !(error instanceof Error && error.name === 'AbortError')) {
			throw new FetchJsonResponseError(response.status);
		}
		throw error;
	} finally {
		window.clearTimeout(timer);
	}
}
