// filepath: /Users/onecheck/PycharmProjects/dayu-inner-dev/frontend/src/stores/systemParameters.ts
import { defineStore } from 'pinia';
import { markRaw } from 'vue';
import { useInstallStateStore } from '/@/stores/installState';

export type SystemParamItem = {
	id: string;
	data: Record<string, any>;
};
export type SystemTask = {
	timestamp: number | string;
	data: SystemParamItem[];
};

const LOCAL_LOG_KEY = 'system_parameters_buffer_v1';

export const useSystemParametersStore = defineStore('system_parameters', {
	state: () => ({
		bufferedTaskCache: [] as SystemTask[],
		maxBufferedTaskCacheSize: 20 as number,
		pollingTimer: null as ReturnType<typeof setTimeout> | null,
		pollingActive: false as boolean,
		pollingGeneration: 0 as number,
		requestInFlight: false as boolean,
		requestController: null as AbortController | null,
		initialized: false as boolean,
	}),
	actions: {
		init() {
			if (this.initialized) return;
			this.initialized = true;
			// load cache from storage
			this.loadFromStorage();
			// start/stop by backend state on boot
			this.syncWithBackendInstallState();
			// subscribe to install store for runtime changes
			try {
				const installStore = useInstallStateStore();
				installStore.$subscribe((mutation, state) => {
					if (state.status === 'install' && state.phase === 'active') this.startPolling();
					else this.stopPolling();
				});
			} catch {
				// Pinia can be unavailable in isolated frontend tests.
			}
		},

		async syncWithBackendInstallState() {
			try {
				const installStore = useInstallStateStore();
				const resp = await fetch('/api/install_state');
				const json = await resp.json();
				installStore.sync(json?.state, json?.phase);
				if (installStore.isReady) this.startPolling();
				else this.stopPolling();
			} catch {
				// ignore network errors
			}
		},

		loadFromStorage() {
			try {
				const raw = localStorage.getItem(LOCAL_LOG_KEY);
				if (!raw) return;
				const parsed = JSON.parse(raw);
				if (Array.isArray(parsed)) {
					const slice = parsed.slice(-this.maxBufferedTaskCacheSize);
					this.bufferedTaskCache.splice(0, this.bufferedTaskCache.length, ...slice);
				}
			} catch {
				// Ignore malformed or unavailable browser storage.
			}
		},

		persistToStorage() {
			try {
				localStorage.setItem(LOCAL_LOG_KEY, JSON.stringify(this.bufferedTaskCache));
			} catch {
				// Storage quota failures must not stop live polling.
			}
		},

		async fetchLatest(signal?: AbortSignal) {
			if (this.requestInFlight) return false;
			this.requestInFlight = true;
			try {
				const response = await fetch('/api/system_parameters', { signal });
				if (!response.ok) throw new Error(`System parameters request failed: ${response.status}`);
				const data = await response.json();
				if (signal?.aborted) return false;
				const newTasks: SystemTask[] = (data || []).map((task: any) => ({
					...task,
					data: (task.data || []).map((item: any) => ({
						id: String(item.id),
						data: item.data,
					})),
				}));
				const merged = [...this.bufferedTaskCache, ...newTasks];
				const sliced = merged.slice(-this.maxBufferedTaskCacheSize);
				// update in place to retain reactivity
				this.bufferedTaskCache.splice(0, this.bufferedTaskCache.length, ...sliced);
				this.persistToStorage();
				return true;
			} catch {
				// Keep the last-known-good frontend buffer on cancellation or failure.
				return false;
			} finally {
				this.requestInFlight = false;
			}
		},

		startPolling() {
			if (this.pollingActive) return;
			this.pollingActive = true;
			const generation = ++this.pollingGeneration;

			const poll = async () => {
				if (!this.pollingActive || generation !== this.pollingGeneration) return;
				const controller = markRaw(new AbortController());
				this.requestController = controller;
				await this.fetchLatest(controller.signal);
				if (this.requestController === controller) this.requestController = null;
				if (!this.pollingActive || generation !== this.pollingGeneration) return;
				this.pollingTimer = setTimeout(poll, 2000);
			};

			void poll();
		},

		stopPolling() {
			this.pollingActive = false;
			this.pollingGeneration += 1;
			if (this.pollingTimer) {
				clearTimeout(this.pollingTimer);
				this.pollingTimer = null;
			}
			this.requestController?.abort();
			this.requestController = null;
		},
	},
});
