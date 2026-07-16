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

export const useSystemParametersStore = defineStore('system_parameters', {
	state: () => ({
		bufferedTaskCache: [] as SystemTask[],
		runtimeInstallId: '' as string,
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
			// The lifecycle store is the only /install_state observer. Resource
			// polling follows its authoritative readiness projection.
			const installStore = useInstallStateStore();
			this.syncRuntimeGeneration(installStore.installId, installStore.isReady);
			installStore.$subscribe(() => {
				this.syncRuntimeGeneration(installStore.installId, installStore.isReady);
			});
		},

		syncRuntimeGeneration(installId: string, ready: boolean) {
			const nextInstallId = String(installId || '');
			const generationChanged = nextInstallId !== this.runtimeInstallId;
			if (generationChanged || !ready) {
				// Fence an in-flight sample from the previous runtime before clearing
				// its presentation buffer. ``ready=false`` is also a read fence: do
				// not present samples from an uninstalling or degraded runtime.
				this.stopPolling();
				this.runtimeInstallId = nextInstallId;
				this.bufferedTaskCache.splice(0, this.bufferedTaskCache.length);
			}
			if (ready) this.startPolling();
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
				return true;
			} catch (error) {
				// Keep the last-known-good frontend buffer on cancellation or failure.
				if (!(error instanceof Error && error.name === 'AbortError')) {
					console.error('Fail to fetch system parameters', error);
				}
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
			if (!this.pollingActive && !this.pollingTimer && !this.requestController) return;
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
