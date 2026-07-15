import { defineStore } from 'pinia';

export const useInstallStateStore = defineStore('install_state', {
	state: () => ({
		status: 'uninstall',
		phase: 'uninstalled',
	}),
	getters: {
		isReady: (state) => state.status === 'install' && state.phase === 'active',
	},
	actions: {
		install(phase = 'active') {
			this.status = 'install';
			this.phase = phase || 'unknown';
		},
		uninstall() {
			this.status = 'uninstall';
			this.phase = 'uninstalled';
		},
		sync(state, phase) {
			if (state === 'install') this.install(phase);
			else if (state === 'uninstall') this.uninstall();
		},
		setPhase(phase) {
			if (this.status === 'install') this.phase = phase || 'unknown';
		},
	},
});
