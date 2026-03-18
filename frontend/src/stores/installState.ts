import { defineStore } from 'pinia';

export const useInstallStateStore = defineStore('install_state', {
	state: () => ({
		status: 'uninstall',
	}),
	actions: {
		install() {
			this.status = 'install';
		},
		uninstall() {
			this.status = 'uninstall';
		},
	},
});
