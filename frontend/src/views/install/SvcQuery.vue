<template>
	<div class="services-panel">
		<div class="panel-header">
			<div>
				<h3>Installed Services</h3>
			</div>

			<div class="panel-actions">
				<el-button round plain @click="refreshAll">
					<el-icon><RefreshRight /></el-icon>
					Refresh
				</el-button>
			</div>
		</div>

		<div class="summary-tags">
			<el-tag :type="installed === 'install' ? 'success' : 'info'" effect="plain">
				{{ installed === 'install' ? 'Installed' : 'Not installed' }}
			</el-tag>
			<el-tag type="info" effect="plain">{{ services.length }} services</el-tag>
			<el-tag type="info" effect="plain">{{ urlData.length }} hosts</el-tag>
		</div>

		<section class="panel-section">
			<div class="section-heading">
				<div class="section-heading__title">Service List</div>
			</div>

			<div v-if="services.length" class="service-chip-list">
				<label v-for="service in services" :key="service" class="service-chip" :class="{ 'is-selected': selected === service }">
					<input v-model="selected" type="radio" :value="service" @change="sendRequest(service)" />
					<span>{{ service }}</span>
				</label>
			</div>

			<div v-else class="empty-inline">No installed services</div>
		</section>

		<section class="panel-section">
			<div class="section-heading">
				<div class="section-heading__title">Current Service Details</div>
			</div>

			<div v-if="urlData.length" class="table-shell">
				<table class="details-table">
					<thead>
						<tr>
							<th>IP Address</th>
							<th>Hostname</th>
							<th>CPU Usage</th>
							<th>Memory Usage</th>
							<th>Bandwidth</th>
							<th>Creation Time</th>
						</tr>
					</thead>
					<tbody>
						<tr v-for="item in urlData" :key="`${item.ip}-${item.hostname}`">
							<td>{{ item.ip }}</td>
							<td>{{ item.hostname }}</td>
							<td>{{ item.cpu }}</td>
							<td>{{ item.memory }}</td>
							<td>{{ item.bandwidth }}</td>
							<td>{{ item.age }}</td>
						</tr>
					</tbody>
				</table>
			</div>

			<div v-else class="empty-inline">Select a service to inspect its deployment details</div>
		</section>

		<div class="action-bar">
			<div class="action-bar__summary">
				{{ installed === 'install' ? 'Uninstall removes the currently deployed application stack.' : 'No deployed application stack to remove.' }}
			</div>

			<el-button type="danger" round :loading="loading" :disabled="installed !== 'install'" @click="uninstallServices">
				Uninstall
			</el-button>
		</div>
	</div>
</template>

<script>
import { ElMessage } from 'element-plus';
import { RefreshRight } from '@element-plus/icons-vue';
import { onBeforeUnmount, onMounted, ref, watch } from 'vue';
import { useInstallStateStore } from '/@/stores/installState';

const INSTALL_CHANGED_EVENT = 'dayu-install-changed';

export default {
	components: {
		RefreshRight,
	},
	data() {
		return {
			services: [],
			urlData: [],
			selected: null,
			selected_service: null,
			handleInstallChanged: null,
		};
	},
	setup() {
		const install_state = useInstallStateStore();
		const installed = ref('uninstall');
		const loading = ref(false);
		let stateTimer = null;

		const syncInstallState = async () => {
			try {
				const response = await fetch('/api/install_state');
				const data = await response.json();
				installed.value = data.state;
				if (data.state === 'install') {
					install_state.install();
				} else {
					install_state.uninstall();
				}
			} catch (error) {
				console.error(error);
				ElMessage.error('System Error');
			}
		};

		watch(
			() => install_state.status,
			(newValue) => {
				installed.value = newValue;
			}
		);

		onMounted(() => {
			stateTimer = window.setInterval(() => {
				syncInstallState();
			}, 3000);
		});

		onBeforeUnmount(() => {
			if (stateTimer) {
				clearInterval(stateTimer);
				stateTimer = null;
			}
		});

		return {
			installed,
			install_state,
			loading,
			syncInstallState,
		};
	},
	methods: {
		async getServiceList() {
			try {
				const response = await fetch('/api/installed_service');
				const data = await response.json();
				this.services = Array.isArray(data) ? data : [];

				if (this.selected && !this.services.includes(this.selected)) {
					this.selected = null;
					this.selected_service = null;
					this.urlData = [];
				}
			} catch (error) {
				console.error(error);
				ElMessage.error('System Error');
			}
		},
		async refreshAll() {
			await this.syncInstallState();
			await this.getServiceList();
			if (this.selected_service) {
				await this.sendRequest(this.selected_service);
			}
		},
		async sendRequest(service) {
			if (!service) {
				this.urlData = [];
				return;
			}

			try {
				this.selected_service = service;
				const response = await fetch(`/api/service_info/${service}`);
				const data = await response.json();
				this.urlData = Array.isArray(data) ? data : [];
			} catch (error) {
				console.error(error);
				ElMessage.error('System Error');
			}
		},
		async uninstallServices() {
			this.loading = true;
			try {
				const response = await fetch('/api/stop_service', {
					method: 'POST',
				});
				const data = await response.json();

				if (data.state === 'success') {
					this.install_state.uninstall();
					this.selected = null;
					this.selected_service = null;
					this.urlData = [];
					await this.syncInstallState();
					await this.getServiceList();
					ElMessage({
						message: data.msg,
						showClose: true,
						type: 'success',
						duration: 3000,
					});
					window.dispatchEvent(new Event(INSTALL_CHANGED_EVENT));
				} else {
					ElMessage({
						message: data.msg,
						showClose: true,
						type: 'error',
						duration: 3000,
					});
				}
			} catch (error) {
				console.error(error);
				ElMessage.error('Network Error');
			} finally {
				this.loading = false;
			}
		},
	},
	async mounted() {
		await this.refreshAll();
		this.handleInstallChanged = () => {
			this.refreshAll();
		};
		window.addEventListener(INSTALL_CHANGED_EVENT, this.handleInstallChanged);
	},
	beforeUnmount() {
		if (this.handleInstallChanged) {
			window.removeEventListener(INSTALL_CHANGED_EVENT, this.handleInstallChanged);
		}
	},
};
</script>

<style scoped lang="scss">
.services-panel {
	display: grid;
	gap: 22px;
}

.panel-header,
.section-heading,
.action-bar {
	display: flex;
	align-items: flex-start;
	justify-content: space-between;
	gap: 14px;
}

.panel-header h3 {
	margin: 0;
	font-size: 26px;
	color: #0f172a;
}

.panel-actions,
.summary-tags {
	display: flex;
	flex-wrap: wrap;
	gap: 8px;
}

.summary-tags {
	margin-top: -6px;
}

.panel-section {
	padding: 18px;
	border-radius: 22px;
	border: 1px solid #e2e8f0;
	background:
		linear-gradient(135deg, rgba(37, 99, 235, 0.04), transparent 34%),
		#ffffff;
	display: grid;
	gap: 16px;
}

.section-heading__title,
.action-bar__summary {
	font-size: 13px;
	font-weight: 700;
	letter-spacing: 0.06em;
	text-transform: uppercase;
	color: #475569;
}

.service-chip-list {
	display: flex;
	flex-wrap: wrap;
	gap: 10px;
}

.service-chip {
	display: inline-flex;
	align-items: center;
	gap: 8px;
	padding: 9px 12px;
	border-radius: 999px;
	border: 1px solid #dbe4ee;
	background: #f8fafc;
	cursor: pointer;
	transition: border-color 0.2s ease, box-shadow 0.2s ease, transform 0.2s ease;
}

.service-chip:hover {
	border-color: #93c5fd;
	transform: translateY(-1px);
}

.service-chip.is-selected {
	border-color: #3b82f6;
	background: #eff6ff;
	box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.1);
}

.service-chip input {
	margin: 0;
}

.service-chip span {
	font-size: 13px;
	font-weight: 700;
	color: #0f172a;
}

.table-shell {
	overflow-x: auto;
	border-radius: 18px;
	border: 1px solid #e2e8f0;
}

.details-table {
	width: 100%;
	border-collapse: collapse;
	background: #ffffff;
}

.details-table th,
.details-table td {
	padding: 12px 14px;
	text-align: center;
	border-bottom: 1px solid #e2e8f0;
	font-size: 13px;
}

.details-table th {
	background: #f8fafc;
	font-weight: 700;
	color: #334155;
}

.details-table td {
	color: #475569;
}

.details-table tbody tr:hover {
	background: #f8fbff;
}

.empty-inline {
	min-height: 84px;
	display: grid;
	place-items: center;
	text-align: center;
	border: 1px dashed #cbd5e1;
	border-radius: 16px;
	background: #f8fafc;
	font-size: 14px;
	color: #64748b;
}

.action-bar {
	padding-top: 4px;
	border-top: 1px solid #e2e8f0;
}

.action-bar__summary {
	text-transform: none;
	letter-spacing: normal;
	line-height: 1.6;
}

@media (max-width: 768px) {
	.panel-header,
	.section-heading,
	.action-bar {
		flex-direction: column;
		align-items: flex-start;
	}
}
</style>
