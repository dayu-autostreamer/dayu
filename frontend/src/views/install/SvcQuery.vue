<template>
	<div class="services-panel">
		<div class="panel-header">
			<div>
				<h3>Installed Services</h3>
				<p class="runtime-phase">Runtime phase: {{ install_state.phase }}</p>
			</div>

			<div class="panel-actions">
				<el-button round plain @click="refreshAll">
					<el-icon><RefreshRight /></el-icon>
					Refresh
				</el-button>
			</div>
		</div>

		<section class="panel-section">
			<div class="section-heading">
				<div class="section-heading__title">Service List</div>
			</div>

			<div v-if="services.length" class="service-chip-list">
				<label
					v-for="service in services"
					:key="service"
					class="service-chip"
					:class="{ 'is-selected': selected === service }"
				>
					<input
						v-model="selected"
						name="installed-service"
						type="radio"
						:value="service"
						@change="sendRequest(service)"
					/>
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
					<caption class="visually-hidden">
						Pod resource usage and the shared edge-to-cloud bandwidth measurement for the selected service
					</caption>
					<thead>
						<tr>
							<th scope="col">IP Address</th>
							<th scope="col">Hostname</th>
							<th scope="col">
								CPU / Node
								<span class="column-hint">Pod usage vs node resources</span>
							</th>
							<th scope="col">
								Memory / Node
								<span class="column-hint">Pod usage vs node resources</span>
							</th>
							<th scope="col">
								Shared Edge → Cloud
								<span class="column-hint">Singleton iperf probe</span>
							</th>
							<th scope="col">Creation Time</th>
						</tr>
					</thead>
					<tbody>
						<tr v-for="item in urlData" :key="`${item.ip}-${item.hostname}`">
							<td>{{ item.ip }}</td>
							<td>{{ item.hostname }}</td>
							<td>
								<div class="resource-meter">
									<div class="resource-meter__value">{{ formatUtilization(item.cpu) }}</div>
									<div
										v-if="hasUtilization(item.cpu)"
										class="resource-meter__track"
										role="progressbar"
										aria-label="Pod CPU share of node resources"
										:aria-valuenow="meterWidth(item.cpu)"
										:aria-valuetext="resourceAriaText(item.cpu, 'CPU')"
										aria-valuemin="0"
										aria-valuemax="100"
									>
										<span
											class="resource-meter__fill"
											:class="metricSeverity(item.cpu)"
											:style="{ width: `${meterWidth(item.cpu)}%` }"
										></span>
									</div>
									<div class="resource-meter__caption">{{ formatCpuDetail(item.cpu) }}</div>
									<div v-if="item.cpu?.status === 'stale'" class="resource-meter__stale">Last known sample</div>
								</div>
							</td>
							<td>
								<div class="resource-meter">
									<div class="resource-meter__value">{{ formatUtilization(item.memory) }}</div>
									<div
										v-if="hasUtilization(item.memory)"
										class="resource-meter__track"
										role="progressbar"
										aria-label="Pod memory share of node resources"
										:aria-valuenow="meterWidth(item.memory)"
										:aria-valuetext="resourceAriaText(item.memory, 'memory')"
										aria-valuemin="0"
										aria-valuemax="100"
									>
										<span
											class="resource-meter__fill"
											:class="metricSeverity(item.memory)"
											:style="{ width: `${meterWidth(item.memory)}%` }"
										></span>
									</div>
									<div class="resource-meter__caption">{{ formatMemoryDetail(item.memory) }}</div>
									<div v-if="item.memory?.status === 'stale'" class="resource-meter__stale">Last known sample</div>
								</div>
							</td>
							<td>
								<div class="bandwidth-metric">
									<strong>{{ formatBandwidth(item.bandwidth) }}</strong>
									<span>{{ bandwidthCaption(item.bandwidth) }}</span>
								</div>
							</td>
							<td>{{ item.age }}</td>
						</tr>
					</tbody>
				</table>
			</div>

			<div v-else class="empty-inline">Select a service to inspect its deployment details</div>
		</section>

		<div class="action-bar">
			<el-button type="danger" round :loading="loading" :disabled="installed !== 'install'" @click="uninstallServices">
				Uninstall
			</el-button>
		</div>
	</div>
</template>

<script>
import { ElMessage } from 'element-plus';
import { RefreshRight } from '@element-plus/icons-vue';
import { computed, markRaw, onBeforeUnmount, onMounted, ref, watch } from 'vue';
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
			serviceListController: null,
			serviceInfoController: null,
			serviceInfoTimer: null,
		};
	},
	setup() {
		const install_state = useInstallStateStore();
		const installed = ref('uninstall');
		const runtimeReady = computed(() => install_state.isReady);
		const loading = ref(false);
		let stateTimer = null;
		let stateController = null;
		let stateRequest = null;
		let statePollingActive = false;

		const syncInstallState = async () => {
			if (stateRequest) return stateRequest;
			const controller = new AbortController();
			stateController = controller;
			const request = (async () => {
				try {
					const response = await fetch('/api/install_state', { signal: controller.signal });
					if (!response.ok) throw new Error(`Install state request failed: ${response.status}`);
					const data = await response.json();
					installed.value = data.state;
					install_state.sync(data.state, data.phase);
				} catch (error) {
					if (error?.name !== 'AbortError') {
						console.error(error);
						ElMessage.error('System Error');
					}
				}
			})();
			stateRequest = request;
			try {
				return await request;
			} finally {
				if (stateRequest === request) stateRequest = null;
				if (stateController === controller) stateController = null;
			}
		};

		const pollInstallState = async () => {
			if (!statePollingActive) return;
			await syncInstallState();
			if (statePollingActive) {
				stateTimer = window.setTimeout(pollInstallState, 3000);
			}
		};

		const startInstallStatePolling = () => {
			if (statePollingActive) return;
			statePollingActive = true;
			void pollInstallState();
		};

		const stopInstallStatePolling = () => {
			statePollingActive = false;
			if (stateTimer) {
				clearTimeout(stateTimer);
				stateTimer = null;
			}
			stateController?.abort();
			stateController = null;
		};

		watch(
			() => install_state.status,
			(newValue) => {
				installed.value = newValue;
			}
		);

		onMounted(() => {
			startInstallStatePolling();
		});

		onBeforeUnmount(() => {
			stopInstallStatePolling();
		});

		return {
			installed,
			install_state,
			loading,
			runtimeReady,
			startInstallStatePolling,
			stopInstallStatePolling,
			syncInstallState,
		};
	},
	watch: {
		runtimeReady(ready) {
			if (ready) void this.getServiceList();
			else this.clearRuntimeDetails();
		},
	},
	methods: {
		isFiniteNumber(value) {
			return typeof value === 'number' && Number.isFinite(value);
		},
		formatNumber(value, maximumFractionDigits = 1) {
			return new Intl.NumberFormat('en-US', { maximumFractionDigits }).format(value);
		},
		hasUtilization(metric) {
			return (
				['available', 'stale'].includes(metric?.status) &&
				this.isFiniteNumber(metric?.utilization_percent) &&
				metric.utilization_percent >= 0
			);
		},
		formatUtilization(metric) {
			if (metric?.status === 'collecting') return 'Collecting metrics';
			if (!['available', 'stale'].includes(metric?.status)) return 'Metrics unavailable';
			if (!this.hasUtilization(metric)) {
				return this.hasAbsoluteUsage(metric) ? 'Usage available' : 'Metrics unavailable';
			}
			if (metric.utilization_percent > 0 && metric.utilization_percent < 0.1) return '<0.1%';
			return `${this.formatNumber(metric.utilization_percent, 1)}%`;
		},
		hasAbsoluteUsage(metric) {
			return this.isFiniteNumber(metric?.usage_millicores) || this.isFiniteNumber(metric?.usage_bytes);
		},
		resourceAriaText(metric, resource) {
			const freshness = metric?.status === 'stale' ? ', last known sample' : '';
			return `${resource}: ${this.formatUtilization(metric)}${freshness}`;
		},
		meterWidth(metric) {
			return this.hasUtilization(metric) ? Math.min(100, Math.max(0, metric.utilization_percent)) : 0;
		},
		metricSeverity(metric) {
			if (!this.hasUtilization(metric)) return '';
			if (metric.utilization_percent >= 90) return 'is-danger';
			if (metric.utilization_percent >= 70) return 'is-warning';
			return 'is-normal';
		},
		formatMillicores(value) {
			if (!this.isFiniteNumber(value)) return '—';
			if (value >= 1000) return `${this.formatNumber(value / 1000, 2)} cores`;
			return `${this.formatNumber(value, value < 10 ? 1 : 0)} mCPU`;
		},
		formatBytes(value) {
			if (!this.isFiniteNumber(value) || value < 0) return '—';
			if (value === 0) return '0 B';
			const units = ['B', 'KiB', 'MiB', 'GiB', 'TiB'];
			const index = Math.min(Math.floor(Math.log(value) / Math.log(1024)), units.length - 1);
			return `${this.formatNumber(value / 1024 ** index, 1)} ${units[index]}`;
		},
		referenceLabel(metric) {
			if (metric?.basis === 'node_allocatable') return 'node allocatable';
			if (metric?.basis === 'node_capacity') return 'node capacity';
			return 'node reference unavailable';
		},
		formatCpuDetail(metric) {
			if (!this.isFiniteNumber(metric?.usage_millicores)) return '';
			const usage = this.formatMillicores(metric.usage_millicores);
			if (!this.isFiniteNumber(metric.reference_millicores)) return `${usage} · ${this.referenceLabel(metric)}`;
			return `${usage} / ${this.formatMillicores(metric.reference_millicores)} ${this.referenceLabel(metric)}`;
		},
		formatMemoryDetail(metric) {
			if (!this.isFiniteNumber(metric?.usage_bytes)) return '';
			const usage = this.formatBytes(metric.usage_bytes);
			if (!this.isFiniteNumber(metric.reference_bytes)) return `${usage} · ${this.referenceLabel(metric)}`;
			return `${usage} / ${this.formatBytes(metric.reference_bytes)} ${this.referenceLabel(metric)}`;
		},
		formatBandwidth(metric) {
			if (metric?.status === 'collecting') return 'Collecting probe';
			if (metric?.status === 'ambiguous') return 'Probe conflict';
			if (!['available', 'stale'].includes(metric?.status) || !this.isFiniteNumber(metric.mbps))
				return 'Probe unavailable';
			return `${this.formatNumber(metric.mbps, 2)} Mbps`;
		},
		bandwidthCaption(metric) {
			if (metric?.status === 'available' && metric.probe_node) return `Measured by ${metric.probe_node}`;
			if (metric?.status === 'stale' && metric.probe_node) return `Last known · measured by ${metric.probe_node}`;
			if (metric?.status === 'ambiguous') return 'Multiple active probe values';
			if (metric?.status === 'collecting') return 'Waiting for shared probe';
			return 'No valid shared measurement';
		},
		stopServiceInfoPolling() {
			if (this.serviceInfoTimer) {
				window.clearTimeout(this.serviceInfoTimer);
				this.serviceInfoTimer = null;
			}
			this.serviceInfoController?.abort();
			this.serviceInfoController = null;
		},
		clearRuntimeDetails() {
			this.stopServiceInfoPolling();
			this.serviceListController?.abort();
			this.services = [];
			this.selected = null;
			this.selected_service = null;
			this.urlData = [];
		},
		async getServiceList() {
			if (!this.runtimeReady) {
				this.clearRuntimeDetails();
				return;
			}
			this.serviceListController?.abort();
			const controller = markRaw(new AbortController());
			this.serviceListController = controller;
			try {
				const response = await fetch('/api/installed_service', { signal: controller.signal });
				if (!response.ok) throw new Error(`Service list request failed: ${response.status}`);
				const data = await response.json();
				if (controller.signal.aborted || this.serviceListController !== controller) return;
				this.services = Array.isArray(data) ? data : [];

				if (this.selected && !this.services.includes(this.selected)) {
					this.stopServiceInfoPolling();
					this.selected = null;
					this.selected_service = null;
					this.urlData = [];
				}
			} catch (error) {
				if (error?.name !== 'AbortError') {
					console.error(error);
					ElMessage.error('System Error');
				}
			} finally {
				if (this.serviceListController === controller) this.serviceListController = null;
			}
		},
		async refreshAll() {
			await this.syncInstallState();
			if (!this.runtimeReady) {
				this.clearRuntimeDetails();
				return;
			}
			await this.getServiceList();
			if (this.selected_service) {
				await this.sendRequest(this.selected_service);
			}
		},
		async sendRequest(service) {
			this.stopServiceInfoPolling();
			if (!service || !this.runtimeReady || !this.services.includes(service)) {
				this.urlData = [];
				return;
			}

			const controller = markRaw(new AbortController());
			this.serviceInfoController = controller;
			try {
				this.selected_service = service;
				const response = await fetch(`/api/service_info/${service}`, { signal: controller.signal });
				if (!response.ok) throw new Error(`Service detail request failed: ${response.status}`);
				const data = await response.json();
				if (controller.signal.aborted || this.serviceInfoController !== controller || this.selected_service !== service)
					return;
				this.urlData = Array.isArray(data) ? data : [];
			} catch (error) {
				if (error?.name !== 'AbortError') {
					console.error(error);
					ElMessage.error('System Error');
				}
			} finally {
				const shouldContinue =
					this.serviceInfoController === controller && this.runtimeReady && this.selected_service === service;
				if (this.serviceInfoController === controller) this.serviceInfoController = null;
				if (shouldContinue) {
					this.serviceInfoTimer = window.setTimeout(() => {
						this.serviceInfoTimer = null;
						void this.sendRequest(service);
					}, 5000);
				}
			}
		},
		async uninstallServices() {
			this.loading = true;
			this.install_state.setPhase('uninstalling');
			this.clearRuntimeDetails();
			try {
				const response = await fetch('/api/stop_service', {
					method: 'POST',
				});
				const data = await response.json();

				if (data.state === 'success') {
					await this.syncInstallState();
					ElMessage({
						message: data.msg,
						showClose: true,
						type: 'success',
						duration: 3000,
					});
					window.dispatchEvent(new Event(INSTALL_CHANGED_EVENT));
				} else {
					await this.syncInstallState();
					ElMessage({
						message: data.msg,
						showClose: true,
						type: 'error',
						duration: 3000,
					});
				}
			} catch (error) {
				console.error(error);
				await this.syncInstallState();
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
		this.stopServiceInfoPolling();
		this.serviceListController?.abort();
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

.runtime-phase {
	margin: 6px 0 0;
	font-size: 13px;
	color: #64748b;
}

.panel-actions {
	display: flex;
	flex-wrap: wrap;
	gap: 8px;
}

.panel-section {
	padding: 18px;
	border-radius: 22px;
	border: 1px solid #e2e8f0;
	background: linear-gradient(135deg, rgba(37, 99, 235, 0.04), transparent 34%), #ffffff;
	display: grid;
	gap: 16px;
}

.section-heading__title {
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

.column-hint {
	display: block;
	margin-top: 3px;
	font-size: 11px;
	font-weight: 500;
	color: #64748b;
}

.visually-hidden {
	position: absolute;
	width: 1px;
	height: 1px;
	padding: 0;
	margin: -1px;
	overflow: hidden;
	clip: rect(0, 0, 0, 0);
	white-space: nowrap;
	border: 0;
}

.details-table td {
	color: #475569;
}

.details-table tbody tr:hover {
	background: #f8fbff;
}

.resource-meter,
.bandwidth-metric {
	min-width: 170px;
	display: grid;
	gap: 5px;
	text-align: left;
}

.resource-meter__value,
.bandwidth-metric strong {
	font-size: 13px;
	font-weight: 700;
	color: #0f172a;
}

.resource-meter__track {
	height: 6px;
	overflow: hidden;
	border-radius: 999px;
	background: #e2e8f0;
}

.resource-meter__fill {
	display: block;
	height: 100%;
	border-radius: inherit;
	transition: width 0.25s ease;
}

.resource-meter__fill.is-normal {
	background: #22c55e;
}

.resource-meter__fill.is-warning {
	background: #f59e0b;
}

.resource-meter__fill.is-danger {
	background: #ef4444;
}

.resource-meter__caption,
.bandwidth-metric span {
	font-size: 11px;
	line-height: 1.35;
	color: #64748b;
}

.resource-meter__stale {
	font-size: 10px;
	font-weight: 700;
	color: #b45309;
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
	justify-content: flex-end;
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
