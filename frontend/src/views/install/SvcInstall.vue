<template>
	<div class="install-panel">
		<div class="panel-header">
			<div>
				<h3>Application Installation</h3>
			</div>

			<div class="panel-actions">
				<el-button round plain @click="refreshOptions">
					<el-icon><RefreshRight /></el-icon>
					Refresh
				</el-button>
			</div>
		</div>

		<div class="summary-tags">
			<el-tag :type="installed === 'install' ? 'success' : 'info'" effect="plain">
				{{ installed === 'install' ? 'Installed' : 'Ready to install' }}
			</el-tag>
			<el-tag type="info" effect="plain">{{ selectedSources.length }} sources</el-tag>
			<el-tag :type="readySourceCount === selectedSources.length && selectedSources.length ? 'success' : 'warning'" effect="plain">
				{{ readySourceCount }}/{{ selectedSources.length || 0 }} mapped
			</el-tag>
		</div>

		<div class="selection-grid">
			<section class="selector-card">
				<div class="selector-card__label">Scheduler Policy</div>
				<el-select v-model="selectedPolicyIndex" placeholder="Choose scheduler policy" class="selector-card__control">
					<template v-for="(option, index) in policyOptions" :key="index">
						<el-option v-if="isValidIndex(index, policyOptions)" :value="index" :label="option.policy_name" />
					</template>
				</el-select>
				<div class="selector-card__meta">
					{{ selectedPolicyIndex !== null && isValidIndex(selectedPolicyIndex, policyOptions) ? policyOptions[selectedPolicyIndex].policy_name : 'No policy selected' }}
				</div>
			</section>

			<section class="selector-card">
				<div class="selector-card__label">Datasource Configuration</div>
				<el-select
					v-model="selectedDatasourceIndex"
					@change="handleDatasourceChange"
					placeholder="Choose datasource configuration"
					class="selector-card__control"
				>
					<template v-for="(option, index) in datasourceOptions" :key="index">
						<el-option v-if="isValidIndex(index, datasourceOptions)" :value="index" :label="option.source_name" />
					</template>
				</el-select>
				<div class="selector-card__meta">
					{{
						selectedDatasourceIndex !== null && isValidIndex(selectedDatasourceIndex, datasourceOptions)
							? `${datasourceOptions[selectedDatasourceIndex].source_name} · ${selectedSources.length} sources`
							: 'No datasource selected'
					}}
				</div>
			</section>
		</div>

		<section v-if="selectedSources.length" class="mapping-section">
			<div class="section-heading">
				<div class="section-heading__title">Source Mapping</div>
				<div class="section-heading__actions">
					<el-button size="small" round @click="assignAllNodes" :disabled="!nodeOptions.length">All Nodes</el-button>
					<el-button size="small" round @click="clearAllNodeBindings" :disabled="!selectedSources.length">Clear Nodes</el-button>
				</div>
			</div>

			<div class="source-grid">
				<article
					v-for="(source, index) in selectedSources"
					:key="`${source.id}-${source.name}`"
					class="source-card"
					:class="{ 'is-ready': isSourceReady(source) }"
				>
					<div class="source-card__header">
						<div class="source-card__title-group">
							<div class="source-card__title">Source {{ source.id }}</div>
							<div class="source-card__subtitle">{{ source.name }}</div>
						</div>

						<el-tag :type="isSourceReady(source) ? 'success' : 'warning'" size="small" effect="plain">
							{{ isSourceReady(source) ? 'Ready' : 'Incomplete' }}
						</el-tag>
					</div>

					<div class="field-stack">
						<div class="field-block">
							<div class="field-block__label">Dag</div>
							<el-select
								v-model="source.dag_selected"
								@change="updateDagSelection(index, source.dag_selected)"
								placeholder="Assign dag"
								class="field-block__control"
							>
								<template v-for="(option, dagIndex) in dagOptions" :key="dagIndex">
									<el-option v-if="isValidIndex(dagIndex, dagOptions)" :label="option.dag_name" :value="option.dag_id" />
								</template>
							</el-select>
						</div>

						<div class="field-block">
							<div class="field-block__label">Edge Nodes</div>
							<el-select
								v-model="source.node_selected"
								@change="updateNodeSelection(index, source.node_selected)"
								placeholder="Bind edge nodes"
								class="field-block__control"
								multiple
								collapse-tags
								collapse-tags-tooltip
							>
								<template v-for="(option, nodeIndex) in nodeOptions" :key="nodeIndex">
									<el-option v-if="isValidIndex(nodeIndex, nodeOptions)" :label="option.name" :value="option.name" />
								</template>
							</el-select>
						</div>
					</div>

					<div class="source-card__footer">
						<div class="source-card__footer-text">
							{{ source.node_selected?.length || 0 }} node{{ (source.node_selected?.length || 0) === 1 ? '' : 's' }} bound
						</div>
						<div class="source-card__footer-actions">
							<button type="button" class="mini-link" @click="assignAllNodesForSource(index)" :disabled="!nodeOptions.length">All</button>
							<button type="button" class="mini-link" @click="clearNodesForSource(index)" :disabled="!(source.node_selected?.length)">Clear</button>
						</div>
					</div>
				</article>
			</div>
		</section>

		<div v-else class="empty-state">
			<el-icon class="empty-state__icon"><Connection /></el-icon>
			<div class="empty-state__title">No sources to map</div>
			<div class="empty-state__subtitle">Choose a datasource configuration to continue.</div>
		</div>

		<div class="action-bar">
			<div class="action-bar__summary">
				{{ installSummary }}
			</div>

			<div class="builder-buttons">
				<el-button
					type="primary"
					round
					native-type="submit"
					:loading="loading"
					:disabled="installed === 'install'"
					@click="submitService"
				>
					<el-icon><Promotion /></el-icon>
					Install
				</el-button>

				<el-button round @click="handleClear">Clear</el-button>
			</div>
		</div>
	</div>
</template>

<script>
import { ElButton, ElMessage } from 'element-plus';
import { Connection, Promotion, RefreshRight } from '@element-plus/icons-vue';
import axios from 'axios';
import { onMounted, ref, watch } from 'vue';
import { useInstallStateStore } from '/@/stores/installState';

const INSTALL_STATE_KEY = 'savedInstallConfig';
const DRAFT_STATE_KEY = 'savedDraftConfig';
const INSTALL_CHANGED_EVENT = 'dayu-install-changed';

export default {
	components: {
		ElButton,
		Connection,
		Promotion,
		RefreshRight,
	},
	data() {
		return {
			loading: false,
		};
	},
	computed: {
		readySourceCount() {
			return this.selectedSources.filter((source) => this.isSourceReady(source)).length;
		},
		installSummary() {
			if (this.installed === 'install') {
				return 'Services are already installed. Uninstall from the right panel before installing again.';
			}

			if (!this.selectedSources.length) {
				return 'Choose a datasource configuration to begin mapping.';
			}

			if (this.readySourceCount === this.selectedSources.length) {
				return 'All sources are ready for installation.';
			}

			return `${this.selectedSources.length - this.readySourceCount} source mapping${this.selectedSources.length - this.readySourceCount === 1 ? '' : 's'} still need attention.`;
		},
	},
	setup() {
		const LENGTH_KEYS = {
			policy: 'prev_len:policy',
			datasource: 'prev_len:datasource',
			dag: 'prev_len:dag',
			node: 'prev_len:node',
		};

		const selectedPolicyIndex = ref(null);
		const selectedDatasourceIndex = ref(null);
		const selectedSources = ref([]);
		const installed = ref('uninstall');
		const policyOptions = ref([]);
		const datasourceOptions = ref([]);
		const dagOptions = ref([]);
		const nodeOptions = ref([]);
		const install_state = useInstallStateStore();

		const isValidIndex = (index, array) =>
			Number.isSafeInteger(index) && index >= 0 && Array.isArray(array) && index < array.length && Object.prototype.hasOwnProperty.call(array, index);

		const safeClone = (value) => {
			try {
				return JSON.parse(JSON.stringify(value));
			} catch {
				return null;
			}
		};

		const loadStorage = (key) => {
			try {
				const data = localStorage.getItem(key);
				return data ? JSON.parse(data) : null;
			} catch (error) {
				console.error('Fail to load storage', error);
				return null;
			}
		};

		const saveStorage = (key, value) => {
			try {
				localStorage.setItem(key, JSON.stringify(value));
			} catch (error) {
				console.error('Fail to save storage', error);
			}
		};

		const createSourceSelections = (datasource, savedSources = []) => {
			const savedById = new Map((savedSources || []).map((source) => [source.id, source]));
			return (datasource?.source_list || []).map((source) => {
				const saved = savedById.get(source.id) || {};
				const dagSelected = dagOptions.value.some((dag) => dag.dag_id === saved.dag_selected) ? saved.dag_selected : '';
				const nodeSelected = Array.isArray(saved.node_selected)
					? saved.node_selected.filter((nodeName) => nodeOptions.value.some((node) => node.name === nodeName))
					: [];

				return {
					...source,
					dag_selected: dagSelected,
					node_selected: nodeSelected,
				};
			});
		};

		const restoreSelections = () => {
			const activeKey = installed.value === 'install' ? INSTALL_STATE_KEY : DRAFT_STATE_KEY;
			const storedConfig = loadStorage(activeKey) || loadStorage(DRAFT_STATE_KEY) || loadStorage(INSTALL_STATE_KEY);

			if (!storedConfig) {
				selectedPolicyIndex.value = null;
				selectedDatasourceIndex.value = null;
				selectedSources.value = [];
				return;
			}

			selectedPolicyIndex.value = isValidIndex(storedConfig.selectedPolicyIndex, policyOptions.value)
				? storedConfig.selectedPolicyIndex
				: null;

			selectedDatasourceIndex.value = isValidIndex(storedConfig.selectedDatasourceIndex, datasourceOptions.value)
				? storedConfig.selectedDatasourceIndex
				: null;

			if (selectedDatasourceIndex.value !== null) {
				const datasource = datasourceOptions.value[selectedDatasourceIndex.value];
				selectedSources.value = createSourceSelections(datasource, storedConfig.selectedSources);
			} else {
				selectedSources.value = [];
			}
		};

		const refreshOptions = async () => {
			try {
				const [installStateResponse, policyResponse, datasourceResponse, dagResponse, nodeResponse] = await Promise.all([
					axios.get('/api/install_state'),
					axios.get('/api/policy'),
					axios.get('/api/datasource'),
					axios.get('/api/dag_workflow'),
					axios.get('/api/edge_node'),
				]);

				installed.value = installStateResponse.data.state;
				if (installed.value === 'install') {
					install_state.install();
				} else {
					install_state.uninstall();
				}

				policyOptions.value = Array.isArray(policyResponse.data) ? policyResponse.data : [];
				datasourceOptions.value = Array.isArray(datasourceResponse.data) ? datasourceResponse.data : [];
				dagOptions.value = Array.isArray(dagResponse.data) ? dagResponse.data : [];
				nodeOptions.value = Array.isArray(nodeResponse.data) ? nodeResponse.data : [];

				localStorage.setItem(LENGTH_KEYS.policy, policyOptions.value.length);
				localStorage.setItem(LENGTH_KEYS.datasource, datasourceOptions.value.length);
				localStorage.setItem(LENGTH_KEYS.dag, dagOptions.value.length);
				localStorage.setItem(LENGTH_KEYS.node, nodeOptions.value.length);

				restoreSelections();
			} catch (error) {
				console.error('Fail to refresh install options', error);
				ElMessage.error('Fail to refresh install options');
			}
		};

		watch(
			[selectedPolicyIndex, selectedDatasourceIndex, selectedSources, installed],
			([policyIdx, datasourceIdx, sources, installStatus]) => {
				const payload = {
					selectedPolicyIndex: isValidIndex(policyIdx, policyOptions.value) ? policyIdx : null,
					selectedDatasourceIndex: isValidIndex(datasourceIdx, datasourceOptions.value) ? datasourceIdx : null,
					selectedSources: safeClone(sources) || [],
				};

				if (installStatus === 'install') {
					saveStorage(INSTALL_STATE_KEY, payload);
				} else {
					saveStorage(DRAFT_STATE_KEY, payload);
				}
			},
			{ deep: true }
		);

		watch(
			() => install_state.status,
			(newValue, oldValue) => {
				installed.value = newValue;
				if (oldValue === 'install' && newValue === 'uninstall') {
					saveStorage(DRAFT_STATE_KEY, {
						selectedPolicyIndex: selectedPolicyIndex.value,
						selectedDatasourceIndex: selectedDatasourceIndex.value,
						selectedSources: safeClone(selectedSources.value) || [],
					});
					localStorage.removeItem(INSTALL_STATE_KEY);
				}
			}
		);

		onMounted(async () => {
			await refreshOptions();
		});

		return {
			DRAFT_STATE_KEY,
			INSTALL_CHANGED_EVENT,
			INSTALL_STATE_KEY,
			dagOptions,
			datasourceOptions,
			install_state,
			installed,
			isValidIndex,
			nodeOptions,
			policyOptions,
			refreshOptions,
			selectedDatasourceIndex,
			selectedPolicyIndex,
			selectedSources,
		};
	},
	methods: {
		isSourceReady(source) {
			return Boolean(source?.dag_selected) && Array.isArray(source?.node_selected) && source.node_selected.length > 0;
		},
		updateDagSelection(index, selected) {
			this.selectedSources[index].dag_selected = selected;
		},
		updateNodeSelection(index, selected) {
			this.selectedSources[index].node_selected = selected;
		},
		handleDatasourceChange() {
			if (!this.isValidIndex(this.selectedDatasourceIndex, this.datasourceOptions)) {
				this.selectedDatasourceIndex = null;
				this.selectedSources = [];
				return;
			}

			const datasource = this.datasourceOptions[this.selectedDatasourceIndex];
			this.selectedSources = (datasource.source_list || []).map((source) => ({
				...source,
				dag_selected: '',
				node_selected: [],
			}));
		},
		assignAllNodes() {
			const allNodeNames = this.nodeOptions.map((node) => node.name);
			this.selectedSources = this.selectedSources.map((source) => ({
				...source,
				node_selected: [...allNodeNames],
			}));
		},
		clearAllNodeBindings() {
			this.selectedSources = this.selectedSources.map((source) => ({
				...source,
				node_selected: [],
			}));
		},
		assignAllNodesForSource(index) {
			this.selectedSources[index].node_selected = this.nodeOptions.map((node) => node.name);
		},
		clearNodesForSource(index) {
			this.selectedSources[index].node_selected = [];
		},
		async submitService() {
			if (!this.isValidIndex(this.selectedPolicyIndex, this.policyOptions)) {
				ElMessage.error('Please choose scheduler policy');
				return;
			}

			if (!this.isValidIndex(this.selectedDatasourceIndex, this.datasourceOptions)) {
				ElMessage.error('Please choose datasource configuration');
				return;
			}

			for (let i = 0; i < this.selectedSources.length; i += 1) {
				const source = this.selectedSources[i];
				if (!source?.dag_selected) {
					ElMessage.error(`Please assign a dag for source ${source?.id ?? i}`);
					return;
				}
				if (!Array.isArray(source?.node_selected) || source.node_selected.length === 0) {
					ElMessage.error(`Please bind edge nodes for source ${source?.id ?? i}${source?.name ? `: ${source.name}` : ''}`);
					return;
				}
			}

			const payload = {
				source_config_label: this.datasourceOptions[this.selectedDatasourceIndex].source_label,
				policy_id: this.policyOptions[this.selectedPolicyIndex].policy_id,
				source: this.selectedSources,
			};

			this.loading = true;
			try {
				const response = await fetch('/api/install', {
					method: 'POST',
					body: JSON.stringify(payload),
				});
				const data = await response.json();

				if (data.state === 'success') {
					this.install_state.install();
					localStorage.setItem(
						this.INSTALL_STATE_KEY,
						JSON.stringify({
							selectedPolicyIndex: this.selectedPolicyIndex,
							selectedDatasourceIndex: this.selectedDatasourceIndex,
							selectedSources: JSON.parse(JSON.stringify(this.selectedSources)),
						})
					);
					localStorage.removeItem(this.DRAFT_STATE_KEY);

					ElMessage({
						message: data.msg,
						showClose: true,
						type: 'success',
						duration: 3000,
					});
					window.dispatchEvent(new Event(this.INSTALL_CHANGED_EVENT));
					await this.refreshOptions();
				} else {
					ElMessage({
						message: data.msg,
						showClose: true,
						type: 'error',
						duration: 3000,
					});
				}
			} catch (error) {
				console.error('Submission failed', error);
				ElMessage.error('Network Error');
			} finally {
				this.loading = false;
			}
		},
		handleClear() {
			this.selectedPolicyIndex = null;
			this.selectedDatasourceIndex = null;
			this.selectedSources = [];
			localStorage.removeItem(this.INSTALL_STATE_KEY);
			localStorage.removeItem(this.DRAFT_STATE_KEY);
		},
	},
};
</script>

<style scoped lang="scss">
.install-panel {
	display: grid;
	gap: 22px;
}

.panel-header,
.section-heading,
.action-bar,
.source-card__header,
.source-card__footer {
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
.summary-tags,
.section-heading__actions,
.builder-buttons,
.source-card__footer-actions {
	display: flex;
	flex-wrap: wrap;
	gap: 8px;
}

.summary-tags {
	margin-top: -6px;
}

.selection-grid {
	display: grid;
	grid-template-columns: repeat(2, minmax(0, 1fr));
	gap: 16px;
}

.selector-card,
.mapping-section,
.empty-state {
	border-radius: 22px;
	border: 1px solid #e2e8f0;
	background:
		linear-gradient(135deg, rgba(37, 99, 235, 0.04), transparent 34%),
		#ffffff;
}

.selector-card {
	padding: 18px;
	display: grid;
	gap: 12px;
}

.selector-card__label,
.section-heading__title {
	font-size: 13px;
	font-weight: 700;
	letter-spacing: 0.06em;
	text-transform: uppercase;
	color: #475569;
}

.selector-card__control {
	width: 100%;
}

.selector-card__meta,
.action-bar__summary,
.source-card__subtitle,
.source-card__footer-text {
	font-size: 13px;
	line-height: 1.6;
	color: #64748b;
}

.mapping-section {
	padding: 18px;
	display: grid;
	gap: 16px;
}

.source-grid {
	display: grid;
	grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
	gap: 14px;
}

.source-card {
	display: grid;
	gap: 14px;
	padding: 16px;
	border-radius: 18px;
	border: 1px solid #dbe4ee;
	background: #f8fafc;
	transition: border-color 0.2s ease, box-shadow 0.2s ease, transform 0.2s ease;
}

.source-card.is-ready {
	border-color: #86efac;
	background:
		linear-gradient(135deg, rgba(34, 197, 94, 0.06), transparent 36%),
		#f8fafc;
}

.source-card:hover {
	border-color: #93c5fd;
	box-shadow: 0 16px 36px rgba(37, 99, 235, 0.08);
	transform: translateY(-1px);
}

.source-card__title-group,
.field-stack,
.field-block {
	display: grid;
	gap: 8px;
	min-width: 0;
}

.source-card__title {
	font-size: 16px;
	font-weight: 700;
	color: #0f172a;
}

.field-stack {
	gap: 12px;
}

.field-block__label {
	font-size: 12px;
	font-weight: 700;
	color: #475569;
}

.field-block__control {
	width: 100%;
}

.mini-link {
	border: none;
	background: transparent;
	padding: 0;
	font-size: 12px;
	font-weight: 700;
	color: #2563eb;
	cursor: pointer;
}

.mini-link:disabled {
	color: #94a3b8;
	cursor: not-allowed;
}

.empty-state {
	min-height: 220px;
	display: grid;
	place-items: center;
	text-align: center;
	padding: 28px;
	background:
		linear-gradient(135deg, rgba(37, 99, 235, 0.05), transparent 38%),
		linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
}

.empty-state__icon {
	font-size: 34px;
	color: #2563eb;
	margin-bottom: 10px;
}

.empty-state__title {
	font-size: 18px;
	font-weight: 700;
	color: #0f172a;
}

.empty-state__subtitle {
	margin-top: 8px;
	font-size: 14px;
	line-height: 1.6;
	color: #64748b;
}

.action-bar {
	padding-top: 4px;
	border-top: 1px solid #e2e8f0;
}

@media (max-width: 900px) {
	.selection-grid {
		grid-template-columns: 1fr;
	}
}

@media (max-width: 768px) {
	.panel-header,
	.section-heading,
	.action-bar,
	.source-card__header,
	.source-card__footer {
		flex-direction: column;
		align-items: flex-start;
	}

	.source-grid {
		grid-template-columns: 1fr;
	}
}
</style>
