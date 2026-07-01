<template>
	<div class="viz-page layout-pd">
		<section class="section-card toolbar-card">
			<div class="section-heading">
				<div>
					<h3>Result Visualization</h3>
				</div>

				<div class="toolbar-actions">
					<div class="toolbar-field">
						<span class="toolbar-label">Datasource</span>
						<el-select
							v-model="selectedDataSource"
							:disabled="isSourceLoading"
							placeholder="Choose datasource"
							class="toolbar-select"
						>
							<el-option v-for="item in dataSourceList" :key="item.id" :label="item.label" :value="item.id" />
						</el-select>
					</div>

					<el-button
						type="primary"
						plain
						:disabled="!selectedDataSource"
						:loading="isUploading"
						@click="triggerConfigUpload"
					>
						Upload Config
					</el-button>
					<el-button type="primary" :disabled="!selectedDataSource" @click="exportTaskLog">Export Result Log</el-button>

					<input ref="uploadInput" type="file" accept=".yaml,.yml" hidden @change="handleFileUpload" />
				</div>
			</div>
		</section>

		<section class="section-card controls-card">
			<div class="section-heading section-heading--compact">
				<div>
					<h4>Active Visualizations</h4>
				</div>
				<el-tag v-if="selectedDataSource" :type="isSourceLoading ? 'warning' : 'info'" effect="plain">
					{{ isSourceLoading ? 'Loading' : `${currentVisualizationConfig.length} modules` }}
				</el-tag>
			</div>

			<div v-if="selectedDataSource && currentVisualizationConfig.length" class="checkbox-shell">
				<el-checkbox-group v-model="currentActiveVisualizationsArray" class="checkbox-grid">
					<el-checkbox v-for="viz in currentVisualizationConfig" :key="viz.id" :label="viz.id" class="module-checkbox">
						{{ viz.name }}
					</el-checkbox>
				</el-checkbox-group>
			</div>

			<div v-else class="empty-inline">
				{{ selectedDataSource ? 'No visualization modules' : 'Choose a datasource to begin' }}
			</div>
		</section>

		<section class="viz-grid-section">
			<template v-if="!selectedDataSource">
				<div class="empty-state">
					<div class="empty-state__title">No datasource selected</div>
				</div>
			</template>

			<template v-else-if="isSourceLoading">
				<div class="skeleton-grid">
					<div class="skeleton-card" v-for="n in 3" :key="n"></div>
				</div>
			</template>

			<template v-else>
				<el-row :gutter="16">
					<el-col
						v-for="viz in currentVisualizationConfig"
						:key="getVizKey(viz)"
						:xs="24"
						:sm="24"
						:md="getVisualizationSpan(viz.size, 'md')"
						:lg="getVisualizationSpan(viz.size, 'lg')"
						:xl="getVisualizationSpan(viz.size, 'xl')"
						v-show="componentsLoaded && currentActiveVisualizations.has(viz.id)"
						class="viz-grid-col"
					>
						<div class="viz-card">
							<div class="viz-card__header">
								<h3 class="viz-title">{{ viz.name }}</h3>
								<component
									:is="vizControls[viz.type]"
									:key="viz.type + '-' + viz.variablesHash"
									v-if="vizControls[viz.type] && selectedDataSource"
									:config="viz"
									:variable-states="variableStates[selectedDataSource]?.[viz.id] || {}"
									@update:variable-states="updateVariableStates(viz.id, $event)"
								/>
							</div>

							<component
								:is="visualizationComponents[viz.type]"
								v-if="componentsLoaded && visualizationComponents[viz.type] && selectedDataSource"
								:key="`${viz.type}-${selectedDataSource}-${viz.id}-${viz.variablesHash}`"
								:config="viz"
								:data="processedData[viz.id]"
								:variable-states="variableStates[selectedDataSource]?.[viz.id] || {}"
							/>
						</div>
					</el-col>
				</el-row>
			</template>
		</section>
	</div>
</template>

<script>
import { reactive, watch } from 'vue';
import mitt from 'mitt';
import { ElMessage } from 'element-plus';
import { registerVisualizationModules } from '../shared/visualizationRegistry';

const emitter = mitt();

export default {
	data() {
		return {
			selectedDataSource: null,
			dataSourceList: [],
			bufferedTaskCache: reactive({}),
			maxBufferedTaskCacheSize: 20,
			componentsLoaded: false,
			visualizationConfig: {},
			activeVisualizations: {},
			variableStates: {},
			visualizationComponents: {},
			vizControls: {},
			pollingInterval: null,
			isSourceLoading: false,
			isUploading: false,
		};
	},
	computed: {
		processedData() {
			const result = {};
			this.currentVisualizationConfig.forEach((viz) => {
				result[viz.id] = this.processVizData(viz);
			});
			return result;
		},
		currentVisualizationConfig() {
			return this.visualizationConfig[this.selectedDataSource] || [];
		},
		currentActiveVisualizations() {
			return this.activeVisualizations[this.selectedDataSource] || new Set();
		},
		currentActiveVisualizationsArray: {
			get() {
				return Array.from(this.currentActiveVisualizations);
			},
			set(newVal) {
				this.activeVisualizations[this.selectedDataSource] = new Set(newVal);
			},
		},
	},

	async created() {
		this.dataSourceList.forEach((source) => {
			this.bufferedTaskCache[source.id] = reactive([]);
		});

		watch(
			() => this.bufferedTaskCache,
			() => {},
			{ deep: true }
		);

		await this.autoRegisterComponents();
		this.componentsLoaded = true;
		await this.fetchDataSourceList();
		this.setupDataPolling();

		emitter.on('force-update-charts', () => {
			this.$nextTick(() => {
				if (!this.selectedDataSource) return;
				this.currentVisualizationConfig.forEach((viz) => {
					this.variableStates[this.selectedDataSource][viz.id] = {
						...this.variableStates[this.selectedDataSource][viz.id],
					};
				});
			});
		});
	},
	watch: {
		selectedDataSource(newVal) {
			if (newVal) {
				this.handleSourceChange(newVal);
			}
		},
	},
	methods: {
		calculateVariablesHash(variables) {
			return [...variables].sort().join('|');
		},
		getVizKey(viz) {
			return `${viz.id}-${viz.variablesHash}-${viz.size}`;
		},
		arraysEqual(a, b) {
			if (a === b) return true;
			if (!Array.isArray(a) || !Array.isArray(b)) return false;
			if (a.length !== b.length) return false;
			const sortedA = [...a].sort();
			const sortedB = [...b].sort();
			return sortedA.every((val, i) => val === sortedB[i]);
		},
		triggerConfigUpload() {
			if (!this.selectedDataSource) return;
			this.$refs.uploadInput.value = null;
			this.$refs.uploadInput.click();
		},
		async handleFileUpload(event) {
			const file = event.target.files[0];
			if (!file) return;

			this.isUploading = true;
			try {
				const formData = new FormData();
				formData.append('file', file);

				const response = await fetch(`/api/result_visualization_config/${this.selectedDataSource}`, {
					method: 'POST',
					body: formData,
				});
				const data = await response.json();
				await this.fetchVisualizationConfig(this.selectedDataSource);
				this.showMsg(data.state, data.msg);
			} catch (error) {
				ElMessage.error('System Error');
				console.error(error);
			} finally {
				this.isUploading = false;
			}
		},
		async handleSourceChange(sourceId) {
			if (!sourceId || !this.dataSourceList.some((s) => s.id === sourceId)) {
				console.error('Invalid source selection');
				return;
			}

			this.isSourceLoading = true;
			try {
				await this.fetchVisualizationConfig(sourceId);
			} catch (e) {
				console.error('Source change failed:', e);
			} finally {
				this.isSourceLoading = false;
			}

			this.$nextTick(() => {
				emitter.emit('force-update-charts');
			});
		},
		getVisualizationSpan(size, breakpoint) {
			const baseSize = size || 1;
			switch (breakpoint) {
				case 'xl':
					return Math.min(24, baseSize * 8);
				case 'lg':
					return Math.min(24, baseSize > 2 ? 24 : baseSize * 8);
				default:
					return baseSize > 1 ? 24 : 8;
			}
		},
		async autoRegisterComponents() {
			try {
				await registerVisualizationModules({
					templateModules: import.meta.glob('../shared/visualization/*Template.vue'),
					controlModules: import.meta.glob('../shared/visualization/*Controls.vue'),
					templatesTarget: this.visualizationComponents,
					controlsTarget: this.vizControls,
				});
			} catch (error) {
				console.error('Component auto-registration failed:', error);
			}
		},
		processVizData(vizConfig) {
			const sourceId = this.selectedDataSource;
			if (!sourceId || !this.bufferedTaskCache[sourceId]) return [];

			const validVizIds = new Set(this.currentVisualizationConfig.map((v) => String(v.id)));
			return this.bufferedTaskCache[sourceId]
				.filter((task) =>
					task.data?.some((item) => validVizIds.has(String(item.id)) && String(item.id) === String(vizConfig.id))
				)
				.map((task) => {
					const vizDataItem = task.data.find((item) => String(item.id) === String(vizConfig.id));
					return {
						taskId: String(task.task_id),
						...(vizDataItem?.data || {}),
					};
				});
		},
		updateVariableStates(vizId, newStates) {
			if (!this.selectedDataSource) return;

			const validVars = this.currentVisualizationConfig.find((v) => v.id === vizId)?.variables || [];
			this.variableStates[this.selectedDataSource][vizId] = validVars.reduce((acc, varName) => {
				acc[varName] = newStates[varName] ?? true;
				return acc;
			}, {});

			emitter.emit('force-update-charts');
		},
		async fetchDataSourceList() {
			try {
				const response = await fetch('/api/source_list');
				const data = await response.json();

				this.dataSourceList = data.map((source) => ({
					...source,
					id: String(source.id),
				}));
				this.dataSourceList.forEach((source) => {
					this.bufferedTaskCache[source.id] = reactive([]);
				});

				if (!this.selectedDataSource && this.dataSourceList.length) {
					this.selectedDataSource = this.dataSourceList[0].id;
				}
			} catch (error) {
				console.error('Failed to fetch data sources:', error);
			}
		},
		async fetchVisualizationConfig(sourceId) {
			try {
				const response = await fetch(`/api/result_visualization_config/${sourceId}`);
				const data = await response.json();

				const processedConfig = data.map((viz) =>
					reactive({
						...viz,
						id: String(viz.id),
						variables: [...(viz.variables || [])],
						size: Math.min(3, Math.max(1, parseInt(viz.size) || 1)),
						variablesHash: this.calculateVariablesHash(viz.variables || []),
					})
				);

				this.visualizationConfig[sourceId] = processedConfig;
				this.activeVisualizations[sourceId] = new Set();
				this.variableStates[sourceId] = reactive({});

				processedConfig.forEach((viz) => {
					this.activeVisualizations[sourceId].add(viz.id);
					this.variableStates[sourceId][viz.id] = viz.variables.reduce((acc, varName) => {
						acc[varName] = true;
						return acc;
					}, {});
				});
			} catch (error) {
				console.error('Failed to fetch visualization config:', error);
			}
		},
		async getLatestResultData() {
			try {
				const response = await fetch('/api/task_result');
				const data = await response.json();
				const newCache = { ...this.bufferedTaskCache };
				const configUpdates = {};

				Object.entries(data).forEach(([sourceIdStr, tasks]) => {
					const sourceId = String(sourceIdStr);
					if (!Array.isArray(tasks)) return;

					const validTasks = tasks
						.filter((task) => task?.task_id && Array.isArray(task.data))
						.map((task) => ({
							task_id: task.task_id,
							data: task.data.map((item) => ({
								id: String(item.id) || 'unknown',
								data: item.data || {},
							})),
						}));

					newCache[sourceId] = [...(newCache[sourceId] || []), ...validTasks].slice(-this.maxBufferedTaskCacheSize);

					tasks.forEach((task) => {
						task.data?.forEach((item) => {
							const vizId = String(item.id);
							const newVariables = Object.keys(item.data || {});
							const vizConfig = (this.visualizationConfig[sourceId] || []).find((v) => v.id === vizId);

							if (vizConfig && !this.arraysEqual(vizConfig.variables, newVariables)) {
								configUpdates[sourceId] = configUpdates[sourceId] || [];
								configUpdates[sourceId].push({
									vizId,
									newVariables,
								});
							}
						});
					});
				});

				Object.entries(configUpdates).forEach(([sourceId, updates]) => {
					const newConfig = [...(this.visualizationConfig[sourceId] || [])];
					updates.forEach(({ vizId, newVariables }) => {
						const index = newConfig.findIndex((v) => v.id === vizId);
						if (index !== -1) {
							const updatedViz = {
								...newConfig[index],
								variables: [...newVariables],
								variablesHash: this.calculateVariablesHash(newVariables),
							};
							newConfig.splice(index, 1, updatedViz);
						}
					});
					this.visualizationConfig[sourceId] = newConfig;
				});

				this.bufferedTaskCache = reactive({ ...newCache });

				if (this.selectedDataSource && this.visualizationConfig[this.selectedDataSource]) {
					this.visualizationConfig[this.selectedDataSource] = this.visualizationConfig[this.selectedDataSource].map(
						(cfg) => ({ ...cfg })
					);
				}

				this.$nextTick(() => {
					emitter.emit('force-update-charts');
				});
			} catch (error) {
				console.error('Data fetch failed:', error);
			}
		},
		setupDataPolling() {
			this.getLatestResultData();
			this.pollingInterval = setInterval(() => {
				this.getLatestResultData();
			}, 2000);
		},
		exportTaskLog() {
			const link = document.createElement('a');
			link.href = '/api/download_log';
			link.rel = 'noopener';
			document.body.appendChild(link);
			link.click();
			link.remove();
		},
		showMsg(state, msg) {
			ElMessage({
				message: msg,
				showClose: true,
				type: state === 'success' ? 'success' : 'error',
				duration: 3000,
			});
		},
	},
	beforeUnmount() {
		if (this.pollingInterval) {
			clearInterval(this.pollingInterval);
			this.pollingInterval = null;
		}
		emitter.off('force-update-charts');
	},
};
</script>

<style scoped lang="scss">
.viz-page {
	padding: 20px;
	display: grid;
	gap: 20px;
	background: radial-gradient(circle at top left, rgba(37, 99, 235, 0.08), transparent 24%),
		radial-gradient(circle at bottom right, rgba(14, 165, 233, 0.08), transparent 24%);
}

.section-card {
	padding: 22px;
	border-radius: 28px;
	border: 1px solid #e2e8f0;
	background: linear-gradient(135deg, rgba(255, 255, 255, 0.96), rgba(248, 250, 252, 0.96)), #ffffff;
	box-shadow: 0 22px 48px rgba(15, 23, 42, 0.06);
}

.section-heading {
	display: flex;
	align-items: flex-start;
	justify-content: space-between;
	gap: 16px;
}

.section-heading--compact {
	align-items: center;
}

.section-heading h3,
.section-heading h4 {
	margin: 0;
	color: #0f172a;
}

.section-heading h3 {
	font-size: 24px;
}

.section-heading h4 {
	font-size: 15px;
}

.toolbar-actions {
	display: flex;
	flex-wrap: wrap;
	align-items: center;
	justify-content: flex-end;
	gap: 10px;
}

.toolbar-field {
	display: flex;
	align-items: center;
	gap: 10px;
	min-width: 240px;
}

.toolbar-label {
	font-size: 12px;
	font-weight: 700;
	color: #475569;
	text-transform: uppercase;
	letter-spacing: 0.06em;
	white-space: nowrap;
}

.toolbar-select {
	width: 220px;
}

.checkbox-shell {
	margin-top: 14px;
}

.checkbox-grid {
	display: flex;
	flex-wrap: wrap;
	gap: 10px 18px;
}

.module-checkbox {
	margin: 0;
}

.viz-grid-section {
	min-height: 160px;
}

.viz-grid-col {
	margin-bottom: 16px;
}

.viz-card {
	height: 500px;
	min-height: 500px;
	display: flex;
	flex-direction: column;
	border-radius: 24px;
	border: 1px solid #e2e8f0;
	background: linear-gradient(135deg, rgba(37, 99, 235, 0.04), transparent 36%), #ffffff;
	box-shadow: 0 18px 40px rgba(15, 23, 42, 0.05);
	overflow: hidden;
}

.viz-card__header {
	padding: 14px 16px 12px;
	border-bottom: 1px solid #e2e8f0;
	display: grid;
	gap: 10px;
}

.viz-title {
	margin: 0;
	font-size: 17px;
	font-weight: 700;
	color: #0f172a;
	text-align: center;
}

.empty-inline,
.empty-state {
	display: grid;
	place-items: center;
	text-align: center;
	border: 1px dashed #cbd5e1;
	border-radius: 20px;
	background: #f8fafc;
	color: #64748b;
}

.empty-inline {
	min-height: 84px;
	margin-top: 14px;
}

.empty-state {
	min-height: 220px;
}

.empty-state__title {
	font-size: 18px;
	font-weight: 700;
	color: #0f172a;
}

.skeleton-grid {
	display: grid;
	grid-template-columns: repeat(3, minmax(0, 1fr));
	gap: 16px;
}

.skeleton-card {
	height: 280px;
	border-radius: 24px;
	background: #f5f7fa;
	position: relative;
	overflow: hidden;
}

.skeleton-card::after {
	content: '';
	position: absolute;
	top: 0;
	left: -100%;
	width: 100%;
	height: 100%;
	background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.65), transparent);
	animation: skeleton-flash 1.5s infinite;
}

@keyframes skeleton-flash {
	100% {
		left: 100%;
	}
}

@media (max-width: 992px) {
	.toolbar-actions {
		justify-content: flex-start;
	}

	.skeleton-grid {
		grid-template-columns: 1fr;
	}
}

@media (max-width: 768px) {
	.viz-page {
		padding: 14px;
		gap: 16px;
	}

	.section-card {
		padding: 18px;
		border-radius: 22px;
	}

	.section-heading {
		flex-direction: column;
		align-items: flex-start;
	}

	.toolbar-field {
		min-width: 100%;
		flex-wrap: wrap;
	}

	.toolbar-select {
		width: 100%;
	}
}
</style>
