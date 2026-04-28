<template>
	<div class="viz-page layout-pd">
		<section class="section-card toolbar-card">
			<div class="section-heading section-heading--compact">
				<div>
					<h3>System Visualization</h3>
				</div>

				<div class="toolbar-actions">
					<el-button type="primary" @click="exportSystemLog">Export System Log</el-button>
				</div>
			</div>
		</section>

		<section class="section-card controls-card">
			<div class="section-heading section-heading--compact">
				<div>
					<h4>Active Visualizations</h4>
				</div>
				<el-tag v-if="visualizationConfig.length" :type="isConfigLoading ? 'warning' : 'info'" effect="plain">
					{{ isConfigLoading ? 'Loading' : `${visualizationConfig.length} modules` }}
				</el-tag>
			</div>

			<div v-if="visualizationConfig.length" class="checkbox-shell">
				<el-checkbox-group v-model="currentActiveVisualizationsArray" class="checkbox-grid">
					<el-checkbox v-for="viz in visualizationConfig" :key="viz.id" :label="viz.id" class="module-checkbox">
						{{ viz.name }}
					</el-checkbox>
				</el-checkbox-group>
			</div>

			<div v-else class="empty-inline">
				{{ isConfigLoading ? 'Loading modules' : 'No visualization modules' }}
			</div>
		</section>

		<section class="viz-grid-section">
			<template v-if="isConfigLoading || !componentsLoaded">
				<div class="skeleton-grid">
					<div class="skeleton-card" v-for="n in 3" :key="n"></div>
				</div>
			</template>

			<template v-else-if="!visualizationConfig.length">
				<div class="empty-state">
					<div class="empty-state__title">No visualization modules</div>
				</div>
			</template>

			<template v-else>
				<el-row :gutter="16">
					<el-col
						v-for="viz in visualizationConfig"
						:key="getVizKey(viz)"
						:xs="24"
						:sm="24"
						:md="getVisualizationSpan(viz.size, 'md')"
						:lg="getVisualizationSpan(viz.size, 'lg')"
						:xl="getVisualizationSpan(viz.size, 'xl')"
						v-show="currentActiveVisualizations.has(viz.id)"
						class="viz-grid-col"
					>
						<div class="viz-card">
							<div class="viz-card__header">
								<h3 class="viz-title">{{ viz.name }}</h3>
								<component
									:is="vizControls[viz.type]"
									:key="viz.type + '-' + viz.variablesHash"
									v-if="vizControls[viz.type]"
									:config="viz"
									:variable-states="variableStates[viz.id] || {}"
									@update:variable-states="updateVariableStates(viz.id, $event)"
								/>
							</div>

							<component
								:is="visualizationComponents[viz.type]"
								v-if="visualizationComponents[viz.type]"
								:key="`${viz.type}-${viz.id}-${viz.variablesHash}`"
								:config="viz"
								:data="processedData[viz.id]"
								:variable-states="variableStates[viz.id] || {}"
							/>
						</div>
					</el-col>
				</el-row>
			</template>
		</section>
	</div>
</template>

<script>
import { markRaw, reactive } from 'vue';
import mitt from 'mitt';
import { useSystemParametersStore } from '/@/stores/systemParameters';

const emitter = mitt();

export default {
	data() {
		return {
			componentsLoaded: false,
			isConfigLoading: false,
			visualizationConfig: [],
			activeVisualizations: new Set(),
			variableStates: {},
			visualizationComponents: {},
			vizControls: {},
			// stores
			sysParamsStore: null,
		};
	},
	computed: {
		processedData() {
			const buffer = this.sysParamsStore?.bufferedTaskCache || [];
			const result = {};
			this.visualizationConfig.forEach((viz) => {
				result[viz.id] = this.processVizData(viz, buffer);
			});
			return result;
		},
		currentActiveVisualizations() {
			return this.activeVisualizations || new Set();
		},
		currentActiveVisualizationsArray: {
			get() {
				return Array.from(this.activeVisualizations);
			},
			set(newVal) {
				this.activeVisualizations = new Set(newVal);
			},
		},
	},

	async created() {
		this.sysParamsStore = useSystemParametersStore();

		await this.autoRegisterComponents();
		this.componentsLoaded = true;
		await this.fetchVisualizationConfig();

		this.$watch(
			() => this.sysParamsStore.bufferedTaskCache,
			(buffer) => {
				this.syncVariablesFromBuffer(buffer || []);
			},
			{ deep: true, immediate: true }
		);

		emitter.on('force-update-charts', () => {
			this.$nextTick(() => {
				this.visualizationConfig.forEach((viz) => {
					this.variableStates[viz.id] = { ...this.variableStates[viz.id] };
				});
			});
		});
	},

	methods: {
		calculateVariablesHash(variables) {
			return [...(variables || [])].sort().join('|');
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
				const modules = import.meta.glob('./visualization/*Template.vue');
				const controls = import.meta.glob('./visualization/*Controls.vue');

				await Promise.all([
					...Object.entries(modules).map(async ([path, loader]) => {
						const type = path.split('/').pop().replace('Template.vue', '').toLowerCase();
						const comp = await loader();
						this.visualizationComponents[type] = markRaw(comp.default);
					}),
					...Object.entries(controls).map(async ([path, loader]) => {
						const type = path.split('/').pop().replace('Controls.vue', '').toLowerCase();
						const comp = await loader();
						this.vizControls[type] = markRaw(comp.default);
					}),
				]);
			} catch (error) {
				console.error('Component registration failed:', error);
			}
		},

		processVizData(vizConfig, buffer) {
			if (!buffer.length) return [];

			try {
				return buffer
					.filter((task) => {
						return task.data?.some((item) => String(item.id) === String(vizConfig.id));
					})
					.map((task) => {
						const vizDataItem = task.data.find((item) => String(item.id) === String(vizConfig.id));
						return {
							taskId: task.timestamp,
							timestamp: task.timestamp,
							...(vizDataItem?.data || {}),
						};
					});
			} catch (error) {
				console.error('Data process error:', error);
				return [];
			}
		},

		updateVariableStates(vizId, newStates) {
			const validVars = this.visualizationConfig.find((v) => v.id === vizId)?.variables || [];
			this.variableStates[vizId] = validVars.reduce((acc, varName) => {
				acc[varName] = newStates[varName] ?? true;
				return acc;
			}, {});

			emitter.emit('force-update-charts');
		},

		async fetchVisualizationConfig() {
			this.isConfigLoading = true;
			try {
				const response = await fetch('/api/system_visualization_config');
				const data = await response.json();

				this.visualizationConfig = data.map((viz) =>
					reactive({
						...viz,
						id: String(viz.id),
						size: Math.min(3, Math.max(1, parseInt(viz.size) || 1)),
						variables: [...(viz.variables || [])],
						variablesHash: this.calculateVariablesHash(viz.variables),
					})
				);

				this.activeVisualizations = new Set(this.visualizationConfig.map((viz) => viz.id));

				this.variableStates = this.visualizationConfig.reduce((acc, viz) => {
					acc[viz.id] = viz.variables.reduce((vars, varName) => {
						vars[varName] = true;
						return vars;
					}, {});
					return acc;
				}, {});
			} catch (error) {
				console.error('Failed to fetch visualization config:', error);
			} finally {
				this.isConfigLoading = false;
			}
		},

		exportSystemLog() {
			const link = document.createElement('a');
			link.href = '/api/download_system_log';
			link.rel = 'noopener';
			document.body.appendChild(link);
			link.click();
			link.remove();
		},

		syncVariablesFromBuffer(buffer) {
			try {
				const latestByViz = new Map();
				buffer.forEach((task) => {
					(task.data || []).forEach((item) => {
						const vizId = String(item.id);
						latestByViz.set(vizId, Object.keys(item.data || {}));
					});
				});
				latestByViz.forEach((newVars, vizId) => {
					const vizIndex = this.visualizationConfig.findIndex((v) => v.id === vizId);
					if (vizIndex !== -1) {
						const curr = this.visualizationConfig[vizIndex];
						if (!this.arraysEqual(curr.variables, newVars)) {
							const updatedViz = {
								...curr,
								variables: [...newVars],
								variablesHash: this.calculateVariablesHash(newVars),
							};
							this.visualizationConfig.splice(vizIndex, 1, reactive(updatedViz));

							const currentState = this.variableStates[vizId] || {};
							this.variableStates[vizId] = newVars.reduce((acc, name) => {
								acc[name] = name in currentState ? currentState[name] : true;
								return acc;
							}, {});
						}
					}
				});
			} catch (e) {
				// no-op
			}
		},
	},

	beforeUnmount() {
		emitter.off('force-update-charts');
	},
};
</script>

<style scoped lang="scss">
.viz-page {
	padding: 20px;
	display: grid;
	gap: 20px;
	background:
		radial-gradient(circle at top left, rgba(37, 99, 235, 0.08), transparent 24%),
		radial-gradient(circle at bottom right, rgba(14, 165, 233, 0.08), transparent 24%);
}

.section-card {
	padding: 22px;
	border-radius: 28px;
	border: 1px solid #e2e8f0;
	background:
		linear-gradient(135deg, rgba(255, 255, 255, 0.96), rgba(248, 250, 252, 0.96)),
		#ffffff;
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
	align-items: center;
	justify-content: flex-end;
	gap: 10px;
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
	background:
		linear-gradient(135deg, rgba(37, 99, 235, 0.04), transparent 36%),
		#ffffff;
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
}
</style>
