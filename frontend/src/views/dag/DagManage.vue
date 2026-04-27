<template>
	<div class="outline dag-manage-page">
		<section class="section-card composer-section">
			<div class="section-heading">
				<div>
					<h3>Add Application Dag</h3>
				</div>
			</div>

			<div class="builder-grid">
				<div class="builder-main">
					<div class="field-block">
						<div class="new-dag-font-style">Dag Name</div>
						<el-input v-model="newInputName" clearable placeholder="Please fill the dag name" />
					</div>

					<div class="builder-actions">
						<div class="builder-buttons">
							<el-button type="warning" plain @click="draw">{{ drawing ? 'Hide Canvas' : 'Open Canvas' }}</el-button>
							<el-button type="primary" round :disabled="!drawing || !flowNodes.length" @click="handleNewSubmit">Add Dag</el-button>
							<el-button round :disabled="!drawing" @click="clearInput">Reset</el-button>
						</div>
					</div>

					<div v-if="!drawing" class="canvas-placeholder">
						<el-icon class="canvas-placeholder__icon">
							<MagicStick />
						</el-icon>
						<div class="canvas-placeholder__title">Open the canvas to start drawing</div>
					</div>

					<div v-else :class="['draw-container', { 'is-drag-over': isDragOver }]" @drop="handleCanvasDrop">
						<div class="canvas-status-bar">
							<div class="canvas-metrics">
								<el-tag type="info" effect="plain">
									<el-icon><Connection /></el-icon>
									{{ flowNodes.length }} nodes
								</el-tag>
								<el-tag type="success" effect="plain">
									<el-icon><Link /></el-icon>
									{{ flowEdges.length }} links
								</el-tag>
							</div>
						</div>

						<VueFlow
							:id="mainFlowId"
							class="main-flow"
							:nodes="flowNodes"
							:edges="flowEdges"
							:default-viewport="{ zoom: 1 }"
							:min-zoom="0.5"
							:max-zoom="2"
							:fit-view-on-init="true"
							:snap-to-grid="true"
							:snap-grid="[24, 24]"
							:nodes-draggable="true"
							:nodes-connectable="true"
							:elements-selectable="true"
							:default-edge-options="defaultEdgeOptions"
							@dragover="onDragOver"
							@dragleave="onDragLeave"
						>
							<div v-if="!flowNodes.length" class="drag-tip">
								<el-icon class="tip-icon">
									<MagicStick />
								</el-icon>
								<span>Drag services here</span>
							</div>

							<Background pattern-color="#cbd5e1" :gap="20" />
							<div class="minimap-toggle-wrap" :class="{ expanded: isMiniMapExpanded }">
								<button type="button" class="minimap-toggle" @click="toggleMiniMap">
									{{ isMiniMapExpanded ? 'Hide Map' : 'Map' }}
								</button>
							</div>
							<MiniMap v-if="isMiniMapExpanded" class="dag-minimap" />
							<Controls position="bottom-right" />

							<Panel class="process-panel" position="top-right">
								<div class="layout-panel">
									<button type="button" title="fit graph" @click="focusCanvas">Fit</button>
									<button type="button" title="set horizontal layout" @click="applyLayoutGraph('LR')">
										<Icon name="horizontal" />
									</button>
									<button type="button" title="set vertical layout" @click="applyLayoutGraph('TB')">
										<Icon name="vertical" />
									</button>
								</div>
							</Panel>
						</VueFlow>
					</div>
				</div>

				<aside class="service-sidebar">
					<div class="sidebar-header">
						<div class="new-dag-font-style">Service Containers</div>
					</div>

					<div class="service-grid">
						<button
							v-for="service in services"
							:key="service.id"
							type="button"
							class="service-card"
							:style="getServiceCardStyle(service)"
							draggable="true"
							@dragstart="onDragStart($event, '', service)"
						>
							<div class="service-card__header">
								<span class="service-card__name">{{ formatServiceName(service) }}</span>
								<span class="service-card__drag">Drag</span>
							</div>
							<div class="service-card__meta">
								<span class="service-chip">IN {{ formatIoValue(service.input) }}</span>
								<span class="service-chip">OUT {{ formatIoValue(service.output) }}</span>
							</div>
						</button>
					</div>
				</aside>
			</div>
		</section>

		<section class="section-card dag-list-section">
			<div class="section-heading">
				<div>
					<h3>Current Application Dags</h3>
				</div>
			</div>

			<el-table :data="dagList" row-key="dag_id" empty-text="No application DAGs yet" style="width: 100%">
				<el-table-column label="Dag Name" min-width="220">
					<template #default="scope">
						<div class="dag-name-cell">
							<div class="dag-name">{{ scope.row.dag_name }}</div>
							<div class="dag-subtitle">
								{{ getStartNodes(scope.row.dag).length }} start nodes · {{ getDagNodeCount(scope.row.dag) }} services
							</div>
						</div>
					</template>
				</el-table-column>

				<el-table-column label="Overview" min-width="520">
					<template #default="scope">
						<el-popover
							trigger="hover"
							placement="left-start"
							:width="480"
							popper-class="dag-preview-popper"
							@show="prepareDagPreview(scope.row)"
						>
							<template #reference>
								<button type="button" class="dag-overview-button">
									<div class="dag-overview-header">
										<div class="dag-pill-group">
											<span
												v-for="label in summarizeServices(scope.row.dag, previewNodeLimit)"
												:key="label"
												class="service-pill"
											>
												{{ label }}
											</span>
											<span v-if="getOverflowServiceCount(scope.row.dag, previewNodeLimit) > 0" class="service-pill service-pill--muted">
												+{{ getOverflowServiceCount(scope.row.dag, previewNodeLimit) }}
											</span>
										</div>
									</div>

									<div class="stats">
										<el-tag type="info" size="small" effect="plain">
											<el-icon><Connection /></el-icon>
											{{ getDagNodeCount(scope.row.dag) }} nodes
										</el-tag>
										<el-tag type="success" size="small" effect="plain">
											<el-icon><Link /></el-icon>
											{{ countEdges(scope.row.dag) }} links
										</el-tag>
										<el-tag type="warning" size="small" effect="plain">
											start · {{ getStartNodes(scope.row.dag).length }}
										</el-tag>
									</div>
								</button>
							</template>

							<div class="dag-hover-panel">
								<div class="dag-hover-header">
									<div class="dag-title">{{ scope.row.dag_name }}</div>
									<div class="dag-hover-subtitle">
										{{ getDagNodeCount(scope.row.dag) }} nodes · {{ countEdges(scope.row.dag) }} links
									</div>
								</div>

								<div class="dag-pill-group dag-pill-group--wrap">
									<span v-for="label in summarizeServices(scope.row.dag, getDagNodeCount(scope.row.dag))" :key="label" class="service-pill">
										{{ label }}
									</span>
								</div>

								<div class="dag-preview-shell">
									<VueFlow
										:id="`dag-preview-${scope.row.dag_id}`"
										class="preview-flow"
										:nodes="scope.row.nodeList"
										:edges="scope.row.lineList"
										:fit-view-on-init="true"
										:nodes-draggable="false"
										:nodes-connectable="false"
										:elements-selectable="false"
										:zoom-on-scroll="false"
										:pan-on-drag="false"
									>
										<Background pattern-color="#e2e8f0" :gap="24" />
									</VueFlow>
								</div>
							</div>
						</el-popover>
					</template>
				</el-table-column>

				<el-table-column label="Action" width="120" align="center">
					<template #default="scope">
						<el-button size="small" type="danger" plain @click="deleteWorkflow(scope.row.dag_id)">Delete</el-button>
					</template>
				</el-table-column>
			</el-table>
		</section>
	</div>
</template>

<script>
import { ElButton, ElInput, ElMessage, ElPopover, ElTable, ElTableColumn, ElTag } from 'element-plus';
import { nextTick, ref } from 'vue';
import { MarkerType, Panel, useVueFlow, VueFlow } from '@vue-flow/core';
import { Controls } from '@vue-flow/controls';
import { Background } from '@vue-flow/background';
import { MiniMap } from '@vue-flow/minimap';
import useDragAndDrop from './useDnD';
import Icon from './Icon.vue';
import { useLayout } from './useLayout';
import { getServiceNodeFontSize, getServiceTone } from './nodePalette';
import { Connection, Link, MagicStick } from '@element-plus/icons-vue';

const MAIN_FLOW_ID = 'dag-builder-main';
const PREVIEW_NODE_LIMIT = 4;

export default {
	name: 'DagManage',
	components: {
		ElTable,
		ElTableColumn,
		ElPopover,
		ElTag,
		ElInput,
		ElButton,
		VueFlow,
		Background,
		MiniMap,
		Controls,
		Icon,
		Panel,
		Connection,
		Link,
		MagicStick,
	},
	setup() {
		const { onInit, onConnect, fitView } = useVueFlow({ id: MAIN_FLOW_ID });
		const { onDragOver, onDrop, onDragLeave, isDragOver, onDragStart, serviceData } = useDragAndDrop(MAIN_FLOW_ID);
		const layoutMethods = useLayout(MAIN_FLOW_ID);

		const flowNodes = ref([]);
		const flowEdges = ref([]);
		const flowNodeMap = ref({});
		const defaultEdgeOptions = {
			type: 'smoothstep',
			markerEnd: MarkerType.ArrowClosed,
			style: {
				stroke: '#64748b',
				strokeWidth: 2,
			},
		};

		const applyLayoutGraph = async (direction) => {
			try {
				const layoutNodes = layoutMethods.layout([...flowNodes.value], [...flowEdges.value], direction);
				flowNodes.value = layoutNodes;
				await nextTick();
				fitView();
			} catch (error) {
				console.error('Layout failed:', error);
				ElMessage.error('DAG layout error');
			}
		};

		const focusCanvas = async () => {
			await nextTick();
			fitView();
		};

		onInit((vueFlowInstance) => {
			vueFlowInstance.fitView();
		});

		onConnect((connection) => {
			if (!connection.source || !connection.target) {
				return;
			}

			if (connection.source === connection.target) {
				ElMessage.warning('A service cannot connect to itself');
				return;
			}

			const edgeId = `${connection.source}-${connection.target}`;
			if (flowEdges.value.some((edge) => edge.id === edgeId)) {
				ElMessage.warning('This connection already exists');
				return;
			}

			const sourceNode = flowNodeMap.value[connection.source];
			const targetNode = flowNodeMap.value[connection.target];
			if (!sourceNode || !targetNode) {
				return;
			}

			flowEdges.value.push({
				id: edgeId,
				source: connection.source,
				target: connection.target,
				...defaultEdgeOptions,
			});
			sourceNode.data.succ = Array.from(new Set([...(sourceNode.data.succ || []), connection.target]));
			targetNode.data.prev = Array.from(new Set([...(targetNode.data.prev || []), connection.source]));
		});

			return {
				mainFlowId: MAIN_FLOW_ID,
				onDragOver,
			onDrop,
			onDragLeave,
			isDragOver,
			onDragStart,
			serviceData,
			focusCanvas,
			applyLayoutGraph,
			fitView,
			defaultEdgeOptions,
			flowNodes,
			flowEdges,
			flowNodeMap,
			...layoutMethods,
		};
	},
	data() {
		return {
			services: [],
			newInputName: '',
			drawing: false,
			dagList: [],
			refreshTimer: null,
			previewNodeLimit: PREVIEW_NODE_LIMIT,
			isMiniMapExpanded: false,
		};
	},
	methods: {
		flushDrawData() {
			this.flowNodes = [];
			this.flowEdges = [];
			this.flowNodeMap = {};
		},
		draw() {
			this.drawing = !this.drawing;

			if (this.drawing) {
				this.$nextTick(() => {
					this.focusCanvas();
				});
			}
		},
		clearInput() {
			this.newInputName = '';
			this.flushDrawData();
		},
		toggleMiniMap() {
			this.isMiniMapExpanded = !this.isMiniMapExpanded;
		},
		async handleCanvasDrop(event) {
			if (!this.serviceData) {
				return;
			}

			const nodeId = this.serviceData.id;
			if (this.flowNodeMap[nodeId]) {
				ElMessage.warning(`Service "${this.formatServiceName(this.serviceData)}" is already on the canvas`);
				return;
			}

			this.onDrop(event, this.flowNodes, this.flowNodeMap);
			await nextTick();
			this.focusCanvas();
		},
		async deleteWorkflow(dag_id) {
			try {
				const response = await fetch('/api/dag_workflow', {
					method: 'DELETE',
					body: JSON.stringify({ dag_id }),
				});
				const data = await response.json();
				this.showMsg(data.state, data.msg);

				if (data.state === 'success') {
					await this.getDagList();
				}
			} catch (error) {
				ElMessage.error('Network error');
				console.error(error);
			}
		},
		handleNewSubmit() {
			if (!this.newInputName) {
				ElMessage.error('Please fill the dag name');
				return;
			}

			if (!this.flowNodes.length) {
				ElMessage.error('Please choose services');
				return;
			}

			const graph = { _start: [] };
			for (const flowNode of this.flowNodes) {
				const serviceId = flowNode.id;
				if (graph[serviceId]) {
					throw new Error(`Duplicate service_id: ${serviceId}`);
				}

				const prev = flowNode.data?.prev ? [...flowNode.data.prev] : [];
				const succ = flowNode.data?.succ ? [...flowNode.data.succ] : [];
				graph[serviceId] = {
					id: serviceId,
					prev,
					succ,
					service_id: flowNode.data?.service_id || serviceId,
				};

				if (prev.length === 0) {
					graph._start.push(serviceId);
				}
			}

			this.updateDagList({
				dag_name: this.newInputName,
				dag: graph,
			});
		},
		async getDagList() {
			try {
				const response = await fetch('/api/dag_workflow');
				const data = await response.json();
				this.dagList = data.map((dag) => this.buildDagPresentation(dag));
			} catch (error) {
				console.error('Error fetching data:', error);
			}
		},
		fetchData() {
			this.getDagList();
		},
		showMsg(state, msg) {
			ElMessage({
				message: msg,
				showClose: true,
				type: state === 'success' ? 'success' : 'error',
				duration: 3000,
			});
		},
		async updateDagList(data) {
			try {
				const response = await fetch('/api/dag_workflow', {
					method: 'POST',
					headers: {
						'Content-Type': 'application/json',
					},
					body: JSON.stringify(data),
				});
				const result = await response.json();
				this.showMsg(result.state, result.msg);

				if (result.state === 'success') {
					await this.getDagList();
					this.clearInput();
					this.drawing = false;
				}
			} catch (error) {
				console.error('Error sending data:', error);
			}
		},
		async getServiceList() {
			try {
				const response = await fetch('/api/service');
				this.services = await response.json();
			} catch (error) {
				console.error('Error fetching services:', error);
				ElMessage.error('Failed to fetch services');
			}
		},
		prepareDagPreview(row) {
			if (!row?.nodeList || !row?.lineList) {
				Object.assign(row, this.buildDagPresentation(row));
			}
		},
		countEdges(dag) {
			return this.generateEdges(dag).length;
		},
		getDagNodeCount(dag) {
			return Object.keys(dag || {}).filter((key) => key !== '_start').length;
		},
		getStartNodes(dag) {
			const startNodes = dag?._start;
			return Array.isArray(startNodes) ? startNodes : [];
		},
		getDagServiceLabels(dag) {
			return Object.entries(dag || {})
				.filter(([key]) => key !== '_start')
				.map(([key, node]) => node?.service_id || node?.id || key);
		},
		summarizeServices(dag, limit) {
			return this.getDagServiceLabels(dag).slice(0, limit);
		},
		getOverflowServiceCount(dag, limit) {
			return Math.max(this.getDagServiceLabels(dag).length - limit, 0);
		},
		getNodeTone(key) {
			return getServiceTone(key);
		},
		getNodeStyle(key, label = key) {
			const tone = this.getNodeTone(key);
			return {
				backgroundColor: tone.background,
				border: `1px solid ${tone.border}`,
				borderLeft: `4px solid ${tone.accent}`,
				borderRadius: '14px',
				boxShadow: '0 6px 14px rgba(15, 23, 42, 0.06)',
				color: '#0f172a',
				fontSize: getServiceNodeFontSize(label),
			};
		},
		getServiceCardStyle(service) {
			const tone = this.getNodeTone(service?.id || service?.name);
			return {
				backgroundColor: tone.background,
				borderColor: tone.border,
				'--service-accent': tone.accent,
			};
		},
		formatServiceName(service) {
			return service?.name || service?.id || 'Unknown Service';
		},
		formatIoValue(value) {
			return value || '-';
		},
		parseDag(dag) {
			return Object.keys(dag || {})
				.filter((key) => key !== '_start')
				.map((key) => {
					const label = dag[key]?.service_id || dag[key]?.id || key;
					return {
						id: key,
						class: 'dag-node',
						data: {
							label,
							service_id: dag[key]?.service_id || key,
						},
						dimensions: { width: 96, height: 36 },
						style: this.getNodeStyle(key, label),
					};
				});
		},
		generateEdges(dag) {
			const edges = [];
			for (const [source, node] of Object.entries(dag || {})) {
				if (source === '_start' || !Array.isArray(node?.succ)) {
					continue;
				}

				node.succ.forEach((target) => {
					edges.push({
						id: `${source}-${target}`,
						source,
						target,
						type: 'smoothstep',
						markerEnd: MarkerType.ArrowClosed,
						style: {
							stroke: '#64748b',
							strokeWidth: 2,
						},
					});
				});
			}
			return edges;
		},
		buildDagPresentation(dag) {
			const nodeList = this.parseDag(dag.dag);
			const lineList = this.generateEdges(dag.dag);
			const layoutNodes = this.layout(nodeList, lineList, 'LR');

			return {
				...dag,
				nodeList: layoutNodes,
				lineList,
			};
		},
	},
	mounted() {
		this.fetchData();
		this.getServiceList();
		this.refreshTimer = window.setInterval(() => {
			this.fetchData();
		}, 5000);
	},
	beforeUnmount() {
		if (this.refreshTimer) {
			clearInterval(this.refreshTimer);
			this.refreshTimer = null;
		}
	},
};
</script>

<style scoped lang="scss">
.dag-manage-page {
	--dag-service-name-font-size: 12px;
	padding: 20px;
	display: grid;
	gap: 24px;
	background:
		radial-gradient(circle at top left, rgba(59, 130, 246, 0.08), transparent 26%),
		radial-gradient(circle at top right, rgba(16, 185, 129, 0.08), transparent 22%),
		#f8fafc;
	border-radius: 24px;
}

.section-card {
	padding: 24px;
	border-radius: 24px;
	border: 1px solid rgba(148, 163, 184, 0.18);
	background: rgba(255, 255, 255, 0.92);
	box-shadow: 0 20px 45px rgba(15, 23, 42, 0.08);
	backdrop-filter: blur(14px);
}

.section-heading {
	display: flex;
	align-items: flex-start;
	justify-content: space-between;
	gap: 16px;
	margin-bottom: 16px;
}

h3 {
	margin: 0;
	font-size: 24px;
	line-height: 1.2;
	color: #0f172a;
}

.builder-grid {
	display: grid;
	grid-template-columns: minmax(0, 1.65fr) minmax(280px, 0.95fr);
	gap: 20px;
	align-items: start;
}

.builder-main,
.service-sidebar {
	border: 1px solid #e2e8f0;
	border-radius: 20px;
	background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
}

.builder-main {
	padding: 20px;
}

.service-sidebar {
	padding: 20px;
	position: sticky;
	top: 20px;
}

.new-dag-font-style {
	display: flex;
	align-items: center;
	gap: 8px;
	font-size: 15px;
	font-weight: 700;
	color: #0f172a;
	margin-bottom: 12px;
}

.field-block {
	margin-bottom: 18px;
}

.builder-actions {
	display: flex;
	justify-content: flex-end;
	gap: 12px;
	flex-wrap: wrap;
	margin-bottom: 16px;
}

.builder-buttons {
	display: flex;
	flex-wrap: wrap;
	gap: 10px;
}

.tip-icon {
	color: #2563eb;
	font-size: 18px;
}

.canvas-placeholder {
	min-height: 500px;
	display: grid;
	place-items: center;
	text-align: center;
	border: 1.5px dashed #cbd5e1;
	border-radius: 22px;
	background:
		linear-gradient(135deg, rgba(37, 99, 235, 0.06), transparent 38%),
		linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
	padding: 32px;
}

.canvas-placeholder__icon {
	font-size: 34px;
	color: #2563eb;
	margin-bottom: 16px;
}

.canvas-placeholder__title {
	font-size: 18px;
	font-weight: 700;
	color: #0f172a;
}

.draw-container {
	min-height: 520px;
	display: flex;
	flex-direction: column;
	border-radius: 22px;
	border: 1px solid #cbd5e1;
	background: #ffffff;
	overflow: hidden;
	transition: border-color 0.2s ease, box-shadow 0.2s ease, transform 0.2s ease;
}

.draw-container.is-drag-over {
	border-color: #2563eb;
	box-shadow: 0 18px 40px rgba(37, 99, 235, 0.16);
	transform: translateY(-2px);
}

.canvas-status-bar {
	display: flex;
	align-items: center;
	justify-content: flex-end;
	gap: 8px;
	flex-wrap: wrap;
	padding: 10px 14px;
	border-bottom: 1px solid #e2e8f0;
	background: rgba(248, 250, 252, 0.92);
}

.canvas-metrics {
	display: flex;
	gap: 8px;
	flex-wrap: wrap;
}

.main-flow {
	position: relative;
	flex: 1;
	min-height: 430px;
	background:
		linear-gradient(180deg, rgba(248, 250, 252, 0.95), rgba(255, 255, 255, 0.98)),
		#ffffff;
}

.minimap-toggle-wrap {
	position: absolute;
	right: 16px;
	bottom: 76px;
	z-index: 12;
}

.minimap-toggle-wrap.expanded {
	bottom: 188px;
}

.minimap-toggle {
	border: 1px solid #cbd5e1;
	border-radius: 999px;
	background: rgba(255, 255, 255, 0.96);
	color: #334155;
	padding: 7px 12px;
	font-size: 12px;
	font-weight: 700;
	line-height: 1;
	cursor: pointer;
	box-shadow: 0 10px 22px rgba(15, 23, 42, 0.12);
	transition: border-color 0.2s ease, color 0.2s ease, transform 0.2s ease;
}

.minimap-toggle:hover {
	border-color: #93c5fd;
	color: #1d4ed8;
	transform: translateY(-1px);
}

.drag-tip {
	position: absolute;
	top: 20px;
	left: 50%;
	transform: translateX(-50%);
	z-index: 10;
	background: rgba(255, 255, 255, 0.92);
	padding: 8px 14px;
	border-radius: 999px;
	box-shadow: 0 10px 28px rgba(15, 23, 42, 0.12);
	display: flex;
	align-items: center;
	gap: 8px;
	border: 1px solid #e2e8f0;
	animation: float 3s ease-in-out infinite;
}

@keyframes float {
	0%,
	100% {
		transform: translateX(-50%) translateY(0);
	}

	50% {
		transform: translateX(-50%) translateY(-4px);
	}
}

.process-panel,
.layout-panel {
	display: flex;
	gap: 10px;
}

.process-panel {
	padding: 10px;
	border-radius: 14px;
	background: rgba(15, 23, 42, 0.82);
	box-shadow: 0 14px 34px rgba(15, 23, 42, 0.26);
}

.process-panel button {
	width: 42px;
	height: 42px;
	border: none;
	border-radius: 12px;
	background: rgba(255, 255, 255, 0.12);
	color: #ffffff;
	display: inline-flex;
	align-items: center;
	justify-content: center;
	cursor: pointer;
	font-size: 13px;
	font-weight: 700;
	transition: background-color 0.2s ease, transform 0.2s ease;
}

.process-panel button:hover {
	background: #2563eb;
	transform: translateY(-1px);
}

.sidebar-header {
	margin-bottom: 12px;
}

.service-grid {
	display: grid;
	gap: 12px;
	max-height: 640px;
	overflow: auto;
	padding-right: 4px;
}

.service-card {
	width: 100%;
	padding: 12px 12px 13px;
	text-align: left;
	border-radius: 16px;
	border: 1px solid #dbe4ee;
	border-left: 4px solid var(--service-accent);
	cursor: grab;
	transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
}

.service-card:hover {
	transform: translateY(-2px);
	box-shadow: 0 14px 30px rgba(15, 23, 42, 0.08);
	border-color: #94a3b8;
}

.service-card:active {
	cursor: grabbing;
}

.service-card__header {
	display: flex;
	align-items: center;
	justify-content: space-between;
	gap: 12px;
	margin-bottom: 10px;
}

.service-card__name {
	font-size: var(--dag-service-name-font-size);
	font-weight: 700;
	color: #0f172a;
}

.service-card__drag {
	font-size: 11px;
	font-weight: 700;
	letter-spacing: 0.08em;
	text-transform: uppercase;
	color: #2563eb;
}

.service-card__meta {
	display: flex;
	flex-wrap: wrap;
	gap: 8px;
}

.service-chip {
	display: inline-flex;
	align-items: center;
	padding: 4px 8px;
	border-radius: 999px;
	background: rgba(255, 255, 255, 0.9);
	border: 1px solid #dbe4ee;
	font-size: 11px;
	font-weight: 700;
	color: #475569;
}

.dag-list-section {
	overflow: hidden;
}

.dag-name-cell {
	display: grid;
	gap: 4px;
}

.dag-name {
	font-size: 15px;
	font-weight: 700;
	color: #0f172a;
}

.dag-subtitle {
	font-size: 13px;
	color: #64748b;
}

.dag-overview-button {
	width: 100%;
	padding: 14px 16px;
	text-align: left;
	border: 1px solid #e2e8f0;
	border-radius: 18px;
	background:
		linear-gradient(135deg, rgba(37, 99, 235, 0.05), transparent 32%),
		#f8fafc;
	cursor: pointer;
	transition: border-color 0.2s ease, box-shadow 0.2s ease, transform 0.2s ease;
}

.dag-overview-button:hover {
	border-color: #93c5fd;
	box-shadow: 0 14px 32px rgba(37, 99, 235, 0.1);
	transform: translateY(-1px);
}

.dag-overview-header {
	display: flex;
	align-items: center;
	gap: 12px;
	margin-bottom: 12px;
}

.dag-pill-group {
	display: flex;
	flex-wrap: wrap;
	gap: 8px;
}

.dag-pill-group--wrap {
	margin-top: 14px;
}

.service-pill {
	display: inline-flex;
	align-items: center;
	padding: 6px 10px;
	border-radius: 999px;
	background: #ffffff;
	border: 1px solid #dbe4ee;
	font-size: 12px;
	font-weight: 600;
	color: #334155;
}

.service-pill--muted {
	background: #eff6ff;
	border-color: #bfdbfe;
	color: #1d4ed8;
}

.stats {
	display: flex;
	flex-wrap: wrap;
	gap: 8px;
}

.dag-hover-panel {
	display: grid;
	gap: 14px;
}

.dag-hover-header {
	display: grid;
	gap: 4px;
}

.dag-title {
	font-size: 16px;
	font-weight: 700;
	color: #0f172a;
}

.dag-hover-subtitle {
	font-size: 13px;
	color: #64748b;
}

.dag-preview-shell {
	height: 320px;
	border-radius: 18px;
	border: 1px solid #e2e8f0;
	background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
	overflow: hidden;
}

.preview-flow {
	height: 100%;
	width: 100%;
}

.main-flow :deep(.vue-flow__controls),
.preview-flow :deep(.vue-flow__controls) {
	box-shadow: 0 12px 28px rgba(15, 23, 42, 0.12);
	border-radius: 14px;
	overflow: hidden;
}

.main-flow :deep(.vue-flow__minimap) {
	border-radius: 16px;
	overflow: hidden;
	box-shadow: 0 12px 28px rgba(15, 23, 42, 0.12);
	right: 16px;
	bottom: 116px;
}

.main-flow :deep(.dag-node),
.preview-flow :deep(.dag-node) {
	display: flex;
	align-items: center;
	justify-content: center;
	padding: 0 6px;
	font-size: 10px;
	font-weight: 700;
	line-height: 1.15;
	color: #0f172a;
	text-align: center;
	white-space: normal;
	overflow: hidden;
	overflow-wrap: anywhere;
	word-break: break-word;
	display: -webkit-box;
	-webkit-box-orient: vertical;
	-webkit-line-clamp: 2;
}

.main-flow :deep(.dag-node.selected) {
	box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.22), 0 18px 34px rgba(37, 99, 235, 0.14);
}

.preview-flow :deep(.dag-node) {
	cursor: default;
}

.preview-flow :deep(.vue-flow__edge-path),
.main-flow :deep(.vue-flow__edge-path) {
	stroke-linecap: round;
}

.preview-flow :deep(.vue-flow__attribution) {
	display: none;
}

:deep(.dag-preview-popper.el-popover) {
	padding: 16px;
	border-radius: 20px;
	border: 1px solid #e2e8f0;
	box-shadow: 0 22px 44px rgba(15, 23, 42, 0.14);
}

@media (max-width: 1200px) {
	.builder-grid {
		grid-template-columns: 1fr;
	}

	.service-sidebar {
		position: static;
	}
}

@media (max-width: 768px) {
	.dag-manage-page {
		padding: 14px;
		gap: 16px;
	}

	.section-card {
		padding: 18px;
		border-radius: 20px;
	}

	.section-heading,
	.builder-actions,
	.canvas-status-bar,
	.dag-overview-header {
		flex-direction: column;
		align-items: flex-start;
	}

	.canvas-placeholder,
	.draw-container {
		min-height: 440px;
	}

	.drag-tip {
		width: calc(100% - 24px);
		padding: 10px 14px;
		justify-content: center;
	}
}
</style>
