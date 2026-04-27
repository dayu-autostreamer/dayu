<template>
	<div class="outline dag-manage-page">
		<section class="section-card composer-section">
			<div class="section-heading">
				<div>
					<div class="eyebrow">Workflow Builder</div>
					<h3>Add Application Dag</h3>
					<p class="section-description">
						Drag services onto the canvas, connect them from left to right, then save the DAG once the flow looks right.
					</p>
				</div>
				<div class="heading-metrics">
					<el-tag effect="plain" type="info">{{ services.length }} services</el-tag>
					<el-tag effect="plain" type="success">{{ flowNodes.length }} staged nodes</el-tag>
				</div>
			</div>

			<div class="builder-grid">
				<div class="builder-main">
					<div class="field-block">
						<div class="new-dag-font-style">Dag Name</div>
						<el-input v-model="newInputName" clearable placeholder="Please fill the dag name" />
					</div>

					<div class="builder-actions">
						<div class="builder-hint">
							<el-icon class="tip-icon">
								<MagicStick />
							</el-icon>
							<span>1. Drag services 2. Connect nodes 3. Auto layout 4. Save the DAG</span>
						</div>
						<div class="builder-buttons">
							<el-button type="warning" plain @click="draw">{{ drawing ? 'Hide Canvas' : 'Open Canvas' }}</el-button>
							<el-button plain :disabled="!drawing || !flowNodes.length" @click="focusCanvas">Fit View</el-button>
							<el-button type="primary" round :disabled="!flowNodes.length" @click="handleNewSubmit">Save Dag</el-button>
							<el-button round :disabled="!newInputName && !flowNodes.length" @click="clearInput">Reset</el-button>
						</div>
					</div>

					<div v-if="!drawing" class="canvas-placeholder">
						<el-icon class="canvas-placeholder__icon">
							<MagicStick />
						</el-icon>
						<div class="canvas-placeholder__title">Open the canvas to start composing</div>
						<div class="canvas-placeholder__copy">
							The canvas snaps nodes to a grid and supports one-click auto layout for cleaner workflows.
						</div>
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
							<div class="canvas-status-copy">Drop services here and keep arrows flowing left to right for the clearest DAG.</div>
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
								<span>Drag service nodes to build your workflow</span>
							</div>

							<Background pattern-color="#cbd5e1" :gap="20" />
							<MiniMap />
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
						<div class="new-dag-font-style">
							Service Containers
							<el-tooltip placement="right">
								<template #content>From docker Registry: https://hub.docker.com/u/dayuhub</template>
								<el-button size="small" circle>i</el-button>
							</el-tooltip>
						</div>
						<p class="sidebar-copy">Each service can be used once in a DAG. Hover a card to inspect its inputs and outputs.</p>
					</div>

					<div class="service-grid">
						<el-tooltip
							v-for="service in services"
							:key="service.id"
							placement="left"
							:open-delay="350"
							enterable
						>
							<template #content>
								<div class="description">{{ service.description }}</div>
							</template>
							<button
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
								<div class="service-card__desc">{{ service.description }}</div>
							</button>
						</el-tooltip>
					</div>
				</aside>
			</div>
		</section>

		<section class="section-card dag-list-section">
			<div class="section-heading">
				<div>
					<div class="eyebrow">Saved Workflows</div>
					<h3>Current Application Dags</h3>
					<p class="section-description">
						Hover a DAG summary to inspect its structure in place without leaving the list.
					</p>
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
										<span class="hover-hint">Hover to inspect</span>
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
import { ElButton, ElInput, ElMessage, ElPopover, ElTable, ElTableColumn, ElTag, ElTooltip } from 'element-plus';
import { nextTick, ref } from 'vue';
import { MarkerType, Panel, useVueFlow, VueFlow } from '@vue-flow/core';
import { Controls } from '@vue-flow/controls';
import { Background } from '@vue-flow/background';
import { MiniMap } from '@vue-flow/minimap';
import useDragAndDrop from './useDnD';
import Icon from './Icon.vue';
import { useLayout } from './useLayout';
import { Connection, Link, MagicStick } from '@element-plus/icons-vue';

const NODE_TONES = [
	{ background: '#eff6ff', border: '#93c5fd' },
	{ background: '#ecfeff', border: '#67e8f9' },
	{ background: '#f0fdf4', border: '#86efac' },
	{ background: '#fff7ed', border: '#fdba74' },
	{ background: '#fdf2f8', border: '#f9a8d4' },
	{ background: '#eef2ff', border: '#a5b4fc' },
	{ background: '#fefce8', border: '#fde047' },
	{ background: '#f8fafc', border: '#cbd5e1' },
];
const MAIN_FLOW_ID = 'dag-builder-main';
const PREVIEW_NODE_LIMIT = 4;

export default {
	name: 'DagManage',
	components: {
		ElTable,
		ElTableColumn,
		ElTooltip,
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
			const source = String(key || '');
			let hash = 0;
			for (let i = 0; i < source.length; i += 1) {
				hash = (hash << 5) - hash + source.charCodeAt(i);
				hash |= 0;
			}

			return NODE_TONES[Math.abs(hash) % NODE_TONES.length];
		},
		getNodeStyle(key) {
			const tone = this.getNodeTone(key);
			return {
				backgroundColor: tone.background,
				border: `1px solid ${tone.border}`,
				borderRadius: '16px',
				boxShadow: '0 12px 28px rgba(15, 23, 42, 0.08)',
				color: '#0f172a',
			};
		},
		getServiceCardStyle(service) {
			const tone = this.getNodeTone(service?.id || service?.name);
			return {
				backgroundColor: tone.background,
				borderColor: tone.border,
			};
		},
		formatServiceName(service) {
			return service?.name || service?.id || 'Unknown Service';
		},
		parseDag(dag) {
			return Object.keys(dag || {})
				.filter((key) => key !== '_start')
				.map((key) => ({
					id: key,
					class: 'dag-node',
					data: {
						label: dag[key]?.service_id || dag[key]?.id || key,
						service_id: dag[key]?.service_id || key,
					},
					dimensions: { width: 180, height: 56 },
					style: this.getNodeStyle(key),
				}));
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
	margin-bottom: 20px;
}

.eyebrow {
	font-size: 12px;
	font-weight: 700;
	letter-spacing: 0.08em;
	text-transform: uppercase;
	color: #2563eb;
	margin-bottom: 8px;
}

h3 {
	margin: 0;
	font-size: 26px;
	line-height: 1.2;
	color: #0f172a;
}

.section-description {
	margin: 10px 0 0;
	max-width: 720px;
	font-size: 14px;
	line-height: 1.6;
	color: #475569;
}

.heading-metrics {
	display: flex;
	flex-wrap: wrap;
	gap: 8px;
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
	align-items: center;
	justify-content: space-between;
	gap: 16px;
	flex-wrap: wrap;
	margin-bottom: 18px;
	padding: 14px 16px;
	border-radius: 18px;
	background: #f8fafc;
	border: 1px solid #e2e8f0;
}

.builder-hint {
	display: flex;
	align-items: center;
	gap: 10px;
	font-size: 13px;
	font-weight: 500;
	color: #475569;
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
	min-height: 520px;
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
	font-size: 20px;
	font-weight: 700;
	color: #0f172a;
	margin-bottom: 8px;
}

.canvas-placeholder__copy {
	max-width: 420px;
	font-size: 14px;
	line-height: 1.6;
	color: #475569;
}

.draw-container {
	min-height: 560px;
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
	justify-content: space-between;
	gap: 12px;
	flex-wrap: wrap;
	padding: 14px 18px;
	border-bottom: 1px solid #e2e8f0;
	background: rgba(248, 250, 252, 0.92);
}

.canvas-metrics {
	display: flex;
	gap: 8px;
	flex-wrap: wrap;
}

.canvas-status-copy {
	font-size: 13px;
	color: #64748b;
}

.main-flow {
	position: relative;
	flex: 1;
	min-height: 500px;
	background:
		linear-gradient(180deg, rgba(248, 250, 252, 0.95), rgba(255, 255, 255, 0.98)),
		#ffffff;
}

.drag-tip {
	position: absolute;
	top: 20px;
	left: 50%;
	transform: translateX(-50%);
	z-index: 10;
	background: rgba(255, 255, 255, 0.92);
	padding: 10px 18px;
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
	margin-bottom: 16px;
}

.sidebar-copy {
	margin: 0;
	font-size: 13px;
	line-height: 1.6;
	color: #64748b;
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
	padding: 14px 14px 16px;
	text-align: left;
	border-radius: 18px;
	border: 1px solid #cbd5e1;
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
	font-size: 14px;
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

.service-card__desc,
.description {
	font-size: 13px;
	line-height: 1.55;
	color: #475569;
	white-space: pre-wrap;
	word-break: break-word;
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
	justify-content: space-between;
	gap: 16px;
	margin-bottom: 12px;
}

.hover-hint {
	flex-shrink: 0;
	font-size: 12px;
	font-weight: 600;
	color: #2563eb;
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
}

.main-flow :deep(.dag-node),
.preview-flow :deep(.dag-node) {
	display: flex;
	align-items: center;
	justify-content: center;
	padding: 0 14px;
	font-size: 13px;
	font-weight: 700;
	line-height: 1.4;
	color: #0f172a;
	text-align: center;
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
