<template>
	<div class="outline dag-manage-page">
		<header class="page-hero">
			<div class="page-hero__copy">
				<div class="eyebrow">DAG Orchestration</div>
				<h2 class="page-hero__title">Compose workflows with the rhythm of a control room, not a form.</h2>
				<p class="page-hero__description">
					Keep service selection, graph editing and saved workflow inspection in one surface. The layout stays operational, but the interface now feels closer to a production orchestration console.
				</p>
			</div>
			<div class="page-hero__stats">
				<div class="hero-stat-card">
					<div class="hero-stat-card__label">Service Library</div>
					<div class="hero-stat-card__value">{{ services.length }}</div>
					<div class="hero-stat-card__meta">Available processing blocks</div>
				</div>
				<div class="hero-stat-card">
					<div class="hero-stat-card__label">Saved Dags</div>
					<div class="hero-stat-card__value">{{ dagList.length }}</div>
					<div class="hero-stat-card__meta">Reusable orchestration flows</div>
				</div>
			</div>
		</header>

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
						<div class="builder-subhead">
							<div class="builder-subhead__title">Canvas Composer</div>
							<div class="builder-subhead__copy">Snap-aligned nodes, directional edges and one-tap layout controls keep the orchestration readable as the graph grows.</div>
						</div>
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
						<div class="sidebar-note">
							<div class="sidebar-note__title">Library etiquette</div>
							<div class="sidebar-note__copy">Start with sources on the left, keep transformations in the middle, and finish with terminal services on the right.</div>
						</div>
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
									<span class="service-card__drag">Grab</span>
								</div>
								<div class="service-card__desc">{{ service.description }}</div>
								<div class="service-card__meta">
									<span class="service-meta-pill">
										<strong>IN</strong>
										{{ formatPortLabel(service.input) }}
									</span>
									<span class="service-meta-pill">
										<strong>OUT</strong>
										{{ formatPortLabel(service.output) }}
									</span>
								</div>
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
				<div class="heading-metrics">
					<el-tag effect="plain" type="info">{{ dagList.length }} dags</el-tag>
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

									<div class="overview-route">
										<div class="overview-route__row">
											<span class="overview-route__label">Entry</span>
											<div class="overview-route__chips">
												<span v-for="startNode in getStartNodes(scope.row.dag)" :key="startNode" class="route-chip route-chip--start">
													{{ startNode }}
												</span>
											</div>
										</div>
										<div class="overview-route__row">
											<span class="overview-route__label">Terminal</span>
											<div class="overview-route__chips">
												<span
													v-for="terminalNode in getTerminalNodes(scope.row.dag)"
													:key="terminalNode"
													class="route-chip route-chip--end"
												>
													{{ terminalNode }}
												</span>
											</div>
										</div>
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

								<div class="dag-structure-bar">
									<div class="dag-structure-bar__item">
										<span class="dag-structure-bar__label">Entry</span>
										<span class="dag-structure-bar__value">{{ getStartNodes(scope.row.dag).join(', ') || '-' }}</span>
									</div>
									<div class="dag-structure-bar__item">
										<span class="dag-structure-bar__label">Terminal</span>
										<span class="dag-structure-bar__value">{{ getTerminalNodes(scope.row.dag).join(', ') || '-' }}</span>
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
		getTerminalNodes(dag) {
			return Object.entries(dag || {})
				.filter(([key, node]) => key !== '_start' && Array.isArray(node?.succ) && node.succ.length === 0)
				.map(([key, node]) => node?.service_id || node?.id || key);
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
		formatPortLabel(value) {
			return value || 'Not specified';
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
	--dag-bg: #f3f6fb;
	--dag-panel: #ffffff;
	--dag-panel-soft: #f8fafc;
	--dag-panel-muted: #f1f5f9;
	--dag-border: #dbe4ee;
	--dag-border-strong: #c7d2e0;
	--dag-ink: #102033;
	--dag-ink-soft: #425466;
	--dag-ink-muted: #66778a;
	--dag-accent: #1f5eff;
	--dag-accent-soft: #eaf0ff;
	--dag-success-soft: #edfdf3;
	--dag-shadow: 0 18px 40px rgba(15, 23, 42, 0.06);
	--dag-shadow-strong: 0 22px 45px rgba(15, 23, 42, 0.1);
	padding: 20px;
	display: grid;
	gap: 24px;
	background:
		linear-gradient(180deg, rgba(31, 94, 255, 0.035), transparent 280px),
		linear-gradient(90deg, rgba(148, 163, 184, 0.08) 1px, transparent 1px),
		linear-gradient(rgba(148, 163, 184, 0.08) 1px, transparent 1px),
		var(--dag-bg);
	background-size: auto, 24px 24px, 24px 24px, auto;
	border-radius: 24px;
	font-family: 'Avenir Next', 'Segoe UI', 'PingFang SC', 'Hiragino Sans GB', sans-serif;
}

.page-hero {
	display: grid;
	grid-template-columns: minmax(0, 1.6fr) minmax(280px, 0.95fr);
	gap: 18px;
	padding: 24px 26px;
	border: 1px solid rgba(199, 210, 224, 0.85);
	border-radius: 24px;
	background:
		linear-gradient(135deg, rgba(31, 94, 255, 0.05), transparent 34%),
		linear-gradient(180deg, rgba(255, 255, 255, 0.94), rgba(249, 251, 255, 0.96));
	box-shadow: var(--dag-shadow);
}

.page-hero__copy {
	max-width: 780px;
}

.page-hero__title {
	margin: 0;
	max-width: 720px;
	font-size: 34px;
	line-height: 1.14;
	letter-spacing: -0.03em;
	color: var(--dag-ink);
}

.page-hero__description {
	margin: 14px 0 0;
	max-width: 680px;
	font-size: 15px;
	line-height: 1.7;
	color: var(--dag-ink-soft);
}

.page-hero__stats {
	display: grid;
	grid-template-columns: repeat(2, minmax(0, 1fr));
	gap: 12px;
	align-self: stretch;
}

.hero-stat-card {
	display: grid;
	align-content: start;
	gap: 10px;
	padding: 18px;
	border: 1px solid var(--dag-border);
	border-radius: 18px;
	background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
}

.hero-stat-card__label {
	font-size: 11px;
	font-weight: 700;
	letter-spacing: 0.1em;
	text-transform: uppercase;
	color: var(--dag-ink-muted);
}

.hero-stat-card__value {
	font-size: 32px;
	font-weight: 750;
	line-height: 1;
	letter-spacing: -0.04em;
	color: var(--dag-ink);
}

.hero-stat-card__meta {
	font-size: 13px;
	line-height: 1.5;
	color: var(--dag-ink-soft);
}

.section-card {
	padding: 24px;
	border-radius: 24px;
	border: 1px solid rgba(199, 210, 224, 0.92);
	background: rgba(255, 255, 255, 0.96);
	box-shadow: var(--dag-shadow);
}

.section-heading {
	display: flex;
	align-items: flex-start;
	justify-content: space-between;
	gap: 16px;
	margin-bottom: 20px;
	padding-bottom: 16px;
	border-bottom: 1px solid rgba(219, 228, 238, 0.85);
}

.eyebrow {
	font-size: 12px;
	font-weight: 700;
	letter-spacing: 0.08em;
	text-transform: uppercase;
	color: var(--dag-accent);
	margin-bottom: 8px;
}

h3 {
	margin: 0;
	font-size: 26px;
	line-height: 1.2;
	letter-spacing: -0.02em;
	color: var(--dag-ink);
}

.section-description {
	margin: 10px 0 0;
	max-width: 720px;
	font-size: 14px;
	line-height: 1.6;
	color: var(--dag-ink-soft);
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
	border: 1px solid var(--dag-border);
	border-radius: 20px;
	background: linear-gradient(180deg, #ffffff 0%, #f9fbfe 100%);
}

.builder-main {
	padding: 20px;
	box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.8);
}

.service-sidebar {
	padding: 20px;
	position: sticky;
	top: 20px;
	background:
		linear-gradient(180deg, rgba(248, 251, 255, 0.9), rgba(255, 255, 255, 0.98)),
		#ffffff;
}

.new-dag-font-style {
	display: flex;
	align-items: center;
	gap: 8px;
	font-size: 15px;
	font-weight: 700;
	color: var(--dag-ink);
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
	margin-bottom: 16px;
	padding: 14px 16px;
	border-radius: 18px;
	background:
		linear-gradient(135deg, rgba(31, 94, 255, 0.05), transparent 38%),
		var(--dag-panel-soft);
	border: 1px solid var(--dag-border);
}

.builder-hint {
	display: flex;
	align-items: center;
	gap: 10px;
	font-size: 13px;
	font-weight: 500;
	color: var(--dag-ink-soft);
}

.builder-buttons {
	display: flex;
	flex-wrap: wrap;
	gap: 10px;
}

:deep(.el-input__wrapper) {
	min-height: 46px;
	border-radius: 14px;
	box-shadow: 0 0 0 1px rgba(211, 219, 229, 0.95) inset;
	background: rgba(255, 255, 255, 0.96);
}

:deep(.el-input__wrapper.is-focus) {
	box-shadow: 0 0 0 1px rgba(31, 94, 255, 0.7) inset;
}

:deep(.el-input__inner) {
	color: var(--dag-ink);
	font-size: 14px;
}

:deep(.builder-buttons .el-button),
:deep(.section-heading .el-tag),
:deep(.canvas-metrics .el-tag),
:deep(.stats .el-tag) {
	border-radius: 999px;
}

:deep(.builder-buttons .el-button) {
	padding-inline: 18px;
	font-weight: 600;
}

.tip-icon {
	color: var(--dag-accent);
	font-size: 18px;
}

.canvas-placeholder {
	min-height: 520px;
	display: grid;
	place-items: center;
	text-align: center;
	border: 1.5px dashed var(--dag-border-strong);
	border-radius: 22px;
	background:
		linear-gradient(135deg, rgba(31, 94, 255, 0.07), transparent 38%),
		linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
	padding: 32px;
}

.canvas-placeholder__icon {
	font-size: 34px;
	color: var(--dag-accent);
	margin-bottom: 16px;
}

.canvas-placeholder__title {
	font-size: 20px;
	font-weight: 700;
	letter-spacing: -0.02em;
	color: var(--dag-ink);
	margin-bottom: 8px;
}

.canvas-placeholder__copy {
	max-width: 420px;
	font-size: 14px;
	line-height: 1.6;
	color: var(--dag-ink-soft);
}

.draw-container {
	min-height: 560px;
	display: flex;
	flex-direction: column;
	border-radius: 22px;
	border: 1px solid var(--dag-border);
	background: var(--dag-panel);
	overflow: hidden;
	transition: border-color 0.2s ease, box-shadow 0.2s ease, transform 0.2s ease;
}

.draw-container.is-drag-over {
	border-color: var(--dag-accent);
	box-shadow: 0 18px 40px rgba(31, 94, 255, 0.14);
	transform: translateY(-2px);
}

.builder-subhead {
	display: flex;
	align-items: baseline;
	justify-content: space-between;
	gap: 12px;
	padding: 16px 18px 10px;
	border-bottom: 1px solid rgba(219, 228, 238, 0.7);
	background: linear-gradient(180deg, rgba(248, 250, 252, 0.88), rgba(248, 250, 252, 0.4));
}

.builder-subhead__title {
	font-size: 14px;
	font-weight: 700;
	letter-spacing: 0.04em;
	text-transform: uppercase;
	color: var(--dag-ink);
}

.builder-subhead__copy {
	max-width: 520px;
	font-size: 12px;
	line-height: 1.6;
	color: var(--dag-ink-muted);
	text-align: right;
}

.canvas-status-bar {
	display: flex;
	align-items: center;
	justify-content: space-between;
	gap: 12px;
	flex-wrap: wrap;
	padding: 14px 18px;
	border-bottom: 1px solid rgba(219, 228, 238, 0.82);
	background: rgba(248, 250, 252, 0.86);
}

.canvas-metrics {
	display: flex;
	gap: 8px;
	flex-wrap: wrap;
}

.canvas-status-copy {
	font-size: 13px;
	color: var(--dag-ink-muted);
}

.main-flow {
	position: relative;
	flex: 1;
	min-height: 500px;
	background:
		linear-gradient(180deg, rgba(248, 250, 252, 0.9), rgba(255, 255, 255, 0.98)),
		linear-gradient(90deg, rgba(219, 228, 238, 0.55) 1px, transparent 1px),
		linear-gradient(rgba(219, 228, 238, 0.55) 1px, transparent 1px),
		#ffffff;
	background-size: auto, 24px 24px, 24px 24px, auto;
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
	box-shadow: 0 10px 28px rgba(15, 23, 42, 0.1);
	display: flex;
	align-items: center;
	gap: 8px;
	border: 1px solid var(--dag-border);
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
	border-radius: 16px;
	background: rgba(255, 255, 255, 0.94);
	border: 1px solid rgba(203, 213, 225, 0.92);
	box-shadow: 0 16px 36px rgba(15, 23, 42, 0.1);
}

.process-panel button {
	width: 42px;
	height: 42px;
	border: none;
	border-radius: 12px;
	background: var(--dag-panel-soft);
	color: var(--dag-ink);
	display: inline-flex;
	align-items: center;
	justify-content: center;
	cursor: pointer;
	font-size: 13px;
	font-weight: 700;
	transition: background-color 0.2s ease, transform 0.2s ease;
}

.process-panel button:hover {
	background: var(--dag-accent-soft);
	transform: translateY(-1px);
}

.sidebar-header {
	margin-bottom: 16px;
}

.sidebar-copy {
	margin: 0;
	font-size: 13px;
	line-height: 1.6;
	color: var(--dag-ink-muted);
}

.sidebar-note {
	margin-top: 14px;
	padding: 14px 15px;
	border: 1px solid rgba(191, 219, 254, 0.8);
	border-radius: 16px;
	background: linear-gradient(135deg, rgba(239, 246, 255, 0.9), rgba(248, 250, 252, 0.9));
}

.sidebar-note__title {
	font-size: 12px;
	font-weight: 700;
	letter-spacing: 0.08em;
	text-transform: uppercase;
	color: var(--dag-accent);
	margin-bottom: 6px;
}

.sidebar-note__copy {
	font-size: 13px;
	line-height: 1.6;
	color: var(--dag-ink-soft);
}

.service-grid {
	display: grid;
	gap: 12px;
	max-height: 640px;
	overflow: auto;
	padding-right: 6px;
}

.service-card {
	width: 100%;
	padding: 14px 14px 14px;
	text-align: left;
	border-radius: 16px;
	border: 1px solid var(--dag-border-strong);
	cursor: grab;
	transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
	box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.7);
}

.service-card:hover {
	transform: translateY(-2px);
	box-shadow: 0 14px 28px rgba(15, 23, 42, 0.08);
	border-color: #8ea4bf;
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
	color: var(--dag-ink);
}

.service-card__drag {
	font-size: 11px;
	font-weight: 700;
	letter-spacing: 0.08em;
	text-transform: uppercase;
	color: var(--dag-accent);
}

.service-card__desc,
.description {
	font-size: 13px;
	line-height: 1.55;
	color: var(--dag-ink-soft);
	white-space: pre-wrap;
	word-break: break-word;
}

.service-card__meta {
	display: flex;
	flex-wrap: wrap;
	gap: 8px;
	margin-top: 12px;
}

.service-meta-pill {
	display: inline-flex;
	align-items: center;
	gap: 6px;
	padding: 5px 8px;
	border-radius: 999px;
	border: 1px solid rgba(199, 210, 224, 0.95);
	background: rgba(255, 255, 255, 0.78);
	font-size: 11px;
	color: var(--dag-ink-soft);
}

.service-meta-pill strong {
	font-size: 10px;
	letter-spacing: 0.08em;
	color: var(--dag-accent);
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
	color: var(--dag-ink);
}

.dag-subtitle {
	font-size: 13px;
	color: var(--dag-ink-muted);
}

.dag-overview-button {
	width: 100%;
	padding: 16px 16px 15px;
	text-align: left;
	border: 1px solid var(--dag-border);
	border-radius: 18px;
	background:
		linear-gradient(135deg, rgba(31, 94, 255, 0.05), transparent 32%),
		linear-gradient(180deg, #fbfdff 0%, #f6f9fc 100%);
	cursor: pointer;
	transition: border-color 0.2s ease, box-shadow 0.2s ease, transform 0.2s ease;
}

.dag-overview-button:hover {
	border-color: #a4b8cf;
	box-shadow: 0 14px 28px rgba(15, 23, 42, 0.08);
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
	color: var(--dag-accent);
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
	border: 1px solid var(--dag-border);
	font-size: 12px;
	font-weight: 600;
	color: #334155;
}

.service-pill--muted {
	background: var(--dag-accent-soft);
	border-color: #bfdbfe;
	color: #1d4ed8;
}

.stats {
	display: flex;
	flex-wrap: wrap;
	gap: 8px;
}

.overview-route {
	display: grid;
	gap: 8px;
	margin-top: 14px;
	padding-top: 12px;
	border-top: 1px solid rgba(219, 228, 238, 0.9);
}

.overview-route__row {
	display: flex;
	align-items: flex-start;
	gap: 10px;
}

.overview-route__label {
	flex: 0 0 56px;
	font-size: 11px;
	font-weight: 700;
	letter-spacing: 0.08em;
	text-transform: uppercase;
	color: var(--dag-ink-muted);
	padding-top: 4px;
}

.overview-route__chips {
	display: flex;
	flex-wrap: wrap;
	gap: 8px;
}

.route-chip {
	display: inline-flex;
	align-items: center;
	max-width: 180px;
	padding: 5px 8px;
	border-radius: 999px;
	font-size: 11px;
	font-weight: 600;
	overflow: hidden;
	text-overflow: ellipsis;
	white-space: nowrap;
}

.route-chip--start {
	background: var(--dag-accent-soft);
	color: #1e40af;
}

.route-chip--end {
	background: var(--dag-success-soft);
	color: #166534;
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
	color: var(--dag-ink);
}

.dag-hover-subtitle {
	font-size: 13px;
	color: var(--dag-ink-muted);
}

.dag-structure-bar {
	display: grid;
	grid-template-columns: repeat(2, minmax(0, 1fr));
	gap: 10px;
}

.dag-structure-bar__item {
	display: grid;
	gap: 6px;
	padding: 10px 12px;
	border: 1px solid rgba(219, 228, 238, 0.92);
	border-radius: 14px;
	background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
}

.dag-structure-bar__label {
	font-size: 11px;
	font-weight: 700;
	letter-spacing: 0.08em;
	text-transform: uppercase;
	color: var(--dag-ink-muted);
}

.dag-structure-bar__value {
	font-size: 13px;
	font-weight: 600;
	line-height: 1.5;
	color: var(--dag-ink-soft);
}

.dag-preview-shell {
	height: 320px;
	border-radius: 18px;
	border: 1px solid var(--dag-border);
	background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
	overflow: hidden;
}

.preview-flow {
	height: 100%;
	width: 100%;
}

.main-flow :deep(.vue-flow__controls),
.preview-flow :deep(.vue-flow__controls) {
	box-shadow: 0 12px 28px rgba(15, 23, 42, 0.1);
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
	color: var(--dag-ink);
	text-align: center;
}

.main-flow :deep(.dag-node.selected) {
	box-shadow: 0 0 0 3px rgba(31, 94, 255, 0.18), 0 18px 34px rgba(31, 94, 255, 0.12);
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
	border: 1px solid var(--dag-border);
	box-shadow: var(--dag-shadow-strong);
}

:deep(.el-table) {
	--el-table-border-color: transparent;
	--el-table-header-bg-color: transparent;
	--el-table-row-hover-bg-color: transparent;
	background: transparent;
}

:deep(.el-table th.el-table__cell) {
	padding: 10px 0 14px;
	background: transparent;
	font-size: 12px;
	font-weight: 700;
	letter-spacing: 0.08em;
	text-transform: uppercase;
	color: var(--dag-ink-muted);
	border-bottom: 1px solid rgba(219, 228, 238, 0.95);
}

:deep(.el-table td.el-table__cell) {
	padding: 18px 0;
	border-bottom: 1px solid rgba(238, 242, 247, 0.98);
	background: transparent;
}

:deep(.el-table__inner-wrapper::before) {
	display: none;
}

@media (max-width: 1200px) {
	.page-hero {
		grid-template-columns: 1fr;
	}

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

	.page-hero {
		padding: 18px;
	}

	.page-hero__title {
		font-size: 28px;
	}

	.page-hero__stats {
		grid-template-columns: 1fr;
	}

	.section-card {
		padding: 18px;
		border-radius: 20px;
	}

	.section-heading,
	.builder-actions,
	.builder-subhead,
	.canvas-status-bar,
	.dag-overview-header {
		flex-direction: column;
		align-items: flex-start;
	}

	.builder-subhead__copy {
		text-align: left;
	}

	.dag-structure-bar {
		grid-template-columns: 1fr;
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
