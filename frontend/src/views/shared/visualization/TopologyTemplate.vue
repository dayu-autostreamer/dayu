<template>
	<div class="topology-surface" @mouseleave="hideTooltip">
		<svg
			v-if="layoutGraph"
			class="topology-svg"
			:viewBox="`0 0 ${layoutGraph.width} ${layoutGraph.height}`"
			preserveAspectRatio="xMidYMid meet"
			role="img"
			:aria-label="config.name || 'Topology visualization'"
		>
			<defs>
				<marker
					:id="layoutGraph.markerId"
					markerWidth="9"
					markerHeight="9"
					refX="8"
					refY="4.5"
					orient="auto"
					markerUnits="strokeWidth"
				>
					<path d="M 0 0 L 9 4.5 L 0 9 z" fill="#64748b" />
				</marker>
			</defs>

			<path
				v-for="edge in layoutGraph.edges"
				:key="edge.id"
				class="topology-edge"
				:d="edge.path"
				:marker-end="`url(#${layoutGraph.markerId})`"
			/>

			<g
				v-for="node in layoutGraph.nodes"
				:key="node.id"
				class="topology-node"
				:transform="`translate(${node.x}, ${node.y})`"
				@mouseenter="showTooltip(node, $event)"
				@mousemove="moveTooltip($event)"
				@mouseleave="hideTooltip"
			>
				<rect
					:width="node.width"
					:height="node.height"
					rx="7"
					:fill="node.backgroundColor"
					stroke="rgba(15, 23, 42, 0.16)"
					stroke-width="1"
				/>
				<text class="topology-node__text" text-anchor="middle" dominant-baseline="middle">
					<tspan
						v-for="(line, index) in node.lines"
						:key="`${node.id}-${index}`"
						:class="index === 0 ? 'topology-node__title' : 'topology-node__content'"
						:x="node.width / 2"
						:y="node.textStartY + index * node.lineHeight"
						:fill="node.foregroundColor"
					>
						{{ line }}
					</tspan>
				</text>
			</g>
		</svg>

		<div v-else class="topology-empty">
			<el-icon :size="36">
				<PieChart />
			</el-icon>
			<p>No topology data available</p>
		</div>

		<div v-if="tooltip.visible" class="topology-tooltip" :style="tooltipStyle">
			<div class="topology-tooltip__title">{{ tooltip.node.name }}</div>
			<div class="topology-tooltip__row">
				<span>Data:</span>
				<strong :style="{ color: tooltip.node.backgroundColor }">{{ tooltip.node.data }}</strong>
			</div>
		</div>
	</div>
</template>

<script>
import { computed, reactive } from 'vue';
import dagre from '@dagrejs/dagre';
import { PieChart } from '@element-plus/icons-vue';

const GRAPH_PADDING = 10;
const NODE_MIN_WIDTH = 68;
const NODE_MAX_WIDTH = 132;
const NODE_HORIZONTAL_PADDING = 10;
const NODE_VERTICAL_PADDING = 6;
const NODE_LINE_HEIGHT = 10;
const COLOR_PALETTE = ['#2563eb', '#10b981', '#f59e0b', '#ef4444', '#0ea5e9', '#8b5cf6', '#14b8a6', '#f97316'];

function hashString(value) {
	const source = String(value || '');
	let hash = 0;

	for (let index = 0; index < source.length; index += 1) {
		hash = source.charCodeAt(index) + ((hash << 5) - hash);
		hash |= 0;
	}

	return Math.abs(hash);
}

function chunkWord(word, maxLength) {
	if (word.length <= maxLength) {
		return [word];
	}

	const chunks = [];
	for (let index = 0; index < word.length; index += maxLength) {
		chunks.push(word.slice(index, index + maxLength));
	}
	return chunks;
}

function wrapLabel(value, maxLineLength = 18) {
	const normalized = String(value ?? '')
		.replace(/[_-]+/g, ' ')
		.replace(/\s+/g, ' ')
		.trim();

	if (!normalized) {
		return ['N/A'];
	}

	const tokens = normalized.split(' ').flatMap((token) => chunkWord(token, maxLineLength));
	const lines = [];
	let currentLine = '';

	tokens.forEach((token) => {
		const candidate = currentLine ? `${currentLine} ${token}` : token;
		if (candidate.length <= maxLineLength || !currentLine) {
			currentLine = candidate;
			return;
		}
		lines.push(currentLine);
		currentLine = token;
	});

	if (currentLine) {
		lines.push(currentLine);
	}

	return lines;
}

function getContrastColor(hex, opacity = 1) {
	const r = parseInt(hex.slice(1, 3), 16);
	const g = parseInt(hex.slice(3, 5), 16);
	const b = parseInt(hex.slice(5, 7), 16);
	const brightness = (r * 299 + g * 587 + b * 114) / 1000;
	return brightness > 150 ? `rgba(15, 23, 42, ${opacity})` : `rgba(248, 250, 252, ${opacity})`;
}

function getNodeColor(value) {
	const index = hashString(value) % COLOR_PALETTE.length;
	return COLOR_PALETTE[index];
}

function getFirstNonEmptyValue(obj, variables) {
	if (!obj || typeof obj !== 'object' || !Array.isArray(variables)) return null;
	for (const key of variables) {
		const value = obj[key];
		if (value !== null && value !== undefined && value !== '') {
			return value;
		}
	}
	return null;
}

function getNodeMetrics(id, nodeInfo) {
	const serviceName = nodeInfo?.service?.service_name || id;
	const data = nodeInfo?.service?.data ?? 'No data';
	const titleLines = wrapLabel(serviceName, 18);
	const dataLines = wrapLabel(data, 18);
	const lines = [...titleLines, ...dataLines];
	const longestLine = Math.max(...lines.map((line) => line.length), 8);
	const width = Math.min(NODE_MAX_WIDTH, Math.max(NODE_MIN_WIDTH, longestLine * 5.7 + NODE_HORIZONTAL_PADDING * 2));
	const height = NODE_VERTICAL_PADDING * 2 + lines.length * NODE_LINE_HEIGHT;
	const backgroundColor = getNodeColor(String(data));
	const textBlockHeight = (lines.length - 1) * NODE_LINE_HEIGHT;

	return {
		id,
		name: serviceName,
		data,
		lines,
		width,
		height,
		lineHeight: NODE_LINE_HEIGHT,
		textStartY: height / 2 - textBlockHeight / 2,
		backgroundColor,
		foregroundColor: getContrastColor(backgroundColor),
	};
}

function buildEdgePath(sourceNode, targetNode) {
	const startX = sourceNode.x + sourceNode.width;
	const startY = sourceNode.y + sourceNode.height / 2;
	const endX = targetNode.x;
	const endY = targetNode.y + targetNode.height / 2;
	const midX = startX + Math.max(6, (endX - startX) / 2);

	return `M ${startX} ${startY} L ${midX} ${startY} L ${midX} ${endY} L ${endX} ${endY}`;
}

function buildLayout(config, data) {
	const variables = config?.variables;
	if (!Array.isArray(variables)) return null;

	const latestData = [...(data || [])]
		.reverse()
		.map((item) => getFirstNonEmptyValue(item, variables))
		.find((value) => value !== null);

	if (!latestData || typeof latestData !== 'object') return null;

	const entries = Object.entries(latestData);
	if (!entries.length) return null;

	const graph = new dagre.graphlib.Graph();
	graph.setGraph({
		rankdir: 'LR',
		nodesep: 4,
		ranksep: 6,
		marginx: 0,
		marginy: 0,
	});
	graph.setDefaultEdgeLabel(() => ({}));

	const nodes = entries.map(([rawId, nodeInfo]) => {
		const id = String(rawId);
		const node = getNodeMetrics(id, nodeInfo);
		graph.setNode(id, {
			width: node.width,
			height: node.height,
		});
		return node;
	});

	const nodeIds = new Set(nodes.map((node) => node.id));
	const edges = [];

	entries.forEach(([rawSourceId, nodeInfo]) => {
		const source = String(rawSourceId);
		(nodeInfo?.next_nodes || []).forEach((rawTargetId) => {
			const target = String(rawTargetId);
			if (!nodeIds.has(target)) return;
			graph.setEdge(source, target);
			edges.push({
				id: `${source}-${target}`,
				source,
				target,
			});
		});
	});

	dagre.layout(graph);

	const positionedNodes = nodes.map((node) => {
		const position = graph.node(node.id) || { x: node.width / 2, y: node.height / 2 };
		return {
			...node,
			x: position.x - node.width / 2,
			y: position.y - node.height / 2,
		};
	});

	const bounds = positionedNodes.reduce(
		(acc, node) => ({
			minX: Math.min(acc.minX, node.x),
			minY: Math.min(acc.minY, node.y),
			maxX: Math.max(acc.maxX, node.x + node.width),
			maxY: Math.max(acc.maxY, node.y + node.height),
		}),
		{
			minX: Infinity,
			minY: Infinity,
			maxX: -Infinity,
			maxY: -Infinity,
		}
	);

	const normalizedNodes = positionedNodes.map((node) => ({
		...node,
		x: node.x - bounds.minX + GRAPH_PADDING,
		y: node.y - bounds.minY + GRAPH_PADDING,
	}));
	const nodeMap = new Map(normalizedNodes.map((node) => [node.id, node]));
	const normalizedEdges = edges
		.map((edge) => {
			const sourceNode = nodeMap.get(edge.source);
			const targetNode = nodeMap.get(edge.target);
			if (!sourceNode || !targetNode) return null;
			return {
				...edge,
				path: buildEdgePath(sourceNode, targetNode),
			};
		})
		.filter(Boolean);

	return {
		nodes: normalizedNodes,
		edges: normalizedEdges,
		width: Math.max(bounds.maxX - bounds.minX + GRAPH_PADDING * 2, 180),
		height: Math.max(bounds.maxY - bounds.minY + GRAPH_PADDING * 2, 120),
		markerId: `topology-arrow-${hashString(
			`${config?.id || config?.name || 'topology'}-${entries.map(([id]) => id).join('-')}`
		)}`,
	};
}

export default {
	name: 'TopologyTemplate',
	components: { PieChart },
	props: {
		config: {
			type: Object,
			required: true,
			default: () => ({
				id: '',
				name: '',
				type: 'topology',
				variables: [],
			}),
		},
		data: {
			type: Array,
			required: true,
			default: () => [],
		},
	},
	setup(props) {
		const tooltip = reactive({
			visible: false,
			x: 0,
			y: 0,
			node: {},
		});

		const layoutGraph = computed(() => buildLayout(props.config, props.data));
		const tooltipStyle = computed(() => ({
			left: `${tooltip.x + 14}px`,
			top: `${tooltip.y + 14}px`,
		}));

		const showTooltip = (node, event) => {
			tooltip.node = node;
			tooltip.visible = true;
			moveTooltip(event);
		};

		const moveTooltip = (event) => {
			tooltip.x = event.clientX;
			tooltip.y = event.clientY;
		};

		const hideTooltip = () => {
			tooltip.visible = false;
		};

		return {
			layoutGraph,
			tooltip,
			tooltipStyle,
			showTooltip,
			moveTooltip,
			hideTooltip,
		};
	},
};
</script>

<style scoped lang="scss">
.topology-surface {
	position: relative;
	width: 100%;
	height: 100%;
	min-height: 0;
	flex: 1 1 auto;
	border-radius: 18px;
	background: linear-gradient(90deg, rgba(148, 163, 184, 0.08) 1px, transparent 1px),
		linear-gradient(rgba(148, 163, 184, 0.08) 1px, transparent 1px),
		linear-gradient(180deg, rgba(248, 250, 252, 0.96), rgba(255, 255, 255, 0.92)), #ffffff;
	background-size: 24px 24px, 24px 24px, auto, auto;
	overflow: hidden;
}

.topology-svg {
	width: 100%;
	height: 100%;
	min-height: 0;
	display: block;
}

.topology-edge {
	fill: none;
	stroke: #64748b;
	stroke-width: 1.4;
	stroke-linecap: round;
	stroke-linejoin: round;
	opacity: 0.72;
}

.topology-node {
	cursor: default;
	filter: drop-shadow(0 6px 10px rgba(15, 23, 42, 0.1));
}

.topology-node__text {
	pointer-events: none;
}

.topology-node__title {
	font-size: 8.6px;
	font-weight: 700;
}

.topology-node__content {
	font-size: 7.8px;
	font-weight: 600;
}

.topology-empty {
	position: absolute;
	inset: 0;
	display: grid;
	place-items: center;
	align-content: center;
	gap: 10px;
	text-align: center;
	color: #64748b;
}

.topology-empty p {
	margin: 0;
	font-size: 14px;
}

.topology-tooltip {
	position: fixed;
	z-index: 3000;
	max-width: 300px;
	padding: 10px 12px;
	border-radius: 8px;
	background: rgba(255, 255, 255, 0.96);
	box-shadow: 0 14px 36px rgba(15, 23, 42, 0.18);
	pointer-events: none;
	color: #2c3e50;
}

.topology-tooltip__title {
	margin-bottom: 8px;
	font-size: 16px;
	font-weight: 700;
}

.topology-tooltip__row {
	display: flex;
	gap: 8px;
	color: #7f8c8d;
}
</style>
