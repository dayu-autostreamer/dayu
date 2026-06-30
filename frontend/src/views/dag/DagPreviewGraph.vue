<template>
	<div class="preview-graph">
		<svg
			v-if="layoutGraph"
			class="preview-graph__svg"
			:viewBox="`0 0 ${layoutGraph.width} ${layoutGraph.height}`"
			role="img"
			aria-label="Application dag topology preview"
		>
			<defs>
				<marker
					:id="layoutGraph.markerId"
					markerWidth="10"
					markerHeight="10"
					refX="8"
					refY="5"
					orient="auto"
					markerUnits="strokeWidth"
				>
					<path d="M 0 0 L 10 5 L 0 10 z" fill="#64748b" />
				</marker>
			</defs>

			<path
				v-for="edge in layoutGraph.edges"
				:key="edge.id"
				:d="edge.path"
				class="preview-graph__edge"
				:marker-end="`url(#${layoutGraph.markerId})`"
			/>

			<g v-for="node in layoutGraph.nodes" :key="node.id" :transform="`translate(${node.x}, ${node.y})`">
				<rect
					:width="node.width"
					:height="node.height"
					rx="16"
					:fill="node.tone.background"
					:stroke="node.tone.border"
					stroke-width="1.2"
				/>
				<rect :width="5" :height="node.height - 12" x="8" y="6" rx="3" :fill="node.tone.accent" />
				<text
					class="preview-graph__label"
					:x="node.width / 2 + 4"
					:y="node.textY"
					:font-size="node.fontSize"
					text-anchor="middle"
				>
					<tspan
						v-for="(line, index) in node.lines"
						:key="`${node.id}-${index}`"
						:x="node.width / 2 + 4"
						:dy="index === 0 ? 0 : 14"
					>
						{{ line }}
					</tspan>
				</text>
			</g>
		</svg>

		<div v-else class="preview-graph__empty">No services in this DAG</div>
	</div>
</template>

<script>
import dagre from '@dagrejs/dagre';
import { computed } from 'vue';
import { getServiceNodeFontSize, getServiceTone } from './nodePalette';

const GRAPH_PADDING = 24;
const MIN_NODE_WIDTH = 88;
const MAX_NODE_WIDTH = 164;
const BASE_NODE_HEIGHT = 40;
const NODE_LINE_HEIGHT = 14;

function hashString(value) {
	const source = String(value || '');
	let hash = 0;

	for (let i = 0; i < source.length; i += 1) {
		hash = (hash << 5) - hash + source.charCodeAt(i);
		hash |= 0;
	}

	return Math.abs(hash);
}

function chunkWord(word, maxLength) {
	if (word.length <= maxLength) {
		return [word];
	}

	const chunks = [];
	for (let i = 0; i < word.length; i += maxLength) {
		chunks.push(word.slice(i, i + maxLength));
	}

	return chunks;
}

function wrapLabel(label, maxLineLength = 14, maxLines = 3) {
	const normalized = String(label || '')
		.replace(/[_-]+/g, ' ')
		.replace(/\s+/g, ' ')
		.trim();

	if (!normalized) {
		return ['Unknown Service'];
	}

	const tokens = normalized.split(' ').flatMap((token) => chunkWord(token, maxLineLength));
	const lines = [];
	let current = '';

	tokens.forEach((token) => {
		const candidate = current ? `${current} ${token}` : token;
		if (candidate.length <= maxLineLength || !current) {
			current = candidate;
			return;
		}

		lines.push(current);
		current = token;
	});

	if (current) {
		lines.push(current);
	}

	if (lines.length <= maxLines) {
		return lines;
	}

	const limited = lines.slice(0, maxLines);
	const lastLine = limited[maxLines - 1];
	limited[maxLines - 1] = lastLine.length > maxLineLength - 1 ? `${lastLine.slice(0, maxLineLength - 1)}…` : `${lastLine}…`;
	return limited;
}

function getNodeMetrics(id, label) {
	const lines = wrapLabel(label);
	const longestLine = Math.max(...lines.map((line) => line.length));
	const width = Math.min(MAX_NODE_WIDTH, Math.max(MIN_NODE_WIDTH, longestLine * 7.2 + 30));
	const height = BASE_NODE_HEIGHT + Math.max(lines.length - 1, 0) * NODE_LINE_HEIGHT;
	const lineBlockHeight = (lines.length - 1) * NODE_LINE_HEIGHT;

	return {
		id,
		label,
		lines,
		width,
		height,
		textY: height / 2 - lineBlockHeight / 2 + 4,
		fontSize: getServiceNodeFontSize(label),
		tone: getServiceTone(id),
	};
}

function buildEdgePath(sourceNode, targetNode) {
	const startX = sourceNode.x + sourceNode.width;
	const startY = sourceNode.y + sourceNode.height / 2;
	const endX = targetNode.x;
	const endY = targetNode.y + targetNode.height / 2;
	const curve = Math.max((endX - startX) * 0.45, 28);

	return `M ${startX} ${startY} C ${startX + curve} ${startY}, ${endX - curve} ${endY}, ${endX} ${endY}`;
}

function buildLayout(dag) {
	const dagEntries = Object.entries(dag || {}).filter(([key]) => key !== '_start');
	if (!dagEntries.length) {
		return null;
	}

	const graph = new dagre.graphlib.Graph();
	graph.setDefaultEdgeLabel(() => ({}));
	graph.setGraph({
		rankdir: 'LR',
		nodesep: 26,
		ranksep: 54,
		marginx: 0,
		marginy: 0,
	});

	const nodes = dagEntries.map(([id, node]) => {
		const label = node?.service_id || node?.id || id;
		const metrics = getNodeMetrics(id, label);
		graph.setNode(id, {
			width: metrics.width,
			height: metrics.height,
		});
		return metrics;
	});

	const nodeIds = new Set(nodes.map((node) => node.id));
	const edges = [];

	dagEntries.forEach(([source, node]) => {
		if (!Array.isArray(node?.succ)) {
			return;
		}

		node.succ.forEach((target) => {
			if (!nodeIds.has(target)) {
				return;
			}

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
		(accumulator, node) => ({
			minX: Math.min(accumulator.minX, node.x),
			minY: Math.min(accumulator.minY, node.y),
			maxX: Math.max(accumulator.maxX, node.x + node.width),
			maxY: Math.max(accumulator.maxY, node.y + node.height),
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
			if (!sourceNode || !targetNode) {
				return null;
			}

			return {
				...edge,
				path: buildEdgePath(sourceNode, targetNode),
			};
		})
		.filter(Boolean);

	return {
		nodes: normalizedNodes,
		edges: normalizedEdges,
		width: Math.max(bounds.maxX - bounds.minX + GRAPH_PADDING * 2, 220),
		height: Math.max(bounds.maxY - bounds.minY + GRAPH_PADDING * 2, 140),
		markerId: `dag-preview-arrow-${hashString(dagEntries.map(([id]) => id).join('-'))}`,
	};
}

export default {
	name: 'DagPreviewGraph',
	props: {
		dag: {
			type: Object,
			required: true,
		},
	},
	setup(props) {
		const layoutGraph = computed(() => buildLayout(props.dag));

		return {
			layoutGraph,
		};
	},
};
</script>

<style scoped lang="scss">
.preview-graph {
	height: 100%;
	width: 100%;
	display: flex;
	align-items: stretch;
	justify-content: stretch;
	background:
		linear-gradient(90deg, rgba(148, 163, 184, 0.08) 1px, transparent 1px),
		linear-gradient(rgba(148, 163, 184, 0.08) 1px, transparent 1px);
	background-size: 24px 24px;
}

.preview-graph__svg {
	width: 100%;
	height: 100%;
}

.preview-graph__edge {
	fill: none;
	stroke: #64748b;
	stroke-width: 2;
	stroke-linecap: round;
	stroke-linejoin: round;
}

.preview-graph__label {
	fill: #0f172a;
	font-weight: 700;
	dominant-baseline: middle;
}

.preview-graph__empty {
	width: 100%;
	display: flex;
	align-items: center;
	justify-content: center;
	font-size: 13px;
	font-weight: 600;
	color: #64748b;
}
</style>
