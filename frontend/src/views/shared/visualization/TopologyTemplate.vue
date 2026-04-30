<template>
	<div class="viz-surface">
		<div ref="chart" class="topology-chart"></div>

		<div v-if="showEmptyState" class="viz-empty">
			<el-icon :size="36" class="viz-empty__icon">
				<PieChart />
			</el-icon>
			<p>{{ emptyMessage }}</p>
		</div>
	</div>
</template>

<script>
import { ref, watch, onMounted, onBeforeUnmount, computed, nextTick } from 'vue';
import * as echarts from 'echarts';
import { PieChart } from '@element-plus/icons-vue';
import { graphlib, layout as dagreLayout } from '@dagrejs/dagre';

const COLOR_PALETTE = ['#2563eb', '#10b981', '#f59e0b', '#ef4444', '#0ea5e9', '#8b5cf6', '#14b8a6', '#f97316'];

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
		const chartRef = ref(null);
		const chartInstance = ref(null);
		const resizeObserver = ref(null);
		const showEmptyState = ref(true);
		const emptyMessage = ref('No topology data available');
		const colorMap = ref(new Map());

		const getContrastColor = (hex, opacity = 1) => {
			const r = parseInt(hex.slice(1, 3), 16);
			const g = parseInt(hex.slice(3, 5), 16);
			const b = parseInt(hex.slice(5, 7), 16);
			const brightness = (r * 299 + g * 587 + b * 114) / 1000;
			return brightness > 150 ? `rgba(15, 23, 42, ${opacity})` : `rgba(248, 250, 252, ${opacity})`;
		};

		const generateColor = (value) => {
			if (!colorMap.value.has(value)) {
				let hash = 0;
				for (let index = 0; index < value.length; index += 1) {
					hash = value.charCodeAt(index) + ((hash << 5) - hash);
				}
				colorMap.value.set(value, COLOR_PALETTE[Math.abs(hash) % COLOR_PALETTE.length]);
			}
			return colorMap.value.get(value);
		};

		const calculateNodeSize = (text) => {
			const lines = String(text || '').split('\n');
			const maxLineLength = Math.max(...lines.map((line) => line.length), 10);
			return [Math.min(280, Math.max(156, maxLineLength * 9)), Math.max(92, lines.length * 30 + 22)];
		};

		const getFirstNonEmptyValue = (obj, variables) => {
			if (!obj || typeof obj !== 'object' || !Array.isArray(variables)) return null;
			for (const key of variables) {
				const value = obj[key];
				if (value !== null && value !== undefined && value !== '') {
					return value;
				}
			}
			return null;
		};

		const topologyData = computed(() => {
			try {
				const variables = props.config?.variables;
				if (!Array.isArray(variables)) return null;

				const latestData = [...(props.data || [])]
					.reverse()
					.map((item) => getFirstNonEmptyValue(item, variables))
					.find((value) => value !== null);

				if (!latestData || typeof latestData !== 'object') return null;

				colorMap.value.clear();
				const nodes = [];
				const edges = [];

				Object.entries(latestData).forEach(([nodeId, nodeInfo]) => {
					const serviceName = nodeInfo?.service?.service_name || nodeId;
					const payload = nodeInfo?.service?.data ?? 'No data';
					const labelText = `${serviceName}\n${payload}`;
					const [width, height] = calculateNodeSize(labelText);
					const backgroundColor = generateColor(String(payload));
					const foregroundColor = getContrastColor(backgroundColor);

					nodes.push({
						id: nodeId,
						name: serviceName,
						data: payload,
						symbol: 'roundRect',
						symbolSize: [width, height],
						itemStyle: {
							color: backgroundColor,
							borderColor: 'rgba(15, 23, 42, 0.12)',
							borderWidth: 1,
							shadowBlur: 20,
							shadowColor: 'rgba(15, 23, 42, 0.12)',
							borderRadius: 18,
						},
						label: {
							show: true,
							position: 'inside',
							formatter: `{title|${serviceName}}\n{divider|${'─'.repeat(12)}}\n{content|${String(payload)}}`,
							rich: {
								title: {
									fontSize: 14,
									fontWeight: 700,
									color: foregroundColor,
									padding: [2, 0, 4, 0],
								},
								divider: {
									fontSize: 12,
									lineHeight: 12,
									color: getContrastColor(backgroundColor, 0.45),
								},
								content: {
									fontSize: 13,
									fontWeight: 500,
									lineHeight: 18,
									color: foregroundColor,
								},
							},
						},
					});

					(nodeInfo?.next_nodes || []).forEach((nextNodeId) => {
						edges.push({
							source: nodeId,
							target: nextNodeId,
						});
					});
				});

				const graph = new graphlib.Graph();
				graph.setGraph({
					rankdir: 'LR',
					nodesep: 44,
					ranksep: 68,
					marginx: 32,
					marginy: 32,
				});
				graph.setDefaultEdgeLabel(() => ({}));

				nodes.forEach((node) => {
					graph.setNode(node.id, {
						width: node.symbolSize[0],
						height: node.symbolSize[1],
					});
				});
				edges.forEach((edge) => {
					graph.setEdge(edge.source, edge.target);
				});
				dagreLayout(graph);

				nodes.forEach((node) => {
					const position = graph.node(node.id);
					node.x = position?.x || 0;
					node.y = position?.y || 0;
				});

				return { nodes, edges };
			} catch (error) {
				console.error('Process topology data failed:', error);
				return null;
			}
		});

		const handleResize = () => {
			chartInstance.value?.resize();
		};

		const initChart = () => {
			if (!chartRef.value) return;
			if (chartInstance.value) {
				chartInstance.value.dispose();
			}
			chartInstance.value = echarts.init(chartRef.value, null, {
				renderer: 'canvas',
				useDirtyRect: true,
			});
			window.addEventListener('resize', handleResize);
			if (typeof ResizeObserver !== 'undefined') {
				resizeObserver.value = new ResizeObserver(() => {
					handleResize();
				});
				resizeObserver.value.observe(chartRef.value);
			}
		};

		const updateChart = async () => {
			if (!chartInstance.value || !topologyData.value) return;

			await nextTick();

			chartInstance.value.setOption({
				animationDuration: 450,
				tooltip: {
					backgroundColor: 'rgba(15, 23, 42, 0.92)',
					borderWidth: 0,
					textStyle: { color: '#e2e8f0' },
					formatter: (params) => {
						if (params.dataType !== 'node') return '';
						return [
							`${params.data.name}`,
							`Data: ${params.data.data}`,
						].join('<br/>');
					},
				},
				series: [
					{
						type: 'graph',
						layout: 'none',
						left: 10,
						right: 10,
						top: 10,
						bottom: 10,
						roam: true,
						draggable: false,
						zoom: 0.92,
						edgeSymbol: ['none', 'arrow'],
						edgeSymbolSize: [0, 10],
						label: {
							show: true,
						},
						lineStyle: {
							color: 'rgba(100, 116, 139, 0.65)',
							width: 2,
							curveness: 0.14,
						},
						emphasis: {
							scale: false,
							lineStyle: {
								width: 2.4,
								color: '#475569',
							},
						},
						data: topologyData.value.nodes,
						edges: topologyData.value.edges,
					},
				],
			});

			handleResize();
		};

		watch(topologyData, async (value) => {
			showEmptyState.value = !value?.nodes?.length;
			if (!showEmptyState.value) {
				await nextTick();
				updateChart();
			}
		});

		onMounted(() => {
			initChart();
			if (props.data.length > 0) {
				updateChart();
			}
		});

		onBeforeUnmount(() => {
			resizeObserver.value?.disconnect();
			window.removeEventListener('resize', handleResize);
			chartInstance.value?.dispose();
		});

		return {
			chart: chartRef,
			showEmptyState,
			emptyMessage,
		};
	},
};
</script>

<style scoped lang="scss">
.viz-surface {
	position: relative;
	width: 100%;
	height: 100%;
	min-height: 500px;
	border-radius: 18px;
	background:
		radial-gradient(circle at top left, rgba(37, 99, 235, 0.08), transparent 28%),
		linear-gradient(180deg, rgba(248, 250, 252, 0.96), rgba(255, 255, 255, 0.92)),
		#ffffff;
}

.topology-chart {
	width: 100%;
	height: 100%;
	min-height: 500px;
}

.viz-empty {
	position: absolute;
	inset: 0;
	display: grid;
	place-items: center;
	align-content: center;
	gap: 10px;
	text-align: center;
	color: #64748b;
}

.viz-empty__icon {
	color: #94a3b8;
}

.viz-empty p {
	margin: 0;
	font-size: 14px;
}
</style>
