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
const MIN_NODE_SCALE = 0.18;

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
		const resizeFrame = ref(null);
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
			return [Math.min(180, Math.max(94, maxLineLength * 6.8)), Math.max(52, lines.length * 18 + 12)];
		};

		const truncateLabel = (value, maxLength = 24) => {
			const text = String(value ?? '');
			return text.length > maxLength ? `${text.slice(0, maxLength)}...` : text;
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
					const displayPayload = truncateLabel(payload, 26);
					const labelText = `${serviceName}\n${displayPayload}`;
					const [width, height] = calculateNodeSize(labelText);
					const backgroundColor = generateColor(String(payload));
					const foregroundColor = getContrastColor(backgroundColor);

					nodes.push({
						id: nodeId,
						name: serviceName,
						data: payload,
						displayData: displayPayload,
						symbol: 'roundRect',
						symbolSize: [width, height],
						itemStyle: {
							color: backgroundColor,
							borderColor: 'rgba(15, 23, 42, 0.12)',
							borderWidth: 1,
							shadowBlur: 12,
							shadowColor: 'rgba(15, 23, 42, 0.1)',
							borderRadius: 12,
						},
						label: {
							show: true,
							position: 'inside',
							formatter: `{title|${truncateLabel(serviceName, 22)}}\n{divider|${'─'.repeat(8)}}\n{content|${displayPayload}}`,
							rich: {
								title: {
									fontSize: 11,
									fontWeight: 700,
									color: foregroundColor,
									padding: [0, 0, 1, 0],
								},
								divider: {
									fontSize: 9,
									lineHeight: 8,
									color: getContrastColor(backgroundColor, 0.45),
								},
								content: {
									fontSize: 10,
									fontWeight: 500,
									lineHeight: 13,
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
					nodesep: 6,
					ranksep: 8,
					marginx: 4,
					marginy: 4,
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

				const bounds = nodes.reduce(
					(acc, node) => {
						const [width, height] = node.symbolSize;
						acc.minX = Math.min(acc.minX, node.x - width / 2);
						acc.maxX = Math.max(acc.maxX, node.x + width / 2);
						acc.minY = Math.min(acc.minY, node.y - height / 2);
						acc.maxY = Math.max(acc.maxY, node.y + height / 2);
						return acc;
					},
					{
						minX: Number.POSITIVE_INFINITY,
						maxX: Number.NEGATIVE_INFINITY,
						minY: Number.POSITIVE_INFINITY,
						maxY: Number.NEGATIVE_INFINITY,
					}
				);

				bounds.width = Math.max(1, bounds.maxX - bounds.minX);
				bounds.height = Math.max(1, bounds.maxY - bounds.minY);
				bounds.centerX = bounds.minX + bounds.width / 2;
				bounds.centerY = bounds.minY + bounds.height / 2;

				return { nodes, edges, bounds };
			} catch (error) {
				console.error('Process topology data failed:', error);
				return null;
			}
		});

		const handleResize = () => {
			chartInstance.value?.resize();
		};

		const getFittedTopologyData = () => {
			const data = topologyData.value;
			const rect = chartRef.value?.getBoundingClientRect();
			if (!data || !rect?.width || !rect?.height) {
				return data;
			}

			const bounds = data.bounds;
			const padding = 16;
			const availableWidth = Math.max(120, rect.width - padding * 2);
			const availableHeight = Math.max(120, rect.height - padding * 2);
			const scale = Math.max(MIN_NODE_SCALE, Math.min(1, availableWidth / bounds.width, availableHeight / bounds.height));
			const centerX = rect.width / 2;
			const centerY = rect.height / 2;

			const nodes = data.nodes.map((node) => {
				const [width, height] = node.symbolSize;
				const scaledWidth = Math.max(36, width * scale);
				const scaledHeight = Math.max(24, height * scale);
				const labelScale = Math.max(0.55, scale);
				return {
					...node,
					x: centerX + (node.x - bounds.centerX) * scale,
					y: centerY + (node.y - bounds.centerY) * scale,
					symbolSize: [scaledWidth, scaledHeight],
					itemStyle: {
						...node.itemStyle,
						shadowBlur: Math.max(4, 10 * scale),
					},
					label: {
						...node.label,
						rich: {
							title: {
								...node.label.rich.title,
								fontSize: Math.max(7, 11 * labelScale),
							},
							divider: {
								...node.label.rich.divider,
								fontSize: Math.max(6, 9 * labelScale),
								lineHeight: Math.max(5, 8 * labelScale),
							},
							content: {
								...node.label.rich.content,
								fontSize: Math.max(6, 10 * labelScale),
								lineHeight: Math.max(8, 13 * labelScale),
							},
						},
					},
				};
			});

			return {
				nodes,
				edges: data.edges,
			};
		};

		const scheduleFit = () => {
			if (resizeFrame.value) {
				cancelAnimationFrame(resizeFrame.value);
			}
			resizeFrame.value = requestAnimationFrame(() => {
				resizeFrame.value = null;
				handleResize();
				updateChart();
			});
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
			window.addEventListener('resize', scheduleFit);
			if (typeof ResizeObserver !== 'undefined') {
				resizeObserver.value = new ResizeObserver(() => {
					scheduleFit();
				});
				resizeObserver.value.observe(chartRef.value);
			}
		};

		const updateChart = async () => {
			if (!chartInstance.value || !topologyData.value) return;

			await nextTick();
			const fittedData = getFittedTopologyData();

			chartInstance.value.setOption({
				animationDuration: 450,
				tooltip: {
					backgroundColor: 'rgba(255, 255, 255, 0.95)',
					borderWidth: 0,
					formatter: (params) => {
						if (params.dataType !== 'node') return '';
						return `
							<div style="max-width: 300px">
								<div style="font-size:16px;font-weight:bold;color:#2c3e50;margin-bottom:8px">
									${params.data.name}
								</div>
								<div style="color:#7f8c8d">
									Data:
									<span style="color:${params.data.itemStyle.color};font-weight:500">
										${params.data.data}
									</span>
								</div>
							</div>
						`;
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
						zoom: 1,
						nodeScaleRatio: 1,
						scaleLimit: {
							min: 0.15,
							max: 2,
						},
						edgeSymbol: ['none', 'arrow'],
						edgeSymbolSize: [0, 8],
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
						data: fittedData.nodes,
						edges: fittedData.edges,
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
			if (resizeFrame.value) {
				cancelAnimationFrame(resizeFrame.value);
			}
			resizeObserver.value?.disconnect();
			window.removeEventListener('resize', scheduleFit);
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
