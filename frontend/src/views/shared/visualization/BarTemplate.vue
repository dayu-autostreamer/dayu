<template>
	<div class="viz-surface">
		<div ref="container" class="chart-wrapper"></div>

		<div v-if="showEmptyState" class="viz-empty">
			<el-icon :size="36" class="viz-empty__icon">
				<PieChart />
			</el-icon>
			<p>{{ emptyMessage }}</p>
		</div>
	</div>
</template>

<script>
import { ref, computed, onMounted, onBeforeUnmount, watch, nextTick } from 'vue';
import * as echarts from 'echarts';
import { PieChart } from '@element-plus/icons-vue';

const CHART_COLORS = ['#2563eb', '#0ea5e9', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6'];

export default {
	name: 'BarTemplate',
	components: { PieChart },
	props: {
		config: {
			type: Object,
			required: true,
			default: () => ({
				id: '',
				name: '',
				type: 'bar',
				variables: [],
				x_axis: 'Replica',
				y_axis: 'Queue Length',
			}),
		},
		data: {
			type: Array,
			required: true,
			default: () => [],
		},
		variableStates: {
			type: Object,
			required: true,
			default: () => ({}),
		},
	},

	setup(props) {
		const chart = ref(null);
		const container = ref(null);
		const resizeObserver = ref(null);

		const activeServices = computed(() => {
			return props.config.variables?.filter((serviceName) => props.variableStates[serviceName] !== false) || [];
		});

		const latestSnapshot = computed(() => {
			const snapshots = Array.isArray(props.data) ? props.data : [];
			for (let idx = snapshots.length - 1; idx >= 0; idx -= 1) {
				const snapshot = snapshots[idx];
				const hasReplicaPayload =
					snapshot &&
					typeof snapshot === 'object' &&
					(props.config.variables || []).some((serviceName) => Array.isArray(snapshot[serviceName]));
				if (hasReplicaPayload) {
					return snapshot;
				}
			}
			return snapshots.length ? snapshots[snapshots.length - 1] : null;
		});

		const latestTaskId = computed(() => latestSnapshot.value?.taskId || '');

		const normalizedSeries = computed(() => {
			const snapshot = latestSnapshot.value || {};
			const serviceOrder = [];
			const categoryOrder = [];
			const categorySet = new Set();
			const serviceMaps = {};
			const serviceMetadata = {};

			activeServices.value.forEach((serviceName) => {
				const records = Array.isArray(snapshot[serviceName]) ? snapshot[serviceName] : [];
				serviceOrder.push(serviceName);
				serviceMaps[serviceName] = new Map();

				records.forEach((record) => {
					if (!record || typeof record !== 'object') return;
					const category =
						record.replica_label || record.pod_name || record.device || `${serviceName}-${categoryOrder.length + 1}`;
					if (!categorySet.has(category)) {
						categorySet.add(category);
						categoryOrder.push(category);
					}
					serviceMaps[serviceName].set(category, Number(record.queue_length) || 0);
					serviceMetadata[`${serviceName}::${category}`] = {
						device: record.device || '',
						pod_name: record.pod_name || '',
						queue_length: Number(record.queue_length) || 0,
					};
				});
			});

			return {
				categories: categoryOrder,
				series: serviceOrder.map((serviceName, index) => ({
					name: serviceName,
					type: 'bar',
					data: categoryOrder.map((category) => serviceMaps[serviceName].get(category) ?? 0),
					barMaxWidth: 30,
					itemStyle: {
						borderRadius: [10, 10, 0, 0],
						color: CHART_COLORS[index % CHART_COLORS.length],
					},
				})),
				metadata: serviceMetadata,
			};
		});

		const showEmptyState = computed(() => {
			if (!latestSnapshot.value) return true;
			if (!activeServices.value.length) return true;
			return normalizedSeries.value.categories.length === 0;
		});

		const emptyMessage = computed(() => {
			if (!latestSnapshot.value) return 'No data available';
			if (!activeServices.value.length) return 'No active services selected';
			return 'No deployed replicas found';
		});

		const cleanupChart = () => {
			if (chart.value) {
				chart.value.dispose();
				chart.value = null;
			}
		};

		const handleResize = () => {
			chart.value?.resize();
		};

		const setupResizeHandling = () => {
			if (!container.value) return;
			if (typeof ResizeObserver !== 'undefined') {
				resizeObserver.value = new ResizeObserver(() => {
					handleResize();
				});
				resizeObserver.value.observe(container.value);
			}
			window.addEventListener('resize', handleResize);
		};

		const initChart = async () => {
			await nextTick();
			if (!container.value) return false;

			if (chart.value) {
				chart.value.dispose();
			}

			chart.value = echarts.init(container.value, null, {
				renderer: 'canvas',
				useDirtyRect: true,
			});
			return true;
		};

		const getChartOption = () => {
			const { categories, series, metadata } = normalizedSeries.value;
			return {
				color: CHART_COLORS,
				animationDuration: 450,
				tooltip: {
					trigger: 'item',
					backgroundColor: 'rgba(15, 23, 42, 0.92)',
					borderWidth: 0,
					textStyle: { color: '#e2e8f0' },
					formatter: (params) => {
						const key = `${params.seriesName}::${params.name}`;
						const info = metadata[key] || {};
						return [
							`${params.marker} ${params.seriesName}`,
							`Replica: ${params.name}`,
							`Task: ${latestTaskId.value || '-'}`,
							`Device: ${info.device || '-'}`,
							`Pod: ${info.pod_name || '-'}`,
							`Queue Length: ${Number(params.value || 0).toFixed(2)}`,
						].join('<br/>');
					},
				},
				legend: {
					top: 8,
					type: 'scroll',
					icon: 'roundRect',
					itemWidth: 12,
					textStyle: { color: '#334155' },
					data: activeServices.value,
				},
				grid: {
					top: 56,
					left: 18,
					right: 12,
					bottom: categories.length > 10 ? 52 : 28,
					containLabel: true,
				},
				xAxis: {
					type: 'category',
					name: props.config.x_axis,
					nameLocation: 'center',
					nameGap: 44,
					data: categories,
					axisLine: { lineStyle: { color: '#94a3b8' } },
					axisLabel: {
						color: '#475569',
						interval: 0,
						rotate: 24,
						formatter: (value) => (String(value).length > 22 ? `${String(value).slice(0, 22)}...` : value),
					},
				},
				yAxis: {
					type: 'value',
					name: props.config.y_axis,
					nameLocation: 'end',
					nameGap: 18,
					axisLine: { show: false },
					axisLabel: {
						color: '#475569',
						formatter: (value) => Number(value).toFixed(0),
					},
					splitLine: {
						lineStyle: {
							color: 'rgba(148, 163, 184, 0.18)',
						},
					},
				},
				dataZoom:
					categories.length > 10
						? [
								{ type: 'inside', xAxisIndex: 0 },
								{
									type: 'slider',
									xAxisIndex: 0,
									height: 16,
									bottom: 8,
									borderColor: 'transparent',
									backgroundColor: 'rgba(148, 163, 184, 0.12)',
									fillerColor: 'rgba(37, 99, 235, 0.18)',
								},
						  ]
						: [],
				series,
			};
		};

		const renderChart = async () => {
			if (showEmptyState.value) {
				cleanupChart();
				return;
			}

			if (!chart.value) {
				const success = await initChart();
				if (!success) return;
			}

			chart.value.setOption(getChartOption(), true);
			handleResize();
		};

		onMounted(() => {
			setupResizeHandling();
			if (!showEmptyState.value) {
				renderChart();
			}
		});

		onBeforeUnmount(() => {
			resizeObserver.value?.disconnect();
			window.removeEventListener('resize', handleResize);
			cleanupChart();
		});

		watch(
			() => [props.data, props.variableStates],
			() => {
				renderChart();
			},
			{ deep: true, flush: 'post' }
		);

		return {
			container,
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
	min-height: 420px;
	border-radius: 18px;
	background: linear-gradient(180deg, rgba(248, 250, 252, 0.96), rgba(255, 255, 255, 0.9)), #ffffff;
}

.chart-wrapper {
	width: 100%;
	height: 100%;
	min-height: 420px;
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
