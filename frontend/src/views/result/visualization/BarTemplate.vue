<template>
	<div class="chart-container">
		<div ref="container" class="chart-wrapper"></div>
		<div v-if="showEmptyState" class="empty-state">
			<el-icon :size="40">
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

			const series = serviceOrder.map((serviceName) => ({
				name: serviceName,
				type: 'bar',
				data: categoryOrder.map((category) => serviceMaps[serviceName].get(category) ?? 0),
				barMaxWidth: 32,
			}));

			return {
				categories: categoryOrder,
				series,
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

		const cleanChart = () => {
			if (chart.value) {
				chart.value.dispose();
				chart.value = null;
			}
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
				animation: true,
				tooltip: {
					trigger: 'item',
					formatter: (params) => {
						const key = `${params.seriesName}::${params.name}`;
						const info = metadata[key] || {};
						return [
							`Task: ${latestTaskId.value || '-'}`,
							`${params.marker} ${params.seriesName}`,
							`Replica: ${params.name}`,
							`Device: ${info.device || '-'}`,
							`Pod: ${info.pod_name || '-'}`,
							`Queue Length: ${Number(params.value || 0).toFixed(2)}`,
						].join('<br/>');
					},
				},
				legend: {
					data: activeServices.value,
					type: 'scroll',
				},
				grid: {
					left: '3%',
					right: '4%',
					bottom: '20%',
					containLabel: true,
				},
				xAxis: {
					type: 'category',
					name: props.config.x_axis,
					nameLocation: 'center',
					nameGap: 48,
					data: categories,
					axisLabel: {
						interval: 0,
						rotate: 25,
						formatter: (value) => (value.length > 24 ? `${value.slice(0, 24)}...` : value),
					},
				},
				yAxis: {
					type: 'value',
					name: props.config.y_axis,
					nameLocation: 'end',
					nameGap: 20,
					axisLabel: {
						formatter: (value) => Number(value).toFixed(0),
					},
				},
				dataZoom: categories.length > 10
					? [
							{ type: 'inside', xAxisIndex: 0 },
							{ type: 'slider', xAxisIndex: 0, height: 16, bottom: 8 },
					  ]
					: [],
				series,
			};
		};

		const renderChart = async () => {
			if (showEmptyState.value) {
				cleanChart();
				return;
			}
			if (!chart.value) {
				const success = await initChart();
				if (!success) return;
			}
			chart.value.setOption(getChartOption(), true);
		};

		onMounted(() => {
			if (!showEmptyState.value) {
				renderChart();
			}
		});

		onBeforeUnmount(() => {
			cleanChart();
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

<style scoped>
.chart-container {
	position: relative;
	width: 100%;
	height: 100%;
	min-height: 420px;
}

.chart-wrapper {
	width: 100%;
	height: 100%;
	min-height: 320px;
}

.empty-state {
	position: absolute;
	top: 50%;
	left: 50%;
	transform: translate(-50%, -50%);
	text-align: center;
	color: var(--el-text-color-secondary);
}

.empty-state p {
	margin-top: 10px;
	font-size: 14px;
}
</style>
