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

const CHART_COLORS = ['#2563eb', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#0ea5e9'];

export default {
	name: 'CDFTemplate',
	components: { PieChart },
	props: {
		config: {
			type: Object,
			required: true,
			default: () => ({
				id: '',
				name: '',
				type: 'cdf',
				variables: [],
				x_axis: '',
				y_axis: '',
			}),
		},
		data: {
			type: Array,
			required: true,
			validator: (value) => Array.isArray(value) && value.every((item) => item.taskId !== undefined),
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
		const mutationObserver = ref(null);

		const activeVariables = computed(() => {
			return props.config.variables?.filter((varName) => props.variableStates[varName] === true) || [];
		});

		const safeData = computed(() => {
			const result = {};

			if (!props.config.variables?.length) {
				return result;
			}

			(props.config.variables || []).forEach((varName) => {
				if (props.variableStates[varName] !== true) return;
				const values = (props.data || [])
					.map((entry) => entry[varName])
					.filter((value) => value !== undefined && value !== null && !Number.isNaN(Number(value)))
					.map(Number)
					.sort((a, b) => a - b);

				if (!values.length) return;

				const total = values.length;
				const uniqueValues = [...new Set(values)];
				result[varName] = uniqueValues.map((value) => ({
					value,
					probability: values.filter((item) => item <= value).length / total,
				}));
			});

			return result;
		});

		const showEmptyState = computed(() => {
			const hasData = Object.values(safeData.value).some((series) => series?.length > 0);
			return !(hasData && activeVariables.value.length > 0);
		});

		const emptyMessage = computed(() => {
			if (!(props.data || []).length) return 'No data available';
			if (!activeVariables.value.length) return 'No active variables selected';
			return 'No valid numeric data available';
		});

		const cleanupChart = () => {
			if (chart.value) {
				chart.value.dispose();
				chart.value = null;
			}
			if (container.value) {
				container.value.innerHTML = '';
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

			mutationObserver.value = new MutationObserver(() => {
				handleResize();
			});
			mutationObserver.value.observe(container.value, {
				attributes: true,
				attributeFilter: ['style', 'class'],
			});

			window.addEventListener('resize', handleResize);
		};

		const initChart = async () => {
			await nextTick();
			if (!container.value) return false;

			let checks = 0;
			while (checks < 10) {
				const rect = container.value.getBoundingClientRect();
				if (rect.width > 0 && rect.height > 0) break;
				await new Promise((resolve) => setTimeout(resolve, 50));
				checks += 1;
			}

			const rect = container.value.getBoundingClientRect();
			if (!rect.width || !rect.height) {
				return false;
			}

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
			const hasData = Object.values(safeData.value).some((series) => series?.length > 0);
			if (!hasData || !activeVariables.value.length) {
				return {};
			}

			const series = Object.entries(safeData.value).map(([varName, points], index) => ({
				name: varName,
				type: 'line',
				smooth: true,
				symbol: 'emptyCircle',
				symbolSize: 7,
				showSymbol: true,
				lineStyle: { width: 2.5 },
				itemStyle: {
					color: CHART_COLORS[index % CHART_COLORS.length],
					borderWidth: 2,
				},
				areaStyle: { opacity: 0.08 },
				data: points.map((point) => [point.value, point.probability]),
			}));

			return {
				color: CHART_COLORS,
				animationDuration: 450,
				tooltip: {
					trigger: 'item',
					backgroundColor: 'rgba(15, 23, 42, 0.92)',
					borderWidth: 0,
					textStyle: { color: '#e2e8f0' },
					formatter: (params) =>
						[
							`${params.marker} ${params.seriesName}`,
							`Value: ${Number(params.value[0]).toFixed(2)}`,
							`Probability: ${(Number(params.value[1]) * 100).toFixed(1)}%`,
						].join('<br/>'),
				},
				legend: {
					top: 8,
					type: 'scroll',
					icon: 'roundRect',
					itemWidth: 12,
					textStyle: { color: '#334155' },
					data: Object.keys(safeData.value),
				},
				grid: {
					top: 56,
					left: 18,
					right: 16,
					bottom: 28,
					containLabel: true,
				},
				xAxis: {
					name: props.config.x_axis,
					nameLocation: 'center',
					nameGap: 26,
					type: 'value',
					min: 'dataMin',
					max: 'dataMax',
					axisLine: { lineStyle: { color: '#94a3b8' } },
					axisLabel: { color: '#475569' },
					splitLine: {
						lineStyle: {
							color: 'rgba(148, 163, 184, 0.18)',
						},
					},
				},
				yAxis: {
					name: props.config.y_axis,
					type: 'value',
					min: 0,
					max: 1,
					axisLine: { show: false },
					axisLabel: {
						color: '#475569',
						formatter: (value) => `${(Number(value) * 100).toFixed(0)}%`,
					},
					splitLine: {
						lineStyle: {
							color: 'rgba(148, 163, 184, 0.18)',
						},
					},
				},
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
			setTimeout(() => {
				if (!showEmptyState.value) {
					renderChart();
				}
			}, 250);
		});

		onBeforeUnmount(() => {
			resizeObserver.value?.disconnect();
			mutationObserver.value?.disconnect();
			window.removeEventListener('resize', handleResize);
			cleanupChart();
		});

		watch(showEmptyState, (value) => {
			if (value) {
				cleanupChart();
				return;
			}
			renderChart();
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
	min-height: 400px;
	border-radius: 18px;
	background:
		linear-gradient(180deg, rgba(248, 250, 252, 0.96), rgba(255, 255, 255, 0.9)),
		#ffffff;
}

.chart-wrapper {
	width: 100%;
	height: 100%;
	min-height: 400px;
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
