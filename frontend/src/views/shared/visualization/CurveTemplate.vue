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
	name: 'CurveTemplate',
	components: { PieChart },
	props: {
		config: {
			type: Object,
			required: true,
			default: () => ({
				id: '',
				name: '',
				type: 'curve',
				variables: [],
				x_axis: 'Task Index',
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

		const safeData = computed(() => {
			return (props.data || []).map((item) => {
				const cleanItem = { taskId: item.taskId };
				(props.config.variables || []).forEach((varName) => {
					const rawValue = item[varName];
					cleanItem[varName] = rawValue !== null && rawValue !== undefined ? rawValue : 0;
				});
				return cleanItem;
			});
		});

		const activeVariables = computed(() => {
			return props.config.variables?.filter((varName) => props.variableStates[varName] !== false) || [];
		});

		const valueTypes = computed(() => {
			const types = {};
			activeVariables.value.forEach((varName) => {
				const sampleValue = safeData.value.find((entry) => entry[varName] !== undefined)?.[varName];
				types[varName] = typeof sampleValue === 'number' ? 'value' : 'category';
			});
			return types;
		});

		const discreteValueMap = ref({});

		const getDiscreteValue = (varName, value) => {
			if (!discreteValueMap.value[varName]) {
				discreteValueMap.value[varName] = {};
			}
			if (!(value in discreteValueMap.value[varName])) {
				discreteValueMap.value[varName][value] = Object.keys(discreteValueMap.value[varName]).length;
			}
			return discreteValueMap.value[varName][value];
		};

		const getOriginalDiscreteLabel = (varName, code) => {
			const mapping = discreteValueMap.value[varName] || {};
			const entry = Object.entries(mapping).find(([, value]) => value === code);
			return entry ? entry[0] : code;
		};

		const showEmptyState = computed(() => {
			const hasData = safeData.value.length > 0;
			const hasValidData = hasData && activeVariables.value.some((varName) => safeData.value.some((entry) => entry[varName] !== undefined));
			return !hasValidData;
		});

		const emptyMessage = computed(() => {
			if (!safeData.value.length) return 'No data available';
			if (!activeVariables.value.length) return 'No active variables selected';
			return 'No valid values available';
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
			if (!activeVariables.value.length || !safeData.value.length) {
				return {};
			}

			const primaryVariable = activeVariables.value[0];
			const series = activeVariables.value.map((varName, index) => {
				const values = safeData.value.map((entry) => entry[varName]);
				return {
					name: varName,
					type: 'line',
					smooth: true,
					symbol: values.length > 40 ? 'none' : 'circle',
					symbolSize: 7,
					showSymbol: values.length <= 40,
					connectNulls: false,
					progressive: 200,
					lineStyle: { width: 2.5 },
					itemStyle: { color: CHART_COLORS[index % CHART_COLORS.length] },
					data: values.map((value) => {
						if (value === undefined) return null;
						return valueTypes.value[varName] === 'category' ? getDiscreteValue(varName, value) : Number(value);
					}),
				};
			});

			return {
				color: CHART_COLORS,
				animationDuration: 450,
				tooltip: {
					trigger: 'axis',
					backgroundColor: 'rgba(15, 23, 42, 0.92)',
					borderWidth: 0,
					textStyle: { color: '#e2e8f0' },
					formatter: (items) => {
						if (!Array.isArray(items) || !items.length) return '';
						const lines = [String(items[0].axisValueLabel || items[0].name || '-')];
						items.forEach((item) => {
							const varType = valueTypes.value[item.seriesName];
							const rawValue = Array.isArray(item.value) ? item.value[1] : item.value;
							const formattedValue =
								varType === 'category'
									? getOriginalDiscreteLabel(item.seriesName, rawValue)
									: Number(rawValue).toFixed(2);
							lines.push(`${item.marker} ${item.seriesName}: ${formattedValue}`);
						});
						return lines.join('<br/>');
					},
				},
				legend: {
					top: 8,
					type: 'scroll',
					icon: 'roundRect',
					itemWidth: 12,
					textStyle: { color: '#334155' },
					data: activeVariables.value,
				},
				grid: {
					top: 56,
					left: 18,
					right: 16,
					bottom: safeData.value.length > 25 ? 52 : 26,
					containLabel: true,
				},
				xAxis: {
					type: 'category',
					name: props.config.x_axis,
					nameLocation: 'center',
					nameGap: 24,
					data: safeData.value.map((entry) => entry.taskId),
					axisLine: { lineStyle: { color: '#94a3b8' } },
					axisLabel: {
						color: '#475569',
						formatter: (value) => (String(value).length > 10 ? `${String(value).slice(0, 10)}...` : value),
					},
				},
				yAxis: {
					type: valueTypes.value[primaryVariable],
					name: props.config.y_axis,
					nameLocation: 'end',
					nameGap: 20,
					axisLine: { show: false },
					axisLabel: {
						color: '#475569',
						formatter: (value) => {
							if (valueTypes.value[primaryVariable] === 'category') {
								return getOriginalDiscreteLabel(primaryVariable, value);
							}
							return Number(value).toFixed(2);
						},
					},
					splitLine: {
						lineStyle: {
							color: 'rgba(148, 163, 184, 0.18)',
						},
					},
				},
				dataZoom:
					safeData.value.length > 25
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
