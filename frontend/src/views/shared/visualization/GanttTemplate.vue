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
import { computed, nextTick, onBeforeUnmount, onMounted, ref, shallowRef, watch } from 'vue';
import * as echarts from 'echarts';
import { PieChart } from '@element-plus/icons-vue';

const MAX_VISIBLE_LANES = 10;
const GOLDEN_ANGLE = 137.50776405003785;
const BAR_OPACITY = 0.86;
const DARK_LABEL_COLOR = '#0f172a';
const LIGHT_LABEL_COLOR = '#ffffff';

function stableHash(value) {
	let hash = 2166136261;
	const source = String(value ?? '');
	for (let index = 0; index < source.length; index += 1) {
		hash ^= source.charCodeAt(index);
		hash = Math.imul(hash, 16777619);
	}
	hash ^= hash >>> 16;
	hash = Math.imul(hash, 0x7feb352d);
	hash ^= hash >>> 15;
	hash = Math.imul(hash, 0x846ca68b);
	hash ^= hash >>> 16;
	return hash >>> 0;
}

function hslToRgb(hue, saturation, lightness) {
	const normalizedHue = (((hue % 360) + 360) % 360) / 60;
	const normalizedSaturation = saturation / 100;
	const normalizedLightness = lightness / 100;
	const chroma = (1 - Math.abs(2 * normalizedLightness - 1)) * normalizedSaturation;
	const secondary = chroma * (1 - Math.abs((normalizedHue % 2) - 1));
	let red = 0;
	let green = 0;
	let blue = 0;

	if (normalizedHue < 1) [red, green, blue] = [chroma, secondary, 0];
	else if (normalizedHue < 2) [red, green, blue] = [secondary, chroma, 0];
	else if (normalizedHue < 3) [red, green, blue] = [0, chroma, secondary];
	else if (normalizedHue < 4) [red, green, blue] = [0, secondary, chroma];
	else if (normalizedHue < 5) [red, green, blue] = [secondary, 0, chroma];
	else [red, green, blue] = [chroma, 0, secondary];

	const match = normalizedLightness - chroma / 2;
	return [red + match, green + match, blue + match].map((channel) => channel * 255);
}

function relativeLuminance(rgb) {
	const [red, green, blue] = rgb.map((channel) => {
		const value = channel / 255;
		return value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4;
	});
	return 0.2126 * red + 0.7152 * green + 0.0722 * blue;
}

function getLabelTone(hue, saturation, lightness) {
	const displayedRgb = hslToRgb(hue, saturation, lightness).map(
		(channel) => channel * BAR_OPACITY + 255 * (1 - BAR_OPACITY)
	);
	const backgroundLuminance = relativeLuminance(displayedRgb);
	const darkLuminance = relativeLuminance([15, 23, 42]);
	const lightContrast = 1.05 / (backgroundLuminance + 0.05);
	const darkContrast = (backgroundLuminance + 0.05) / (darkLuminance + 0.05);
	return darkContrast >= lightContrast ? 0 : 1;
}

function getTaskPalette(taskId) {
	const hash = stableHash(`task:${taskId}`);
	const numericTaskId = Number(taskId);
	const hueSeed = Number.isSafeInteger(numericTaskId) ? numericTaskId : hash;
	const hue = (((hueSeed * GOLDEN_ANGLE) % 360) + 360) % 360;
	const saturation = 58 + ((hash >>> 8) % 4) * 7;
	const lightness = 38 + ((hash >>> 12) % 4) * 7;

	return {
		barColor: `hsl(${hue.toFixed(3)}, ${saturation}%, ${lightness}%)`,
		labelTone: getLabelTone(hue, saturation, lightness),
	};
}

function normalizePayload(payload) {
	if (Array.isArray(payload)) {
		return { lanes: [], segments: payload };
	}
	if (!payload || typeof payload !== 'object') {
		return { lanes: [], segments: [] };
	}
	return {
		lanes: Array.isArray(payload.lanes) ? payload.lanes : [],
		segments: Array.isArray(payload.segments) ? payload.segments : [],
	};
}

function formatTimestamp(timestampMs, includeDate = true) {
	const date = new Date(timestampMs);
	if (Number.isNaN(date.getTime())) return '-';
	const pad = (value, width = 2) => String(value).padStart(width, '0');
	const time = `${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}.${pad(
		date.getMilliseconds(),
		3
	)}`;
	if (!includeDate) return time;
	return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())} ${time}`;
}

function formatUnixSeconds(value) {
	return Number(value).toFixed(6).replace(/0+$/, '').replace(/\.$/, '');
}

function formatDuration(durationMs) {
	const durationSeconds = durationMs / 1000;
	return `${durationSeconds.toFixed(durationSeconds < 0.001 ? 6 : 3)} s`;
}

function escapeHtml(value) {
	return String(value ?? '')
		.replace(/&/g, '&amp;')
		.replace(/</g, '&lt;')
		.replace(/>/g, '&gt;')
		.replace(/"/g, '&quot;')
		.replace(/'/g, '&#039;');
}

function assignOverlapTracks(segments) {
	const segmentsByLane = new Map();
	segments.forEach((segment) => {
		if (!segmentsByLane.has(segment.lane)) {
			segmentsByLane.set(segment.lane, []);
		}
		segmentsByLane.get(segment.lane).push(segment);
	});

	const result = [];
	segmentsByLane.forEach((laneSegments) => {
		const sortedSegments = [...laneSegments].sort(
			(a, b) => a.startMs - b.startMs || a.endMs - b.endMs || a.taskId.localeCompare(b.taskId)
		);
		const trackEnds = [];
		const assignedSegments = sortedSegments.map((segment) => {
			let trackIndex = trackEnds.findIndex((endTime) => endTime <= segment.startMs);
			if (trackIndex === -1) {
				trackIndex = trackEnds.length;
				trackEnds.push(segment.endMs);
			} else {
				trackEnds[trackIndex] = segment.endMs;
			}
			return { ...segment, trackIndex };
		});

		const trackCount = Math.max(trackEnds.length, 1);
		assignedSegments.forEach((segment) => result.push({ ...segment, trackCount }));
	});
	return result;
}

function renderGanttItem(params, api) {
	const laneIndex = api.value(0);
	const taskId = String(api.value(5) ?? '');
	const labelColor = Number(api.value(6)) === 0 ? DARK_LABEL_COLOR : LIGHT_LABEL_COLOR;
	const startPoint = api.coord([api.value(1), laneIndex]);
	const endPoint = api.coord([api.value(2), laneIndex]);
	const laneHeight = Math.abs(api.size([0, 1])[1]);
	const trackIndex = Number(api.value(3)) || 0;
	const trackCount = Math.max(Number(api.value(4)) || 1, 1);
	const availableHeight = Math.min(laneHeight * 0.76, 30);
	const trackHeight = availableHeight / trackCount;
	const barHeight = Math.max(trackHeight * 0.78, 2);
	const centerY = startPoint[1] - availableHeight / 2 + trackHeight * (trackIndex + 0.5);
	const shape = echarts.graphic.clipRectByRect(
		{
			x: startPoint[0],
			y: centerY - barHeight / 2,
			width: Math.max(endPoint[0] - startPoint[0], 2),
			height: barHeight,
		},
		{
			x: params.coordSys.x,
			y: params.coordSys.y,
			width: params.coordSys.width,
			height: params.coordSys.height,
		}
	);

	if (!shape) return null;
	shape.r = Math.min(4, shape.height / 2);
	return {
		type: 'group',
		clipPath: {
			type: 'rect',
			shape: { ...shape },
		},
		children: [
			{
				type: 'rect',
				shape,
				style: {
					fill: api.visual('color'),
					opacity: BAR_OPACITY,
					stroke: 'rgba(15, 23, 42, 0.24)',
					lineWidth: 1,
				},
				emphasis: {
					style: {
						fill: api.visual('color'),
						opacity: 1,
						stroke: 'rgba(15, 23, 42, 0.72)',
						lineWidth: 1.5,
					},
				},
			},
			{
				type: 'text',
				silent: true,
				style: {
					x: shape.x + shape.width / 2,
					y: shape.y + shape.height / 2,
					text: taskId,
					width: Math.max(shape.width - 4, 1),
					overflow: 'truncate',
					ellipsis: '…',
					align: 'center',
					verticalAlign: 'middle',
					fill: labelColor,
					fontSize: Math.max(8, Math.min(11, shape.height * 0.58)),
					fontWeight: 700,
				},
			},
		],
	};
}

export default {
	name: 'GanttTemplate',
	components: { PieChart },
	props: {
		config: {
			type: Object,
			required: true,
			default: () => ({
				id: '',
				name: '',
				type: 'gantt',
				variables: [],
				x_axis: 'Time',
				y_axis: '',
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
		const chart = shallowRef(null);
		const container = ref(null);
		const resizeObserver = ref(null);
		const mutationObserver = ref(null);
		let unmounted = false;
		let retryTimer = null;

		const activeVariables = computed(
			() => props.config.variables?.filter((variable) => props.variableStates[variable] !== false) || []
		);

		const normalizedTimeline = computed(() => {
			const lanes = [];
			const laneSet = new Set();
			const rawSegments = [];
			const segmentKeys = new Set();

			const addLane = (value) => {
				if (value === null || value === undefined) return '';
				const lane = String(value).trim();
				if (!lane) return '';
				if (!laneSet.has(lane)) {
					laneSet.add(lane);
					lanes.push(lane);
				}
				return lane;
			};

			(props.data || []).forEach((snapshot) => {
				activeVariables.value.forEach((variable) => {
					const payload = normalizePayload(snapshot?.[variable]);
					payload.lanes.forEach(addLane);

					payload.segments.forEach((segment) => {
						if (!segment || typeof segment !== 'object') return;
						const lane = addLane(segment.lane);
						const taskIdValue = segment.task_id ?? snapshot?.taskId;
						const startSeconds = Number(segment.start_time);
						const endSeconds = Number(segment.end_time);
						if (
							!lane ||
							taskIdValue === null ||
							taskIdValue === undefined ||
							!Number.isFinite(startSeconds) ||
							!Number.isFinite(endSeconds) ||
							endSeconds < startSeconds
						) {
							return;
						}

						const taskId = String(taskIdValue);
						const segmentKey = [
							taskId,
							lane,
							segment.service || '',
							segment.device || '',
							startSeconds,
							endSeconds,
						].join('|');
						if (segmentKeys.has(segmentKey)) return;
						segmentKeys.add(segmentKey);

						rawSegments.push({
							taskId,
							lane,
							service: String(segment.service || ''),
							device: String(segment.device || ''),
							startSeconds,
							endSeconds,
							startMs: startSeconds * 1000,
							endMs: endSeconds * 1000,
						});
					});
				});
			});

			const laneIndexes = new Map(lanes.map((lane, index) => [lane, index]));
			const segments = assignOverlapTracks(rawSegments).map((segment) => ({
				...segment,
				laneIndex: laneIndexes.get(segment.lane),
			}));

			return { lanes, segments };
		});

		const showEmptyState = computed(
			() => !activeVariables.value.length || normalizedTimeline.value.segments.length === 0
		);

		const emptyMessage = computed(() => {
			if (!(props.data || []).length) return 'No data available';
			if (!activeVariables.value.length) return 'No active variables selected';
			return 'No complete execute intervals available';
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
				resizeObserver.value = new ResizeObserver(handleResize);
				resizeObserver.value.observe(container.value);
			}
			if (typeof MutationObserver !== 'undefined') {
				mutationObserver.value = new MutationObserver(handleResize);
				mutationObserver.value.observe(container.value, {
					attributes: true,
					attributeFilter: ['style', 'class'],
				});
			}
			window.addEventListener('resize', handleResize);
		};

		const initChart = async () => {
			await nextTick();
			if (unmounted || !container.value) return false;

			let checks = 0;
			while (checks < 10) {
				if (unmounted || !container.value) return false;
				const rect = container.value.getBoundingClientRect();
				if (rect.width > 0 && rect.height > 0) break;
				await new Promise((resolve) => setTimeout(resolve, 50));
				checks += 1;
			}

			if (unmounted || !container.value) return false;
			const rect = container.value.getBoundingClientRect();
			if (!rect.width || !rect.height) return false;
			cleanupChart();
			chart.value = echarts.init(container.value, null, {
				renderer: 'canvas',
				useDirtyRect: true,
			});
			return true;
		};

		const getChartOption = () => {
			const { lanes, segments } = normalizedTimeline.value;
			if (!lanes.length || !segments.length) return {};
			const paletteCache = new Map();

			const series = [
				{
					name: 'Task intervals',
					type: 'custom',
					coordinateSystem: 'cartesian2d',
					renderItem: renderGanttItem,
					encode: { x: [1, 2], y: 0 },
					data: segments.map((segment) => {
						if (!paletteCache.has(segment.taskId)) {
							paletteCache.set(segment.taskId, getTaskPalette(segment.taskId));
						}
						const palette = paletteCache.get(segment.taskId);
						return {
							value: [
								segment.laneIndex,
								segment.startMs,
								segment.endMs,
								segment.trackIndex,
								segment.trackCount,
								segment.taskId,
								palette.labelTone,
							],
							itemStyle: { color: palette.barColor },
							meta: segment,
						};
					}),
				},
			];

			const { minTime, maxTime } = segments.reduce(
				(range, segment) => ({
					minTime: Math.min(range.minTime, segment.startMs),
					maxTime: Math.max(range.maxTime, segment.endMs),
				}),
				{ minTime: Infinity, maxTime: -Infinity }
			);
			const spansMultipleDates = new Date(minTime).toDateString() !== new Date(maxTime).toDateString();
			const dataZoom = [
				{
					id: 'time-inside',
					type: 'inside',
					xAxisIndex: 0,
					filterMode: 'weakFilter',
				},
				{
					id: 'time-slider',
					type: 'slider',
					xAxisIndex: 0,
					filterMode: 'weakFilter',
					height: 16,
					bottom: 8,
					showDetail: false,
					borderColor: 'transparent',
					backgroundColor: 'rgba(148, 163, 184, 0.12)',
					fillerColor: 'rgba(37, 99, 235, 0.18)',
				},
			];
			dataZoom.push({
				id: 'lane-slider',
				type: 'slider',
				yAxisIndex: 0,
				filterMode: 'weakFilter',
				show: lanes.length > MAX_VISIBLE_LANES,
				right: 8,
				top: 32,
				bottom: 54,
				width: 14,
				start: 0,
				end: lanes.length > MAX_VISIBLE_LANES ? (MAX_VISIBLE_LANES / lanes.length) * 100 : 100,
				showDetail: false,
				borderColor: 'transparent',
				backgroundColor: 'rgba(148, 163, 184, 0.12)',
				fillerColor: 'rgba(37, 99, 235, 0.18)',
			});

			return {
				animationDuration: 350,
				aria: {
					enabled: true,
					label: { description: props.config.name || 'Task execution Gantt chart' },
				},
				tooltip: {
					trigger: 'item',
					confine: true,
					backgroundColor: 'rgba(15, 23, 42, 0.94)',
					borderWidth: 0,
					textStyle: { color: '#e2e8f0' },
					formatter: (params) => {
						const segment = params?.data?.meta;
						if (!segment) return '';
						return [
							`${params.marker} Task ${escapeHtml(segment.taskId)}`,
							`Lane: ${escapeHtml(segment.lane)}`,
							`Service: ${escapeHtml(segment.service || '-')}`,
							`Device: ${escapeHtml(segment.device || '-')}`,
							`Start: ${formatTimestamp(segment.startMs)} (${formatUnixSeconds(segment.startSeconds)})`,
							`End: ${formatTimestamp(segment.endMs)} (${formatUnixSeconds(segment.endSeconds)})`,
							`Duration: ${formatDuration(segment.endMs - segment.startMs)}`,
						].join('<br/>');
					},
				},
				grid: {
					top: 28,
					left: 20,
					right: lanes.length > MAX_VISIBLE_LANES ? 34 : 18,
					bottom: 56,
					containLabel: true,
				},
				xAxis: {
					type: 'time',
					name: props.config.x_axis || 'Time',
					nameLocation: 'center',
					nameGap: 42,
					min: 'dataMin',
					max: 'dataMax',
					axisLine: { lineStyle: { color: '#94a3b8' } },
					axisLabel: {
						color: '#475569',
						hideOverlap: true,
						formatter: (value) => formatTimestamp(value, spansMultipleDates),
					},
					splitLine: { lineStyle: { color: 'rgba(148, 163, 184, 0.18)' } },
					axisPointer: { label: { formatter: ({ value }) => formatTimestamp(value) } },
				},
				yAxis: {
					type: 'category',
					name: props.config.y_axis || '',
					nameLocation: 'start',
					nameGap: 14,
					nameTextStyle: { color: '#64748b', fontWeight: 600 },
					inverse: true,
					data: lanes,
					axisLine: { lineStyle: { color: '#94a3b8' } },
					axisTick: { show: false },
					axisLabel: {
						color: '#475569',
						width: 180,
						overflow: 'truncate',
					},
					splitLine: { show: true, lineStyle: { color: 'rgba(148, 163, 184, 0.14)' } },
				},
				dataZoom,
				series,
			};
		};

		const renderChart = async () => {
			if (unmounted) return;
			if (showEmptyState.value) {
				cleanupChart();
				return;
			}
			if (!chart.value) {
				const success = await initChart();
				if (!success) return;
			}
			if (unmounted || !chart.value) return;
			const option = getChartOption();
			const currentOption = chart.value.getOption() || {};
			const previousZooms = new Map((currentOption.dataZoom || []).map((zoom) => [zoom.id, zoom]));
			option.dataZoom = option.dataZoom.map((zoom) => {
				const previous = previousZooms.get(zoom.id);
				if (!previous || (zoom.id === 'lane-slider' && (!zoom.show || previous.show === false))) {
					return zoom;
				}
				return { ...zoom, start: previous.start, end: previous.end };
			});
			chart.value.setOption(option, true);
			handleResize();
		};

		onMounted(() => {
			setupResizeHandling();
			renderChart();
			retryTimer = setTimeout(renderChart, 250);
		});

		onBeforeUnmount(() => {
			unmounted = true;
			if (retryTimer) clearTimeout(retryTimer);
			resizeObserver.value?.disconnect();
			mutationObserver.value?.disconnect();
			window.removeEventListener('resize', handleResize);
			cleanupChart();
		});

		watch(
			() => [props.data, props.variableStates, props.config.variables],
			() => renderChart(),
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
