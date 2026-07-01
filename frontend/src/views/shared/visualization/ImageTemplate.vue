<template>
	<div class="image-surface">
		<div v-if="isLoading" class="image-state-overlay">
			<el-icon class="image-state-icon is-loading" :size="28">
				<Loading />
			</el-icon>
		</div>

		<div v-else-if="loadError" class="image-state-overlay">
			<el-icon class="image-state-icon is-error" :size="34">
				<Warning />
			</el-icon>
			<p>Image load failed</p>
		</div>

		<el-tooltip v-else-if="currentImage" effect="dark" placement="top" :hide-after="0">
			<template #content>
				<div class="image-tooltip">
					<div v-for="row in tooltipRows" :key="row.label" class="image-tooltip__row">
						<span>{{ row.label }}:</span>
						<strong>{{ row.value }}</strong>
					</div>
				</div>
			</template>

			<div class="image-frame">
				<img :src="currentImage" :alt="config.name" class="responsive-image" @error="loadError = true" />
			</div>
		</el-tooltip>

		<div v-else class="image-state-overlay">
			<el-icon class="image-state-icon" :size="34">
				<Picture />
			</el-icon>
			<p>No data available</p>
		</div>
	</div>
</template>

<script>
import { computed, ref, watch } from 'vue';
import { Picture, Warning, Loading } from '@element-plus/icons-vue';

const BASE64_REGEX = /^data:image\/(\w+);base64,/;

export default {
	components: { Picture, Warning, Loading },
	props: {
		config: {
			type: Object,
			required: true,
		},
		data: {
			type: Array,
			default: () => [],
		},
	},
	setup(props) {
		const currentImage = ref(null);
		const currentImageMeta = ref(null);
		const isLoading = ref(false);
		const loadError = ref(false);

		const processBase64 = (input) => {
			if (BASE64_REGEX.test(input)) return input;
			return `data:image/png;base64,${input}`;
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

		const getFirstNonEmptyEntry = (obj, variables) => {
			if (!obj || typeof obj !== 'object' || !Array.isArray(variables)) return null;
			for (const key of variables) {
				const value = obj[key];
				if (value !== null && value !== undefined && value !== '') {
					return { key, value };
				}
			}
			return null;
		};

		const formatImageDataSize = (value) => {
			const rawLength = String(value || '').replace(BASE64_REGEX, '').length;
			const bytes = Math.round((rawLength * 3) / 4);
			if (bytes < 1024) return `${bytes} B`;
			if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
			return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
		};

		const tooltipRows = computed(() => {
			if (!currentImageMeta.value) return [];
			const rows = [];
			if (currentImageMeta.value.taskId) {
				rows.push({ label: 'Task', value: currentImageMeta.value.taskId });
			}
			rows.push({ label: 'Variable', value: currentImageMeta.value.variable });
			rows.push({ label: 'Image Data', value: formatImageDataSize(currentImageMeta.value.value) });
			return rows;
		});

		watch(
			() => [props.data, props.config?.variables],
			([newData, variables]) => {
				loadError.value = false;
				if (!Array.isArray(variables)) {
					currentImage.value = null;
					currentImageMeta.value = null;
					return;
				}

				const validItems = (newData || []).filter((item) => getFirstNonEmptyValue(item, variables) !== null);
				if (!validItems.length) {
					currentImage.value = null;
					currentImageMeta.value = null;
					return;
				}

				const latestItem = validItems
					.slice()
					.reverse()
					.find((item) => getFirstNonEmptyValue(item, variables) !== null);

				try {
					isLoading.value = true;
					const imageEntry = getFirstNonEmptyEntry(latestItem, variables);
					currentImage.value = processBase64(imageEntry.value);
					currentImageMeta.value = {
						taskId: latestItem.taskId || latestItem.timestamp || '',
						variable: imageEntry.key,
						value: imageEntry.value,
					};
				} catch (error) {
					console.error('Image process error:', error);
					loadError.value = true;
					currentImage.value = null;
					currentImageMeta.value = null;
				} finally {
					isLoading.value = false;
				}
			},
			{ deep: true, immediate: true }
		);

		return {
			currentImage,
			tooltipRows,
			isLoading,
			loadError,
		};
	},
};
</script>

<style scoped lang="scss">
.image-surface {
	position: relative;
	width: 100%;
	height: 100%;
	min-height: 320px;
	display: grid;
	place-items: center;
	border-radius: 18px;
	background: radial-gradient(circle at top left, rgba(37, 99, 235, 0.08), transparent 26%),
		linear-gradient(180deg, rgba(248, 250, 252, 0.96), rgba(255, 255, 255, 0.92)), #ffffff;
	overflow: hidden;
}

.image-frame {
	width: calc(100% - 24px);
	height: calc(100% - 24px);
	display: flex;
	align-items: center;
	justify-content: center;
	border-radius: 16px;
	background: linear-gradient(45deg, rgba(226, 232, 240, 0.6) 25%, transparent 25%),
		linear-gradient(-45deg, rgba(226, 232, 240, 0.6) 25%, transparent 25%),
		linear-gradient(45deg, transparent 75%, rgba(226, 232, 240, 0.6) 75%),
		linear-gradient(-45deg, transparent 75%, rgba(226, 232, 240, 0.6) 75%);
	background-size: 24px 24px;
	background-position: 0 0, 0 12px, 12px -12px, -12px 0;
}

.image-tooltip {
	display: grid;
	gap: 6px;
	min-width: 180px;
}

.image-tooltip__row {
	display: flex;
	justify-content: space-between;
	gap: 12px;
}

.image-tooltip__row span {
	color: #cbd5e1;
}

.responsive-image {
	max-width: 100%;
	max-height: 100%;
	object-fit: contain;
	border-radius: 12px;
	box-shadow: 0 18px 40px rgba(15, 23, 42, 0.12);
}

.image-state-overlay {
	position: absolute;
	inset: 0;
	display: grid;
	place-items: center;
	align-content: center;
	gap: 10px;
	text-align: center;
	color: #64748b;
	background: rgba(255, 255, 255, 0.76);
}

.image-state-icon {
	color: #94a3b8;
}

.image-state-icon.is-loading {
	animation: rotate 1.8s linear infinite;
}

.image-state-icon.is-error {
	color: #ef4444;
}

.image-state-overlay p {
	margin: 0;
	font-size: 14px;
}

@keyframes rotate {
	from {
		transform: rotate(0deg);
	}

	to {
		transform: rotate(360deg);
	}
}
</style>
