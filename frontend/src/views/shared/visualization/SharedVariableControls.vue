<template>
	<div class="controls-shell">
		<span class="controls-label">{{ label }}</span>
		<el-checkbox-group v-model="selectedVariables" class="controls-options" @change="handleVariableChange">
			<el-checkbox-button v-for="varName in config.variables" :key="varName" :label="varName" class="controls-option">
				{{ varName }}
			</el-checkbox-button>
		</el-checkbox-group>
	</div>
</template>

<script>
import { ref, watch, toRefs } from 'vue';

export default {
	props: {
		config: {
			type: Object,
			required: true,
		},
		variableStates: {
			type: Object,
			default: () => ({}),
		},
		label: {
			type: String,
			default: 'Variables',
		},
	},
	emits: ['update:variable-states'],

	setup(props, { emit }) {
		const { variableStates } = toRefs(props);
		const selectedVariables = ref([]);

		const syncSelectedVariables = (states) => {
			selectedVariables.value = Object.keys(states || {}).filter((key) => states[key]);
		};

		const handleVariableChange = () => {
			const newStates = {};
			(props.config.variables || []).forEach((varName) => {
				newStates[varName] = selectedVariables.value.includes(varName);
			});
			emit('update:variable-states', newStates);
		};

		syncSelectedVariables(variableStates.value);

		watch(
			() => props.variableStates,
			(newValue) => {
				syncSelectedVariables(newValue);
			},
			{ deep: true }
		);

		return {
			selectedVariables,
			handleVariableChange,
		};
	},
};
</script>

<style scoped lang="scss">
.controls-shell {
	display: flex;
	flex-wrap: wrap;
	align-items: flex-start;
	justify-content: flex-start;
	gap: 10px;
	padding-top: 2px;
	max-height: 64px;
	overflow-y: auto;
	overflow-x: hidden;
	scrollbar-width: thin;
}

.controls-label {
	font-size: 11px;
	font-weight: 700;
	color: #64748b;
}

.controls-options {
	display: flex;
	flex-wrap: wrap;
	gap: 8px;
	justify-content: flex-start;
	flex: 1 1 220px;
	min-width: 0;
}

.controls-option {
	margin: 0;
}

.controls-option :deep(.el-checkbox-button__inner) {
	max-width: 160px;
	border-radius: 999px;
	border: 1px solid #cbd5e1;
	background: #f8fafc;
	color: #334155;
	box-shadow: none;
	padding: 7px 12px;
	font-size: 12px;
	line-height: 1.1;
	overflow: hidden;
	text-overflow: ellipsis;
	white-space: nowrap;
}

.controls-option.is-checked :deep(.el-checkbox-button__inner) {
	border-color: #2563eb;
	background: rgba(37, 99, 235, 0.12);
	color: #1d4ed8;
}
</style>
