<template>
	<SharedVariableControls :config="config" :variable-states="variableStates" label="services" @update:variable-states="$emit('update:variable-states', $event)" />
</template>

<script>
import SharedVariableControls from './SharedVariableControls.vue';

export default {
	components: { SharedVariableControls },
	props: {
		config: {
			type: Object,
			required: true,
		},
		variableStates: {
			type: Object,
			default: () => ({}),
		},
	},
	emits: ['update:variable-states'],
};
</script>
