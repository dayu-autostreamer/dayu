<template>
	<SharedInfoControls :config="config" fallback="topology" />
</template>

<script>
import SharedInfoControls from './SharedInfoControls.vue';

export default {
	components: { SharedInfoControls },
	props: {
		config: {
			type: Object,
			required: true,
		},
	},
};
</script>
