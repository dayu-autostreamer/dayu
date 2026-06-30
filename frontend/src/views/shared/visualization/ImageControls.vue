<template>
	<SharedInfoControls :config="config" fallback="image" />
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
