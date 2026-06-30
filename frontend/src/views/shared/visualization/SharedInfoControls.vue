<template>
	<div class="info-shell">
		<el-tag type="info" size="small" effect="plain" class="info-tag">
			{{ displayValue }}
		</el-tag>
	</div>
</template>

<script>
export default {
	props: {
		config: {
			type: Object,
			required: true,
		},
		fallback: {
			type: String,
			default: 'module',
		},
	},
	computed: {
		displayValue() {
			return this.config?.variables?.[0] || this.fallback;
		},
	},
};
</script>

<style scoped lang="scss">
.info-shell {
	padding-top: 2px;
	text-align: center;
}

.info-tag {
	max-width: min(100%, 320px);
	overflow: hidden;
	text-overflow: ellipsis;
	border-radius: 999px;
	padding-inline: 10px;
}
</style>
