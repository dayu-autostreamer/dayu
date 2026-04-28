<template>
	<div class="install-page layout-pd">
		<div class="install-grid">
			<section class="workspace-panel workspace-panel--primary">
				<SvcInstall />
			</section>

			<section class="workspace-panel workspace-panel--secondary">
				<SvcQuery />
			</section>
		</div>
	</div>
</template>

<script>
import SvcInstall from './SvcInstall.vue';
import SvcQuery from './SvcQuery.vue';

export default {
	components: {
		SvcInstall,
		SvcQuery,
	},
};
</script>

<style scoped lang="scss">
.install-page {
	padding: 20px;
	background:
		radial-gradient(circle at top left, rgba(37, 99, 235, 0.08), transparent 24%),
		radial-gradient(circle at bottom right, rgba(14, 165, 233, 0.08), transparent 24%);
}

.install-grid {
	display: grid;
	grid-template-columns: minmax(0, 1.05fr) minmax(360px, 0.95fr);
	gap: 20px;
	align-items: start;
}

.workspace-panel {
	padding: 24px;
	border-radius: 28px;
	border: 1px solid #e2e8f0;
	background:
		linear-gradient(135deg, rgba(255, 255, 255, 0.96), rgba(248, 250, 252, 0.96)),
		#ffffff;
	box-shadow: 0 22px 48px rgba(15, 23, 42, 0.06);
}

.workspace-panel--secondary {
	position: sticky;
	top: 20px;
}

@media (max-width: 1200px) {
	.install-grid {
		grid-template-columns: 1fr;
	}

	.workspace-panel--secondary {
		position: static;
	}
}

@media (max-width: 768px) {
	.install-page {
		padding: 14px;
	}

	.workspace-panel {
		padding: 18px;
		border-radius: 22px;
	}
}
</style>
