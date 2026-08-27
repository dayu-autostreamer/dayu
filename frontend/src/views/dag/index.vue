<template>
	<div class="home-container layout-pd">
		<el-row :gutter="15" class="home-card-two mb15">
			<el-col :xs="24">
				<div class="home-card-item">
					<div style="height: 100%">
						<div class="flex-margin flex w100">
							<div class="flex-auto">
								<DagManage />
							</div>
						</div>
					</div>
				</div>
			</el-col>
		</el-row>
	</div>
</template>

<script setup>
import DagManage from './DagManage.vue';
</script>

<style scoped lang="scss">
.home-container {
	overflow: hidden;

	.home-card-two {
		.home-card-item {
			width: 100%;
			height: 90vh;
			border-radius: 2px;
			transition: all ease 0.3s;
			padding: 5px;
			overflow: scroll;
			background: var(--el-color-white);
			color: var(--el-text-color-primary);
			border: 1px solid var(--next-border-color-light);

			&:hover {
				box-shadow: 0 2px 12px var(--next-color-dark-hover);
				transition: all ease 0.3s;
			}
		}
	}
}
</style>
