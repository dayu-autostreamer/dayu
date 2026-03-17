<template>
	<div class="layout-navbars-container">
		<BreadcrumbIndex />
		<TagsView v-if="setShowTagsView" />
	</div>
</template>

<script setup lang="ts" name="layoutNavBars">
import { defineAsyncComponent, computed } from 'vue';
import { storeToRefs } from 'pinia';
import { useThemeConfig } from '/@/stores/themeConfig';

const BreadcrumbIndex = defineAsyncComponent(() => import('/@/layout/navBars/topBar/index.vue'));
const TagsView = defineAsyncComponent(() => import('/@/layout/navBars/tagsView/tagsView.vue'));

const storesThemeConfig = useThemeConfig();
const { themeConfig } = storeToRefs(storesThemeConfig);

const setShowTagsView = computed(() => {
	let { layout, isTagsview } = themeConfig.value;
	return layout !== 'classic' && isTagsview;
});
</script>

<style scoped lang="scss">
.layout-navbars-container {
	display: flex;
	flex-direction: column;
	width: 100%;
	height: 100%;
}
</style>
