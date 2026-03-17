<template>
	<div class="layout-parent">
		<router-view v-slot="{ Component }" :key="$route.fullPath">
			<transition :name="setTransitionName" mode="out-in">
				<keep-alive :include="getKeepAliveNames">
					<component :is="Component" :key="state.refreshRouterViewKey" class="w100" v-show="!isIframePage" />
				</keep-alive>
			</transition>
		</router-view>
		<transition :name="setTransitionName" mode="out-in">
			<Iframes class="w100" v-show="isIframePage" :refreshKey="state.iframeRefreshKey" :name="setTransitionName" :list="state.iframeList" />
		</transition>
	</div>
</template>

<script setup lang="ts" name="layoutParentView">
import { defineAsyncComponent, computed, reactive, onBeforeMount, onUnmounted, nextTick, watch, onMounted } from 'vue';
import { useRoute, useRouter } from 'vue-router';
import { storeToRefs } from 'pinia';
import { useKeepALiveNames } from '/@/stores/keepAliveNames';
import { useThemeConfig } from '/@/stores/themeConfig';
import { Session } from '/@/utils/storage';
import mittBus from '/@/utils/mitt';

const Iframes = defineAsyncComponent(() => import('/@/layout/routerView/iframes.vue'));

const route = useRoute();
const router = useRouter();
const storesKeepAliveNames = useKeepALiveNames();
const storesThemeConfig = useThemeConfig();
const { keepAliveNames, cachedViews } = storeToRefs(storesKeepAliveNames);
const { themeConfig } = storeToRefs(storesThemeConfig);
const state = reactive<ParentViewState>({
	refreshRouterViewKey: '',
	iframeRefreshKey: '',
	keepAliveNameList: [],
	iframeList: [],
});

const setTransitionName = computed(() => {
	return themeConfig.value.animation;
});

const getKeepAliveNames = computed(() => {
	return themeConfig.value.isTagsview ? cachedViews.value : state.keepAliveNameList;
});

const isIframePage = computed(() => {
	return route.meta.isIframe;
});

const getIframeListRoutes = () => {
	router.getRoutes().forEach((v) => {
		if (v.meta.isIframe) {
			v.meta.isIframeOpen = false;
			v.meta.loading = true;
			state.iframeList.push({ ...v });
		}
	});
};

const handleTagsViewRefresh = (fullPath: string) => {
	state.keepAliveNameList = keepAliveNames.value.filter((name: string) => route.name !== name);
	state.refreshRouterViewKey = '';
	state.iframeRefreshKey = '';
	nextTick(() => {
		state.refreshRouterViewKey = fullPath;
		state.iframeRefreshKey = fullPath;
		state.keepAliveNameList = keepAliveNames.value;
	});
};

onBeforeMount(() => {
	state.keepAliveNameList = keepAliveNames.value;
	mittBus.on('onTagsViewRefreshRouterView', handleTagsViewRefresh);
});

onMounted(() => {
	getIframeListRoutes();
	nextTick(() => {
		setTimeout(() => {
			if (themeConfig.value.isCacheTagsView) {
				const tagsViewArr: RouteItem[] = Session.get('tagsViewList') || [];
				cachedViews.value = tagsViewArr.filter((item) => item.meta?.isKeepAlive).map((item) => item.name as string);
			}
		}, 0);
	});
});

onUnmounted(() => {
	mittBus.off('onTagsViewRefreshRouterView', handleTagsViewRefresh);
});

watch(
	() => route.fullPath,
	() => {
		state.refreshRouterViewKey = decodeURI(route.fullPath);
	},
	{
		immediate: true,
	}
);
</script>
