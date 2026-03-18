<template>
	<div class="h100" v-show="!isTagsViewCurrenFull">
		<el-aside class="layout-aside" :class="setCollapseStyle">
			<Logo v-if="setShowLogo" />
			<el-scrollbar
				class="flex-auto"
				ref="layoutAsideScrollbarRef"
				@mouseenter="onAsideEnterLeave(true)"
				@mouseleave="onAsideEnterLeave(false)"
			>
				<Vertical :menuList="state.menuList" />
			</el-scrollbar>
		</el-aside>
	</div>
</template>

<script setup lang="ts" name="layoutAside">
import { defineAsyncComponent, reactive, computed, watch, onBeforeMount, onUnmounted, ref } from 'vue';
import { storeToRefs } from 'pinia';
import { useRoutesList } from '/@/stores/routesList';
import { useThemeConfig } from '/@/stores/themeConfig';
import { useTagsViewRoutes } from '/@/stores/tagsViewRoutes';
import mittBus from '/@/utils/mitt';

const Logo = defineAsyncComponent(() => import('/@/layout/logo/index.vue'));
const Vertical = defineAsyncComponent(() => import('/@/layout/navMenu/vertical.vue'));

const layoutAsideScrollbarRef = ref();
const stores = useRoutesList();
const storesThemeConfig = useThemeConfig();
const storesTagsViewRoutes = useTagsViewRoutes();
const { routesList } = storeToRefs(stores);
const { themeConfig } = storeToRefs(storesThemeConfig);
const { isTagsViewCurrenFull } = storeToRefs(storesTagsViewRoutes);
const state = reactive<AsideState>({
	menuList: [],
	clientWidth: 0,
});

const ensureLayoutAsideMobileMode = () => {
	const asideElement = document.querySelector('.layout-container') as HTMLElement | null;
	if (!asideElement || document.querySelector('.layout-aside-mobile-mode')) return;

	const maskElement = document.createElement('div');
	maskElement.setAttribute('class', 'layout-aside-mobile-mode');
	maskElement.addEventListener('click', closeLayoutAsideMobileMode);
	asideElement.appendChild(maskElement);
};

const setCollapseStyle = computed(() => {
	const { layout, isCollapse, menuBar } = themeConfig.value;
	const asideBrTheme = ['#FFFFFF', '#FFF', '#fff', '#ffffff'];
	const asideBrColor = asideBrTheme.includes(menuBar) ? 'layout-el-aside-br-color' : '';

	if (state.clientWidth <= 1000) {
		if (isCollapse) {
			document.body.setAttribute('class', 'el-popup-parent--hidden');
			ensureLayoutAsideMobileMode();
			return [asideBrColor, 'layout-aside-mobile', 'layout-aside-mobile-open'];
		}

		closeLayoutAsideMobileMode();
		return [asideBrColor, 'layout-aside-mobile', 'layout-aside-mobile-close'];
	} else {
		if (layout === 'columns' || layout === 'classic') {
			if (isCollapse) return [asideBrColor, 'layout-aside-pc-1'];
			return [asideBrColor, 'layout-aside-pc-220'];
		}

		if (isCollapse) return [asideBrColor, 'layout-aside-pc-64'];
		return [asideBrColor, 'layout-aside-pc-220'];
	}
});

const setShowLogo = computed(() => {
	const { layout, isShowLogo } = themeConfig.value;
	return (isShowLogo && layout === 'defaults') || (isShowLogo && layout === 'columns');
});

const closeLayoutAsideMobileMode = () => {
	const el = document.querySelector('.layout-aside-mobile-mode');
	el?.setAttribute('style', 'animation: error-img-two 0.3s');
	setTimeout(() => {
		el?.parentNode?.removeChild(el);
	}, 300);
	const clientWidth = document.body.clientWidth;
	if (clientWidth < 1000) themeConfig.value.isCollapse = false;
	document.body.setAttribute('class', '');
};

const setFilterRoutes = () => {
	if (themeConfig.value.layout === 'columns') return false;
	state.menuList = filterRoutesFun(routesList.value);
};

const filterRoutesFun = <T extends RouteItem>(arr: T[]): T[] => {
	return arr
		.filter((item: T) => !item.meta?.isHide)
		.map((item: T) => {
			item = Object.assign({}, item);
			if (item.children) item.children = filterRoutesFun(item.children);
			return item;
		});
};

const initMenuFixed = (clientWidth: number) => {
	state.clientWidth = clientWidth;
};

const onAsideEnterLeave = (isEntering: boolean) => {
	const { layout } = themeConfig.value;
	if (layout !== 'columns') return false;
	if (!isEntering) mittBus.emit('restoreDefault');
	if (themeConfig.value.isColumnsMenuHoverPreload) stores.setColumnsMenuHover(isEntering);
};

const handleSetSendColumnsChildren = (res: MittMenu) => {
	state.menuList = res.children;
};

const handleSetSendClassicChildren = (res: MittMenu) => {
	const { layout, isClassicSplitMenu } = themeConfig.value;
	if (layout === 'classic' && isClassicSplitMenu) {
		themeConfig.value.isCollapse = res.children.length <= 1;
		state.menuList = [];
		state.menuList = res.children;
	}
};

const handleFilterRoutesRefresh = () => {
	setFilterRoutes();
};

const handleLayoutMobileResize = (res: LayoutMobileResize) => {
	initMenuFixed(res.clientWidth);
	closeLayoutAsideMobileMode();
};

onBeforeMount(() => {
	initMenuFixed(document.body.clientWidth);
	setFilterRoutes();
	mittBus.on('setSendColumnsChildren', handleSetSendColumnsChildren);
	mittBus.on('setSendClassicChildren', handleSetSendClassicChildren);
	mittBus.on('getBreadcrumbIndexSetFilterRoutes', handleFilterRoutesRefresh);
	mittBus.on('layoutMobileResize', handleLayoutMobileResize);
});

onUnmounted(() => {
	mittBus.off('setSendColumnsChildren', handleSetSendColumnsChildren);
	mittBus.off('setSendClassicChildren', handleSetSendClassicChildren);
	mittBus.off('getBreadcrumbIndexSetFilterRoutes', handleFilterRoutesRefresh);
	mittBus.off('layoutMobileResize', handleLayoutMobileResize);
});

watch(
	() => [
		themeConfig.value.isShowLogoChange,
		themeConfig.value.isShowLogo,
		themeConfig.value.layout,
		themeConfig.value.isClassicSplitMenu,
	],
	([isShowLogoChange, isShowLogo, layout, isClassicSplitMenu]) => {
		if (isShowLogoChange !== isShowLogo) {
			if (layoutAsideScrollbarRef.value) layoutAsideScrollbarRef.value.update();
		}
		if (layout === 'classic' && isClassicSplitMenu) return false;
	}
);

watch(
	() => routesList.value,
	() => {
		setFilterRoutes();
	}
);
</script>
