import { defineAsyncComponent, nextTick } from 'vue';
import type { App } from 'vue';
import * as svg from '@element-plus/icons-vue';
import { storeToRefs } from 'pinia';
import { i18n } from '/@/i18n/index';
import router from '/@/router/index';
import pinia from '/@/stores/index';
import { useThemeConfig } from '/@/stores/themeConfig';
import { Local } from '/@/utils/storage';
import { verifyUrl } from '/@/utils/toolsValidate';

const SvgIcon = defineAsyncComponent(() => import('/@/components/svgIcon/index.vue'));

export function elSvg(app: App) {
	Object.values(svg).forEach((icon) => {
		app.component(`ele-${icon.name}`, icon);
	});
	app.component('SvgIcon', SvgIcon);
}

export function useTitle() {
	const stores = useThemeConfig(pinia);
	const { themeConfig } = storeToRefs(stores);

	nextTick(() => {
		const globalTitle = themeConfig.value.globalTitle;
		const { path, meta } = router.currentRoute.value;
		const pageTitle = path === '/login' ? (meta.title as string) : setTagsViewNameI18n(router.currentRoute.value);
		document.title = pageTitle ? `${pageTitle} - ${globalTitle}` : globalTitle;
	});
}

export function setTagsViewNameI18n(item: RouteToFrom) {
	const { query, params, meta } = item;
	const pattern = /^\{("(zh-cn|en|zh-tw)":"[^,]+",?){1,3}}$/;
	const tagsViewName = query?.tagsViewName || params?.tagsViewName;

	if (tagsViewName && pattern.test(tagsViewName as string)) {
		const localizedName = JSON.parse(tagsViewName as string);
		return localizedName[i18n.global.locale.value];
	}

	if (tagsViewName) {
		return tagsViewName;
	}

	return i18n.global.t(meta?.title as string);
}

export const globalComponentSize = (): string => {
	const stores = useThemeConfig(pinia);
	const { themeConfig } = storeToRefs(stores);
	return Local.get('themeConfig')?.globalComponentSize || themeConfig.value.globalComponentSize;
};

export function isMobile(): boolean {
	return /phone|pad|pod|iphone|ipod|ios|ipad|android|mobile|blackberry|iemobile|mqqbrowser|juc|fennec|wosbrowser|browserng|webos|symbian|windows phone/i.test(
		navigator.userAgent
	);
}

export function handleOpenLink(val: RouteItem) {
	const { origin, pathname } = window.location;
	router.push(val.path);

	if (verifyUrl(val.meta?.isLink as string)) {
		window.open(val.meta?.isLink);
		return;
	}

	window.open(`${origin}${pathname}#${val.meta?.isLink}`);
}

const other = {
	elSvg,
	useTitle,
	setTagsViewNameI18n,
	globalComponentSize,
	isMobile,
	handleOpenLink,
};

export default other;
