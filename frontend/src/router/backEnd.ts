import { RouteRecordRaw } from 'vue-router';
import { storeToRefs } from 'pinia';
import pinia from '/@/stores/index';
import { useUserInfo } from '/@/stores/userInfo';
import { useRequestOldRoutes } from '/@/stores/requestOldRoutes';
import { Session } from '/@/utils/storage';
import { NextLoading } from '/@/utils/loading';
import { dynamicRoutes, notFoundAndNoPower } from '/@/router/route';
import { formatTwoStageRoutes, formatFlatteningRoutes, router } from '/@/router/index';
import { useRoutesList } from '/@/stores/routesList';
import { useTagsViewRoutes } from '/@/stores/tagsViewRoutes';
import { useMenuApi } from '/@/api/menu/index';

const menuApi = useMenuApi();

/**
 * Load view modules that can be resolved from backend route definitions.
 */
const layoutModules: any = import.meta.glob('../layout/routerView/*.{vue,tsx}');
const viewsModules: any = import.meta.glob('../views/**/*.{vue,tsx}');
const dynamicViewsModules: Record<string, Function> = Object.assign({}, { ...layoutModules }, { ...viewsModules });

/**
 * Initialize backend-driven routes after a page refresh.
 */
export async function initBackEndControlRoutes() {
	if (window.nextLoading === undefined) NextLoading.start();
	if (!Session.get('token')) return false;
	await useUserInfo().setUserInfos();
	const res = await getBackEndControlRoutes();
	if (res.data.length <= 0) return Promise.resolve(true);
	useRequestOldRoutes().setRequestOldRoutes(JSON.parse(JSON.stringify(res.data)));
	dynamicRoutes[0].children = await backEndComponent(res.data);
	await setAddRoute();
	setFilterMenuAndCacheTagsViewRoutes();
}

/**
 * Populate the sidebar route store and the flattened tags-view cache.
 */
export async function setFilterMenuAndCacheTagsViewRoutes() {
	const storesRoutesList = useRoutesList(pinia);
	storesRoutesList.setRoutesList(dynamicRoutes[0].children as any);
	setCacheTagsViewRoutes();
}

/**
 * Cache flattened routes for tags view and route search.
 */
export function setCacheTagsViewRoutes() {
	const storesTagsView = useTagsViewRoutes(pinia);
	storesTagsView.setTagsViewRoutes(formatTwoStageRoutes(formatFlatteningRoutes(dynamicRoutes))[0].children);
}

/**
 * Finalize dynamic routes and append fallback pages.
 */
export function setFilterRouteEnd() {
	let filterRouteEnd: any = formatTwoStageRoutes(formatFlatteningRoutes(dynamicRoutes));
	filterRouteEnd[0].children = [...filterRouteEnd[0].children, ...notFoundAndNoPower];
	return filterRouteEnd;
}

/**
 * Register the current dynamic routes with the router instance.
 */
export async function setAddRoute() {
	for (const route of setFilterRouteEnd() as RouteRecordRaw[]) {
		router.addRoute(route);
	}
}

/**
 * Fetch backend route data for the current role.
 */
export function getBackEndControlRoutes() {
	const stores = useUserInfo(pinia);
	const { userInfos } = storeToRefs(stores);
	const auth = userInfos.value.roles[0];
	if (auth === 'dayu') return menuApi.getAdminMenu();
	else return menuApi.getTestMenu();
}

/**
 * Refresh backend route data.
 */
export async function setBackEndControlRefreshRoutes() {
	await getBackEndControlRoutes();
}

/**
 * Resolve backend component names to lazy-loaded view modules.
 */
export function backEndComponent(routes: any) {
	if (!routes) return;
	return routes.map((item: any) => {
		if (item.component) item.component = dynamicImport(dynamicViewsModules, item.component as string);
		item.children && backEndComponent(item.children);
		return item;
	});
}

/**
 * Match a backend component path to a local module import.
 */
export function dynamicImport(dynamicViewsModules: Record<string, Function>, component: string) {
	const keys = Object.keys(dynamicViewsModules);
	const matchKeys = keys.filter((key) => {
		const k = key.replace(/..\/views|../, '');
		return k.startsWith(`${component}`) || k.startsWith(`/${component}`);
	});
	if (matchKeys?.length === 1) {
		const matchKey = matchKeys[0];
		return dynamicViewsModules[matchKey];
	}
	if (matchKeys?.length > 1) {
		return false;
	}
}
