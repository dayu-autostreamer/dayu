import { RouteRecordRaw } from 'vue-router';
import { storeToRefs } from 'pinia';
import { formatTwoStageRoutes, formatFlatteningRoutes, router } from '/@/router/index';
import { dynamicRoutes, notFoundAndNoPower } from '/@/router/route';
import pinia from '/@/stores/index';
import { Session } from '/@/utils/storage';
import { useUserInfo } from '/@/stores/userInfo';
import { useTagsViewRoutes } from '/@/stores/tagsViewRoutes';
import { useRoutesList } from '/@/stores/routesList';
import { NextLoading } from '/@/utils/loading';

/**
 * Initialize frontend-defined routes after a page refresh.
 */
export async function initFrontEndControlRoutes() {
	if (window.nextLoading === undefined) NextLoading.start();
	if (!Session.get('token')) return false;
	await useUserInfo(pinia).setUserInfos();
	if (useUserInfo().userInfos.roles.length <= 0) return Promise.resolve(true);
	await setAddRoute();
	setFilterMenuAndCacheTagsViewRoutes();
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
 * Remove previously registered dynamic routes.
 */
export async function frontEndsResetRoute() {
	for (const route of setFilterRouteEnd() as RouteRecordRaw[]) {
		const routeName: any = route.name;
		router.hasRoute(routeName) && router.removeRoute(routeName);
	}
}

/**
 * Filter dynamic routes for the current user and append fallback pages.
 */
export function setFilterRouteEnd() {
	let filterRouteEnd: any = formatTwoStageRoutes(formatFlatteningRoutes(dynamicRoutes));
	filterRouteEnd[0].children = [...setFilterRoute(filterRouteEnd[0].children), ...notFoundAndNoPower];
	return filterRouteEnd;
}

/**
 * Filter the flat route list by role before registering routes.
 */
export function setFilterRoute(chil: any) {
	const stores = useUserInfo(pinia);
	const { userInfos } = storeToRefs(stores);
	let filterRoute: any = [];
	chil.forEach((route: any) => {
		if (route.meta.roles) {
			route.meta.roles.forEach((metaRoles: any) => {
				userInfos.value.roles.forEach((roles: any) => {
					if (metaRoles === roles) filterRoute.push({ ...route });
				});
			});
		}
	});
	return filterRoute;
}

/**
 * Cache flattened routes for tags view and route search.
 */
export function setCacheTagsViewRoutes() {
	const stores = useUserInfo(pinia);
	const storesTagsView = useTagsViewRoutes(pinia);
	const { userInfos } = storeToRefs(stores);
	let rolesRoutes = setFilterHasRolesMenu(dynamicRoutes, userInfos.value.roles);
	storesTagsView.setTagsViewRoutes(formatTwoStageRoutes(formatFlatteningRoutes(rolesRoutes))[0].children);
}

/**
 * Populate the sidebar route store and the flattened tags-view cache.
 */
export function setFilterMenuAndCacheTagsViewRoutes() {
	const stores = useUserInfo(pinia);
	const storesRoutesList = useRoutesList(pinia);
	const { userInfos } = storeToRefs(stores);
	storesRoutesList.setRoutesList(setFilterHasRolesMenu(dynamicRoutes[0].children, userInfos.value.roles));
	setCacheTagsViewRoutes();
}

/**
 * Check whether a route is visible to the current role set.
 */
export function hasRoles(roles: any, route: any) {
	if (route.meta && route.meta.roles) return roles.some((role: any) => route.meta.roles.includes(role));
	else return true;
}

/**
 * Recursively filter nested routes by role.
 */
export function setFilterHasRolesMenu(routes: any, roles: any) {
	const menu: any = [];
	routes.forEach((route: any) => {
		const item = { ...route };
		if (hasRoles(roles, item)) {
			if (item.children) item.children = setFilterHasRolesMenu(item.children, roles);
			menu.push(item);
		}
	});
	return menu;
}
