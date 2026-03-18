import { createRouter, createWebHashHistory } from 'vue-router';
import NProgress from 'nprogress';
import 'nprogress/nprogress.css';
import pinia from '/@/stores/index';
import { storeToRefs } from 'pinia';
import { useKeepALiveNames } from '/@/stores/keepAliveNames';
import { useRoutesList } from '/@/stores/routesList';
import { useThemeConfig } from '/@/stores/themeConfig';
import { Session } from '/@/utils/storage';
import { staticRoutes, notFoundAndNoPower } from '/@/router/route';
import { initFrontEndControlRoutes } from '/@/router/frontEnd';
import { initBackEndControlRoutes } from '/@/router/backEnd';

/**
 * The app supports both frontend-defined routes and backend-provided routes.
 * Toggle the mode through `themeConfig.isRequestRoutes`.
 */

const storesThemeConfig = useThemeConfig(pinia);
const { themeConfig } = storeToRefs(storesThemeConfig);
const { isRequestRoutes } = themeConfig.value;

/**
 * Create the shared router instance.
 */
export const router = createRouter({
	history: createWebHashHistory(),
	routes: [...notFoundAndNoPower, ...staticRoutes],
});

/**
 * Flatten a nested route tree into a single list.
 */
export function formatFlatteningRoutes(arr: any) {
	if (arr.length <= 0) return false;
	for (let i = 0; i < arr.length; i++) {
		if (arr[i].children) {
			arr = arr.slice(0, i + 1).concat(arr[i].children, arr.slice(i + 1));
		}
	}
	return arr;
}

/**
 * Rebuild a two-level route tree and collect keep-alive names.
 */
export function formatTwoStageRoutes(arr: any) {
	if (arr.length <= 0) return false;
	const newArr: any = [];
	const cacheList: Array<string> = [];
	arr.forEach((v: any) => {
		if (v.path === '/') {
			newArr.push({
				component: v.component,
				name: v.name,
				path: v.path,
				redirect: v.redirect,
				meta: v.meta,
				children: [],
			});
		} else {
			if (v.path.indexOf('/:') > -1) {
				v.meta['isDynamic'] = true;
				v.meta['isDynamicPath'] = v.path;
			}
			newArr[0].children.push({ ...v });
			if (newArr[0].meta.isKeepAlive && v.meta.isKeepAlive) {
				cacheList.push(v.name);
				const stores = useKeepALiveNames(pinia);
				stores.setCacheKeepAlive(cacheList);
			}
		}
	});
	return newArr;
}

router.beforeEach(async (to, from, next) => {
	NProgress.configure({ showSpinner: false });
	if (to.meta.title) NProgress.start();
	const token = Session.get('token');
	if (to.path === '/login' && !token) {
		next();
		NProgress.done();
	} else {
		if (!token) {
			next(`/login?redirect=${to.path}&params=${JSON.stringify(to.query ? to.query : to.params)}`);
			Session.clear();
			NProgress.done();
		} else if (token && to.path === '/login') {
			next('/home');
			NProgress.done();
		} else {
			const storesRoutesList = useRoutesList(pinia);
			const { routesList } = storeToRefs(storesRoutesList);
			if (routesList.value.length === 0) {
				if (isRequestRoutes) {
					await initBackEndControlRoutes();
					// Rehydrate routes after refresh before retrying navigation.
					next({ path: to.path, query: to.query });
				} else {
					await initFrontEndControlRoutes();
					next({ path: to.path, query: to.query });
				}
			} else {
				next();
			}
		}
	}
});

router.afterEach(() => {
	NProgress.done();
});

export default router;
