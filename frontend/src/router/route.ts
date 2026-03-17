import { RouteRecordRaw } from 'vue-router';

/**
 * Keep route paths aligned with directory names to make view lookup predictable.
 *
 * Route meta fields:
 * meta: {
 *      title:          Menu, tags-view, and search label key
 *      isLink:         External link target when paired with `isIframe: false`
 *      isHide:         Hide the route from navigation
 *      isKeepAlive:    Cache the page component
 *      isAffix:        Pin the route in tags view
 *      isIframe:       Render the route inside an iframe
 *      roles:          Allowed roles for the route
 *      icon:           Menu and tags-view icon class
 * }
 */

declare module 'vue-router' {
	interface RouteMeta {
		title?: string;
		isLink?: string;
		isHide?: boolean;
		isKeepAlive?: boolean;
		isAffix?: boolean;
		isIframe?: boolean;
		roles?: string[];
		icon?: string;
	}
}

/**
 * Frontend-defined dynamic routes.
 * Add new application pages under the top-level `children` array.
 */
export const dynamicRoutes: Array<RouteRecordRaw> = [
	{
		path: '/',
		name: '/',
		component: () => import('/@/layout/index.vue'),
		redirect: '/home',
		meta: {
			isKeepAlive: true,
		},
		children: [
			{
				path: '/home',
				name: 'home',
				meta: {
					title: 'message.router.home',
					isLink: 'https://dayu-autostreamer.github.io/',
					isHide: false,
					isIframe: true,
					roles: ['dayu', 'common'],
					icon: 'iconfont icon-shouye',
				},
			},
			{
				path: '/dag',
				name: 'dag',
				component: () => import('/@/views/dag/index.vue'),
				meta: {
					title: 'message.router.dag',
					isLink: '',
					isHide: false,
					isKeepAlive: true,
					isAffix: false,
					isIframe: false,
					roles: ['dayu', 'common'],
					icon: 'iconfont icon-zidingyibuju',
				},
			},
			{
				path: '/datasource',
				name: 'datasource',
				component: () => import('/@/views/datasource/index.vue'),
				meta: {
					title: 'message.router.datasource',
					isLink: '',
					isHide: false,
					isKeepAlive: true,
					isAffix: false,
					isIframe: false,
					roles: ['dayu', 'common'],
					icon: 'iconfont icon-zhongduancanshu',
				},
			},
			{
				path: '/install',
				name: 'install',
				component: () => import('/@/views/install/index.vue'),
				meta: {
					title: 'message.router.install',
					isLink: '',
					isHide: false,
					isKeepAlive: true,
					isAffix: false,
					isIframe: false,
					roles: ['dayu', 'common'],
					icon: 'iconfont icon-xingqiu',
				},
			},
			{
				path: '/result',
				name: 'result',
				component: () => import('/@/views/result/index.vue'),
				meta: {
					title: 'message.router.result',
					isLink: '',
					isHide: false,
					isKeepAlive: true,
					isAffix: false,
					isIframe: false,
					roles: ['dayu', 'common'],
					icon: 'iconfont icon-shuju',
				},
			},
			{
				path: '/system',
				name: 'system',
				component: () => import('/@/views/system/index.vue'),
				meta: {
					title: 'message.router.system',
					isLink: '',
					isHide: false,
					isKeepAlive: true,
					isAffix: false,
					isIframe: false,
					roles: ['dayu', 'common'],
					icon: 'iconfont icon-ico_shuju',
				},
			},
		],
	},
];

/**
 * Shared fallback pages.
 */
export const notFoundAndNoPower = [
	{
		path: '/:path(.*)*',
		name: 'notFound',
		component: () => import('/@/views/error/404.vue'),
		meta: {
			title: 'message.staticRoutes.notFound',
			isHide: true,
		},
	},
	{
		path: '/401',
		name: 'noPower',
		component: () => import('/@/views/error/401.vue'),
		meta: {
			title: 'message.staticRoutes.noPower',
			isHide: true,
		},
	},
];

/**
 * Static routes that are always present.
 */
export const staticRoutes: Array<RouteRecordRaw> = [
	{
		path: '/login',
		name: 'login',
		component: () => import('/@/views/login/index.vue'),
		meta: {
			title: 'Login',
		},
	},
];
