const dayuMenu = [
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
		component: 'dag/index',
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
		component: 'datasource/index',
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
		component: 'install/index',
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
		component: 'result/index',
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
		component: 'system/index',
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
];

/**
 * Local Dayu route data used when backend-controlled routes are enabled.
 */
export function useMenuApi() {
	const getDayuMenu = async () => ({
		data: JSON.parse(JSON.stringify(dayuMenu)),
	});

	return {
		getDayuMenu,
		getCommonMenu: getDayuMenu,
	};
}
