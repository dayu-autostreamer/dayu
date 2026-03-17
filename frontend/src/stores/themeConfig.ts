import { defineStore } from 'pinia';

/**
 * Default layout and visual preferences.
 * Persisted values are stored in local storage and can be restored from the settings drawer.
 */
export const defaultThemeConfig = {
	ver: 1,
	isDrawer: false,

	primary: '#409eff',
	isIsDark: false,

	topBar: '#ffffff',
	topBarColor: '#606266',
	isTopBarColorGradual: false,

	menuBar: '#545c64',
	menuBarColor: '#eaeaea',
	menuBarActiveColor: 'rgba(0, 0, 0, 0.2)',
	isMenuBarColorGradual: false,

	columnsMenuBar: '#545c64',
	columnsMenuBarColor: '#e6e6e6',
	isColumnsMenuBarColorGradual: false,
	isColumnsMenuHoverPreload: false,

	isCollapse: false,
	isUniqueOpened: true,
	isFixedHeader: false,
	isFixedHeaderChange: false,
	isClassicSplitMenu: false,
	isLockScreen: false,
	lockScreenTime: 30,

	isShowLogo: false,
	isShowLogoChange: false,
	isBreadcrumb: true,
	isTagsview: true,
	isBreadcrumbIcon: false,
	isTagsviewIcon: false,
	isCacheTagsView: false,
	isSortableTagsView: true,
	isShareTagsView: false,
	isFooter: false,
	isGrayscale: false,
	isInvert: false,

	tagsStyle: 'tags-style-five',
	animation: 'slide-right',
	columnsAsideStyle: 'columns-round',
	columnsAsideLayout: 'columns-vertical',

	layout: 'defaults',
	isRequestRoutes: false,

	globalTitle: 'DAYU',
	globalI18n: 'en',
	globalComponentSize: 'large',
};

export const useThemeConfig = defineStore('themeConfig', {
	state: (): ThemeConfigState => ({
		themeConfig: { ...defaultThemeConfig },
	}),
	actions: {
		setThemeConfig(data: ThemeConfigState) {
			this.themeConfig = data.themeConfig;
		},
	},
});
