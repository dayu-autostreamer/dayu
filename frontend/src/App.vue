<template>
	<el-config-provider :size="getGlobalComponentSize" :locale="getGlobalI18n">
		<router-view v-show="setLockScreen" />
		<LockScreen v-if="themeConfig.isLockScreen" />
		<Setings ref="setingsRef" v-show="setLockScreen" />
		<CloseFull v-if="!themeConfig.isLockScreen" />
	</el-config-provider>
</template>

<script setup lang="ts" name="app">
import { defineAsyncComponent, computed, ref, onBeforeMount, onMounted, onUnmounted, nextTick, watch } from 'vue';
import { useRoute } from 'vue-router';
import { useI18n } from 'vue-i18n';
import { storeToRefs } from 'pinia';
import { useTagsViewRoutes } from '/@/stores/tagsViewRoutes';
import { useThemeConfig } from '/@/stores/themeConfig';
import other from '/@/utils/other';
import { Local, Session } from '/@/utils/storage';
import mittBus from '/@/utils/mitt';
import setIntroduction from '/@/utils/setIconfont';

const LockScreen = defineAsyncComponent(() => import('/@/layout/lockScreen/index.vue'));
const Setings = defineAsyncComponent(() => import('/@/layout/navBars/topBar/setings.vue'));
const CloseFull = defineAsyncComponent(() => import('/@/layout/navBars/topBar/closeFull.vue'));

const { messages, locale } = useI18n();
const setingsRef = ref<{ openDrawer: () => void } | null>(null);
const route = useRoute();
const stores = useTagsViewRoutes();
const storesThemeConfig = useThemeConfig();
const { themeConfig } = storeToRefs(storesThemeConfig);

const setLockScreen = computed(() => {
	return themeConfig.value.isLockScreen ? themeConfig.value.lockScreenTime > 1 : themeConfig.value.lockScreenTime >= 0;
});

const getGlobalComponentSize = computed(() => {
	return other.globalComponentSize();
});
const getGlobalI18n = computed(() => {
	return messages.value[locale.value];
});

const handleOpenSettingsDrawer = () => {
	setingsRef.value?.openDrawer();
};

onBeforeMount(() => {
	setIntroduction.cssCdn();
	setIntroduction.jsCdn();
});

onMounted(() => {
	nextTick(() => {
		mittBus.on('openSetingsDrawer', handleOpenSettingsDrawer);

		const localThemeConfig = Local.get('themeConfig');
		if (localThemeConfig) {
			if (localThemeConfig.ver !== themeConfig.value.ver) {
				Local.remove('themeConfig');
				Local.set('themeConfig', themeConfig.value);
			}

			storesThemeConfig.setThemeConfig({ themeConfig: Local.get('themeConfig') });
			document.documentElement.style.cssText = Local.get('themeConfigStyle') || '';
		}

		if (Session.get('isTagsViewCurrenFull')) {
			stores.setCurrenFullscreen(Session.get('isTagsViewCurrenFull'));
		}
	});
});

onUnmounted(() => {
	mittBus.off('openSetingsDrawer', handleOpenSettingsDrawer);
});

watch(
	() => route.path,
	() => {
		other.useTitle();
	},
	{
		deep: true,
	}
);
</script>
