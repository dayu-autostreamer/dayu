import type { App } from 'vue';
import { useUserInfo } from '/@/stores/userInfo';
import { judementSameArr } from '/@/utils/arrayOperation';

/**
 * Register permission directives.
 */
export function authDirective(app: App) {
	const removeElement = (el: HTMLElement) => {
		el.parentNode?.removeChild(el);
	};

	app.directive('auth', {
		mounted(el: HTMLElement, binding) {
			const stores = useUserInfo();
			if (!stores.userInfos.authBtnList.some((v: string) => v === binding.value)) removeElement(el);
		},
	});

	app.directive('auths', {
		mounted(el: HTMLElement, binding) {
			const stores = useUserInfo();
			const hasPermission = binding.value.some((permission: string) => stores.userInfos.authBtnList.includes(permission));
			if (!hasPermission) removeElement(el);
		},
	});

	app.directive('auth-all', {
		mounted(el: HTMLElement, binding) {
			const stores = useUserInfo();
			const flag = judementSameArr(binding.value, stores.userInfos.authBtnList);
			if (!flag) removeElement(el);
		},
	});
}
