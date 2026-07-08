import { defineStore } from 'pinia';
import Cookies from 'js-cookie';
import { Session } from '/@/utils/storage';

/**
 * 用户信息
 * @methods setUserInfos 设置用户信息
 */
export const useUserInfo = defineStore('userInfo', {
	state: (): UserInfosState => ({
		userInfos: {
			userName: '',
			photo: '',
			time: 0,
			roles: [],
			authBtnList: [],
		},
	}),
	actions: {
		async setUserInfos() {
			// 存储用户信息到浏览器缓存
			if (Session.get('userInfo')) {
				this.userInfos = Session.get('userInfo');
			} else {
				const userInfos = <UserInfos>await this.getApiUserInfo();
				this.userInfos = userInfos;
			}
		},
		// Local user data used by the current single-entry Dayu login flow.
		async getApiUserInfo() {
			return new Promise((resolve) => {
				setTimeout(() => {
					const userName = Cookies.get('userName');
					let defaultRoles: Array<string> = [];
					let defaultAuthBtnList: Array<string> = [];
					const dayuRoles: Array<string> = ['dayu'];
					const dayuAuthBtnList: Array<string> = ['btn.add', 'btn.del', 'btn.edit', 'btn.link'];
					const commonRoles: Array<string> = ['common'];
					const commonAuthBtnList: Array<string> = ['btn.add', 'btn.link'];
					if (userName === 'dayu') {
						defaultRoles = dayuRoles;
						defaultAuthBtnList = dayuAuthBtnList;
					} else {
						defaultRoles = commonRoles;
						defaultAuthBtnList = commonAuthBtnList;
					}
					const userInfos = {
						userName: userName,
						photo: '/images/avatar.jpg',
						time: new Date().getTime(),
						roles: defaultRoles,
						authBtnList: defaultAuthBtnList,
					};
					Session.set('userInfo', userInfos);
					resolve(userInfos);
				}, 0);
			});
		},
	},
});
