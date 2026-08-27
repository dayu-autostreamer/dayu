import { createApp } from 'vue';
import ElementPlus from 'element-plus';
import pinia from '/@/stores/index';
import App from '/@/App.vue';
import router from '/@/router';
import { directive } from '/@/directive/index';
import { i18n } from '/@/i18n/index';
import other from '/@/utils/other';
import '/@/theme/index.scss';
import '@vue-flow/core/dist/style.css';
import '@vue-flow/core/dist/theme-default.css';
import { useInstallStateStore } from '/@/stores/installState';
import { useSystemParametersStore } from '/@/stores/systemParameters';

const app = createApp(App);

directive(app);
other.elSvg(app);

app.use(pinia).use(router).use(ElementPlus).use(i18n);

const installState = useInstallStateStore();
installState.init();
const sysParams = useSystemParametersStore();
sysParams.init();

app.mount('#app');
