import { ElMessage } from 'element-plus';
import { useI18n } from 'vue-i18n';

export default function () {
	const { t } = useI18n();

	const copyWithFallback = (text: string) => {
		const textarea = document.createElement('textarea');
		textarea.value = text;
		textarea.setAttribute('readonly', 'true');
		textarea.style.position = 'fixed';
		textarea.style.opacity = '0';
		document.body.appendChild(textarea);
		textarea.select();
		document.execCommand('copy');
		document.body.removeChild(textarea);
	};

	const copyText = async (text: string) => {
		try {
			try {
				if (navigator.clipboard?.writeText) await navigator.clipboard.writeText(text);
				else copyWithFallback(text);
			} catch {
				copyWithFallback(text);
			}
			ElMessage.success(t('message.layout.copyTextSuccess'));
			return text;
		} catch (e) {
			ElMessage.error(t('message.layout.copyTextError'));
			throw e;
		}
	};

	return {
		copyText,
	};
}
