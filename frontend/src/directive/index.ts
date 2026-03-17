import type { App } from 'vue';
import { authDirective } from '/@/directive/authDirective';
import { wavesDirective, dragDirective } from '/@/directive/customDirective';

/**
 * Register shared application directives.
 */
export function directive(app: App) {
	authDirective(app);
	wavesDirective(app);
	dragDirective(app);
}
