import { markRaw } from 'vue';

const registerModuleSet = async (modules, register) => {
	await Promise.all(
		Object.entries(modules).map(async ([path, loader]) => {
			const fileName = path.split('/').pop() || '';
			const type = fileName.replace(/(Template|Controls)\.vue$/, '').toLowerCase();
			try {
				const component = await loader();
				register(type, markRaw(component.default));
			} catch (error) {
				console.error(`Failed to load visualization module: ${path}`, error);
			}
		})
	);
};

export const registerVisualizationModules = async ({
	templateModules = {},
	controlModules = {},
	templatesTarget,
	controlsTarget,
}) => {
	await Promise.all([
		registerModuleSet(templateModules, (type, component) => {
			templatesTarget[type] = component;
		}),
		registerModuleSet(controlModules, (type, component) => {
			controlsTarget[type] = component;
		}),
	]);
};
