module.exports = {
	root: true,
	env: {
		browser: true,
		es2021: true,
		node: true,
	},
	extends: ['eslint:recommended', 'plugin:vue/vue3-essential', 'plugin:@typescript-eslint/recommended'],
	parser: 'vue-eslint-parser',
	parserOptions: {
		parser: '@typescript-eslint/parser',
		ecmaVersion: 'latest',
		sourceType: 'module',
		extraFileExtensions: ['.vue'],
	},
	ignorePatterns: ['dist', 'build', 'node_modules'],
	rules: {
		'no-console': 'off',
		'no-debugger': 'warn',
		'no-var': 'error',
		'prefer-const': 'warn',
		'vue/multi-word-component-names': 'off',
		'vue/no-v-html': 'off',
		'@typescript-eslint/no-explicit-any': 'off',
		'@typescript-eslint/ban-ts-comment': 'off',
		'@typescript-eslint/no-empty-function': 'off',
	},
};
