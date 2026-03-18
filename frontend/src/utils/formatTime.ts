/**
 * Format a date with custom tokens such as `YYYY-mm-dd HH:MM:SS`.
 * Supported extended tokens:
 * - `WWW`: weekday
 * - `QQQQ`: quarter
 * - `ZZZ`: week number
 */
export function formatDate(date: Date, format: string): string {
	const weekday = date.getDay().toString();
	const weekNumber = getWeek(date).toString();
	const quarterNumber = Math.floor((date.getMonth() + 3) / 3).toString();
	const tokens: Record<string, string> = {
		'Y+': date.getFullYear().toString(),
		'm+': (date.getMonth() + 1).toString(),
		'd+': date.getDate().toString(),
		'H+': date.getHours().toString(),
		'M+': date.getMinutes().toString(),
		'S+': date.getSeconds().toString(),
		'q+': quarterNumber,
	};
	const weekLabels: Record<string, string> = {
		'0': '日',
		'1': '一',
		'2': '二',
		'3': '三',
		'4': '四',
		'5': '五',
		'6': '六',
	};
	const quarterLabels: Record<string, string> = {
		'1': '一',
		'2': '二',
		'3': '三',
		'4': '四',
	};

	if (/(W+)/.test(format)) {
		format = format.replace(
			RegExp.$1,
			RegExp.$1.length > 1
				? RegExp.$1.length > 2
					? `星期${weekLabels[weekday]}`
					: `周${weekLabels[weekday]}`
				: weekLabels[weekday]
		);
	}
	if (/(Q+)/.test(format)) {
		format = format.replace(
			RegExp.$1,
			RegExp.$1.length === 4 ? `第${quarterLabels[quarterNumber]}季度` : quarterLabels[quarterNumber]
		);
	}
	if (/(Z+)/.test(format)) {
		format = format.replace(RegExp.$1, RegExp.$1.length === 3 ? `第${weekNumber}周` : weekNumber);
	}

	for (const key in tokens) {
		const match = new RegExp(`(${key})`).exec(format);
		if (match) {
			format = format.replace(
				match[1],
				RegExp.$1.length === 1 ? tokens[key] : tokens[key].padStart(RegExp.$1.length, '0')
			);
		}
	}

	return format;
}

/**
 * Return the ISO-like week number for a given date.
 */
export function getWeek(dateTime: Date): number {
	const target = new Date(dateTime.getTime());
	const weekday = target.getDay() || 7;

	target.setDate(target.getDate() - weekday + 6);

	let firstDay = new Date(target.getFullYear(), 0, 1);
	const dayOfWeek = firstDay.getDay();
	let offset = 1;

	if (dayOfWeek !== 0) {
		offset = 7 - dayOfWeek + 1;
	}

	firstDay = new Date(target.getFullYear(), 0, 1 + offset);
	const dayDiff = Math.ceil((target.valueOf() - firstDay.valueOf()) / 86400000);

	return Math.ceil(dayDiff / 7);
}

/**
 * Convert a date into a relative label such as `刚刚`, `3分钟前`, or a formatted date.
 */
export function formatPast(param: string | Date, format = 'YYYY-mm-dd'): string {
	const inputDate = param instanceof Date ? param : new Date(param);
	const elapsed = Date.now() - inputDate.getTime();

	if (elapsed < 10000) {
		return '刚刚';
	}
	if (elapsed < 60000) {
		return `${Math.floor(elapsed / 1000)}秒前`;
	}
	if (elapsed < 3600000) {
		return `${Math.floor(elapsed / 60000)}分钟前`;
	}
	if (elapsed < 86400000) {
		return `${Math.floor(elapsed / 3600000)}小时前`;
	}
	if (elapsed < 259200000) {
		return `${Math.floor(elapsed / 86400000)}天前`;
	}

	return formatDate(inputDate, format);
}

/**
 * Return a greeting that matches the current time of day.
 */
export function formatAxis(param: Date): string {
	const hour = new Date(param).getHours();

	if (hour < 12) return 'Good morning';
	if (hour < 17) return 'Good afternoon';
	if (hour < 22) return 'Good evening';
	return 'Good night';
}
