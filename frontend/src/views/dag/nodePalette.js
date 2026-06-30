function hashString(value) {
	const source = String(value || '');
	let hash = 0;

	for (let i = 0; i < source.length; i += 1) {
		hash = (hash << 5) - hash + source.charCodeAt(i);
		hash |= 0;
	}

	return Math.abs(hash);
}

export function getServiceTone(key) {
	const hash = hashString(key);
	const hue = hash % 360;
	const saturation = 56 + (hash % 14);
	const backgroundLightness = 97 - ((hash >> 3) % 4);
	const borderLightness = 82 - ((hash >> 5) % 6);
	const accentLightness = 40 + ((hash >> 7) % 8);

	return {
		accent: `hsl(${hue} ${Math.min(saturation + 10, 82)}% ${accentLightness}%)`,
		background: `hsl(${hue} ${saturation}% ${backgroundLightness}%)`,
		border: `hsl(${hue} ${Math.max(saturation - 6, 42)}% ${borderLightness}%)`,
	};
}

export function getServiceNodeFontSize(label) {
	const length = String(label || '').length;

	if (length > 22) {
		return '8px';
	}

	if (length > 16) {
		return '9px';
	}

	if (length > 10) {
		return '10px';
	}

	return '11px';
}
