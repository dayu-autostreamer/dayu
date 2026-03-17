import type { App } from 'vue';

type WavesElement = HTMLElement & {
	__wavesHandler__?: (event: MouseEvent) => void;
};

/**
 * Register the waves click effect directive.
 */
export function wavesDirective(app: App) {
	app.directive('waves', {
		mounted(el: WavesElement, binding) {
			el.classList.add('waves-effect');
			if (binding.value) el.classList.add(`waves-${binding.value}`);

			function setConvertStyle(obj: Record<string, string | number>) {
				let style: string = '';
				for (let i in obj) {
					if (Object.prototype.hasOwnProperty.call(obj, i)) style += `${i}:${obj[i]};`;
				}
				return style;
			}

			const onCurrentClick = (event: MouseEvent) => {
				let elDiv = document.createElement('div');
				elDiv.classList.add('waves-ripple');
				el.appendChild(elDiv);
				const { left, top } = el.getBoundingClientRect();
				let styles = {
					left: `${event.clientX - left}px`,
					top: `${event.clientY - top}px`,
					opacity: 1,
					transform: `scale(${(el.clientWidth / 100) * 10})`,
					'transition-duration': `750ms`,
					'transition-timing-function': `cubic-bezier(0.250, 0.460, 0.450, 0.940)`,
				};
				elDiv.setAttribute('style', setConvertStyle(styles));
				setTimeout(() => {
					elDiv.setAttribute(
						'style',
						setConvertStyle({
							opacity: 0,
							transform: styles.transform,
							left: styles.left,
							top: styles.top,
						})
					);
					setTimeout(() => {
						elDiv && el.removeChild(elDiv);
					}, 750);
				}, 450);
			};

			el.__wavesHandler__ = onCurrentClick;
			el.addEventListener('mousedown', onCurrentClick, false);
		},
		unmounted(el: WavesElement) {
			if (!el.__wavesHandler__) return;
			el.removeEventListener('mousedown', el.__wavesHandler__);
			delete el.__wavesHandler__;
		},
	});
}

/**
 * Register the dialog drag directive.
 */
export function dragDirective(app: App) {
	app.directive('drag', {
		mounted(el, binding) {
			if (!binding.value) return;

			const dragDom = document.querySelector(binding.value[0]) as HTMLElement;
			const dragHeader = document.querySelector(binding.value[1]) as HTMLElement;

			dragHeader.onmouseover = () => (dragHeader.style.cursor = `move`);

			function down(e: any, type: string) {
				const disX = type === 'pc' ? e.clientX - dragHeader.offsetLeft : e.touches[0].clientX - dragHeader.offsetLeft;
				const disY = type === 'pc' ? e.clientY - dragHeader.offsetTop : e.touches[0].clientY - dragHeader.offsetTop;

				const screenWidth = document.body.clientWidth;
				const screenHeight = document.documentElement.clientHeight;
				const dragDomWidth = dragDom.offsetWidth;
				const dragDomHeight = dragDom.offsetHeight;

				const minDragDomLeft = dragDom.offsetLeft;
				const maxDragDomLeft = screenWidth - dragDom.offsetLeft - dragDomWidth;

				const minDragDomTop = dragDom.offsetTop;
				const maxDragDomTop = screenHeight - dragDom.offsetTop - dragDomHeight;

				let styL: any = getComputedStyle(dragDom).left;
				let styT: any = getComputedStyle(dragDom).top;

				if (styL.includes('%')) {
					styL = +document.body.clientWidth * (+styL.replace(/\%/g, '') / 100);
					styT = +document.body.clientHeight * (+styT.replace(/\%/g, '') / 100);
				} else {
					styL = +styL.replace(/\px/g, '');
					styT = +styT.replace(/\px/g, '');
				}

				return {
					disX,
					disY,
					minDragDomLeft,
					maxDragDomLeft,
					minDragDomTop,
					maxDragDomTop,
					styL,
					styT,
				};
			}

			function move(e: any, type: string, obj: any) {
				let { disX, disY, minDragDomLeft, maxDragDomLeft, minDragDomTop, maxDragDomTop, styL, styT } = obj;

				let left = type === 'pc' ? e.clientX - disX : e.touches[0].clientX - disX;
				let top = type === 'pc' ? e.clientY - disY : e.touches[0].clientY - disY;

				if (-left > minDragDomLeft) {
					left = -minDragDomLeft;
				} else if (left > maxDragDomLeft) {
					left = maxDragDomLeft;
				}

				if (-top > minDragDomTop) {
					top = -minDragDomTop;
				} else if (top > maxDragDomTop) {
					top = maxDragDomTop;
				}

				dragDom.style.cssText += `;left:${left + styL}px;top:${top + styT}px;`;
			}

			dragHeader.onmousedown = (e) => {
				const obj = down(e, 'pc');
				document.onmousemove = (e) => {
					move(e, 'pc', obj);
				};
				document.onmouseup = () => {
					document.onmousemove = null;
					document.onmouseup = null;
				};
			};

			dragHeader.ontouchstart = (e) => {
				const obj = down(e, 'app');
				document.ontouchmove = (e) => {
					move(e, 'app', obj);
				};
				document.ontouchend = () => {
					document.ontouchmove = null;
					document.ontouchend = null;
				};
			};
		},
	});
}
