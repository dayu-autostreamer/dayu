import { useVueFlow } from '@vue-flow/core';
import { ref, watch } from 'vue';

const state = {
	serviceData: ref(null),
	draggedType: ref(null),
	isDragOver: ref(false),
	isDragging: ref(false),
};
const GRID_SIZE = 24;
const NODE_TONES = [
	{ background: '#eff6ff', border: '#93c5fd' },
	{ background: '#ecfeff', border: '#67e8f9' },
	{ background: '#f0fdf4', border: '#86efac' },
	{ background: '#fff7ed', border: '#fdba74' },
	{ background: '#fdf2f8', border: '#f9a8d4' },
	{ background: '#eef2ff', border: '#a5b4fc' },
	{ background: '#fefce8', border: '#fde047' },
	{ background: '#f8fafc', border: '#cbd5e1' },
];

export default function useDragAndDrop(flowId = 'default') {
	const { draggedType, isDragOver, isDragging, serviceData } = state;

	const { screenToFlowCoordinate, onNodesInitialized, updateNode } = useVueFlow({ id: flowId });

	watch(isDragging, (dragging) => {
		document.body.style.userSelect = dragging ? 'none' : '';
	});

	function onDragStart(event, type, service) {
		if (event.dataTransfer) {
			event.dataTransfer.setData('application/vueflow', type);
			event.dataTransfer.effectAllowed = 'move';
		}

		draggedType.value = type;
		isDragging.value = true;
		serviceData.value = service;

		document.addEventListener('drop', onDragEnd);
	}

	function onDragOver(event) {
		event.preventDefault();

		if (draggedType.value) {
			isDragOver.value = true;

			if (event.dataTransfer) {
				event.dataTransfer.dropEffect = 'move';
			}
		}
	}

	function onDragLeave() {
		isDragOver.value = false;
	}

	function onDragEnd() {
		isDragging.value = false;
		isDragOver.value = false;
		draggedType.value = null;
		serviceData.value = null;
		document.removeEventListener('drop', onDragEnd);
	}

	function getNodeTone(key) {
		const source = String(key || '');
		let hash = 0;
		for (let i = 0; i < source.length; i += 1) {
			hash = (hash << 5) - hash + source.charCodeAt(i);
			hash |= 0;
		}

		return NODE_TONES[Math.abs(hash) % NODE_TONES.length];
	}
	function snapToGrid(value) {
		return Math.round(value / GRID_SIZE) * GRID_SIZE;
	}

	function onDrop(event, nodeList, nodeMap) {
		event.preventDefault();

		if (!serviceData.value) {
			onDragEnd();
			return;
		}

		const position = screenToFlowCoordinate({
			x: event.clientX,
			y: event.clientY,
		});

		const nodeId = serviceData.value.id;
		const nodeName = serviceData.value.name || serviceData.value.id;
		const tone = getNodeTone(nodeId);
		const nodeData = {
			label: nodeName,
			prev: [],
			succ: [],
			service_id: serviceData.value.id,
			service_name: nodeName,
			description: serviceData.value.description,
		};
		const newNode = {
			id: nodeId,
			type: draggedType.value,
			class: 'dag-node',
			style: {
				backgroundColor: tone.background,
				border: `1px solid ${tone.border}`,
				borderRadius: '16px',
				boxShadow: '0 12px 28px rgba(15, 23, 42, 0.08)',
				color: '#0f172a',
				width: '180px',
				height: '56px',
			},
			data: nodeData,
			sourcePosition: 'right',
			targetPosition: 'left',
			position: {
				x: snapToGrid(position.x),
				y: snapToGrid(position.y),
			},
		};

		const { off } = onNodesInitialized(() => {
			updateNode(nodeId, (node) => ({
				position: {
					x: snapToGrid(node.position.x),
					y: snapToGrid(node.position.y),
				},
			}));

			off();
		});

		nodeMap[nodeId] = newNode;
		nodeList.push(newNode);
		onDragEnd();
	}

	return {
		draggedType,
		isDragOver,
		isDragging,
		serviceData,
		onDragStart,
		onDragLeave,
		onDragOver,
		onDrop,
	};
}
