import dagre from '@dagrejs/dagre';
import { Position, useVueFlow } from '@vue-flow/core';
import { ref } from 'vue';

export function useLayout(flowId = 'default') {
	const { findNode } = useVueFlow({ id: flowId });
	const graph = ref(new dagre.graphlib.Graph());
	const previousDirection = ref('LR');
	const defaultNodeSize = {
		width: 96,
		height: 36,
	};
	const defaultLayoutOptions = {
		nodesep: 20,
		ranksep: 42,
		marginx: 16,
		marginy: 16,
	};

	function layout(nodes, edges, direction, options = {}) {
		if (!Array.isArray(nodes)) {
			console.error('Invalid nodes:', nodes);
			return [];
		}
		const nodesCopy = [...nodes];
		const edgesCopy = [...edges];

		const dagreGraph = new dagre.graphlib.Graph();
		graph.value = dagreGraph;
		dagreGraph.setDefaultEdgeLabel(() => ({}));
		const isHorizontal = direction === 'LR';
		const layoutOptions = { ...defaultLayoutOptions, ...options };
		dagreGraph.setGraph({
			rankdir: direction,
			nodesep: layoutOptions.nodesep,
			ranksep: layoutOptions.ranksep,
			marginx: layoutOptions.marginx,
			marginy: layoutOptions.marginy,
		});
		previousDirection.value = direction;

		nodesCopy.forEach((node) => {
			const graphNode = findNode(node.id);

			const dimensions = graphNode?.dimensions || node.dimensions || defaultNodeSize;

			dagreGraph.setNode(node.id, {
				width: dimensions.width,
				height: dimensions.height,
			});
		});
		if (Array.isArray(edgesCopy)) {
			edgesCopy.forEach((edge) => {
				if (edge?.source && edge?.target) {
					dagreGraph.setEdge(edge.source, edge.target);
				}
			});
		}

		try {
			dagre.layout(dagreGraph);
		} catch (e) {
			console.error('Dagre layout failed:', e);
			return nodesCopy;
		}

		return nodesCopy.map((node) => {
			try {
				const nodeWithPosition = dagreGraph.node(node.id);
				const graphNode = findNode(node.id);
				const dimensions = graphNode?.dimensions || node.dimensions || defaultNodeSize;
				const fallbackPosition = node.position || { x: 0, y: 0 };

				return {
					...node,
					targetPosition: isHorizontal ? Position.Left : Position.Top,
					sourcePosition: isHorizontal ? Position.Right : Position.Bottom,
					position: {
						x: nodeWithPosition ? nodeWithPosition.x - dimensions.width / 2 : fallbackPosition.x,
						y: nodeWithPosition ? nodeWithPosition.y - dimensions.height / 2 : fallbackPosition.y,
					},
				};
			} catch {
				return node;
			}
		});
	}

	return { graph, layout, previousDirection };
}
