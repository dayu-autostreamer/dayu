import dagre from '@dagrejs/dagre'
import {Position, useVueFlow} from '@vue-flow/core'
import {ref} from 'vue'

export function useLayout() {
    const {findNode} = useVueFlow()
    const graph = ref(new dagre.graphlib.Graph())
    const previousDirection = ref('LR')

    function layout(nodes, edges, direction) {
        if (!Array.isArray(nodes)) {
            console.error('Invalid nodes:', nodes)
            return [] // 返回空数组避免后续错误
        }

        const dagreGraph = new dagre.graphlib.Graph()
        graph.value = dagreGraph
        dagreGraph.setDefaultEdgeLabel(() => ({}))
        const isHorizontal = direction === 'LR'
        dagreGraph.setGraph({rankdir: direction})
        previousDirection.value = direction

        nodes.forEach(node => {
            const graphNode = findNode(node.id)
            const dimensions = graphNode?.dimensions || {
                width: 200,
                height: 50
            }

            dagreGraph.setNode(node.id, {
                width: dimensions.width,
                height: dimensions.height
            })
        })
        if (Array.isArray(edges)) {
            edges.forEach(edge => {
                if (edge?.source && edge?.target) {
                    dagreGraph.setEdge(edge.source, edge.target)
                }
            })
        }

        try {
            dagre.layout(dagreGraph)
        } catch (e) {
            console.error('Dagre layout failed:', e)
            return nodes
        }

        return nodes.map(node => {
            try {
                const pos = dagreGraph.node(node.id) || {x: 0, y: 0}
                return {
                    ...node,
                    targetPosition: isHorizontal ? Position.Left : Position.Top,
                    sourcePosition: isHorizontal ? Position.Right : Position.Bottom,
                    position: {
                        x: pos.x - (node.dimensions?.width || 200) / 2, // 中心对齐
                        y: pos.y - (node.dimensions?.height || 50) / 2
                    }
                }
            } catch {
                return node
            }
        })
    }

    return {graph, layout, previousDirection}
}