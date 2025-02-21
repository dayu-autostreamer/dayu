import { useVueFlow } from '@vue-flow/core'
import { ref, watch } from 'vue'

const state = {
    serviceData: ref(null),
    draggedType: ref(null),
    isDragOver: ref(false),
    isDragging: ref(false),
}

export default function useDragAndDrop() {
    const { draggedType, isDragOver, isDragging, serviceData } = state

    const { screenToFlowCoordinate, onNodesInitialized, updateNode } = useVueFlow()

    watch(isDragging, (dragging) => {
        document.body.style.userSelect = dragging ? 'none' : ''
    })

    function onDragStart(event, type, service) {
        if (event.dataTransfer) {
            event.dataTransfer.setData('application/vueflow', type)
            event.dataTransfer.effectAllowed = 'move'
        }

        draggedType.value = type
        isDragging.value = true
        serviceData.value = service

        document.addEventListener('drop', onDragEnd)
    }

    function onDragOver(event) {
        event.preventDefault()

        if (draggedType.value) {
            isDragOver.value = true

            if (event.dataTransfer) {
                event.dataTransfer.dropEffect = 'move'
            }
        }
    }

    function onDragLeave() {
        isDragOver.value = false
    }

    function onDragEnd() {
        isDragging.value = false
        isDragOver.value = false
        draggedType.value = null
        serviceData.value = null
        document.removeEventListener('drop', onDragEnd)
    }

    function randomColor() {
        const colors = [
            "#3674B5",
            "#578FCA",
            "#A1E3F9",
            "#D1F8EF",
            "#DFF2EB",
            "#B9E5E8",
            "#7AB2D3",
            "#4A628A"
        ];
        const randomIndex = Math.floor(Math.random() * colors.length);
        return colors[randomIndex];
    }
    function onDrop(event, nodeList, nodeMap) {
        const position = screenToFlowCoordinate({
            x: event.clientX,
            y: event.clientY,
        })

        const nodeId = serviceData.value.name
        const nodeData = {
            label: nodeId,
            prev: [],
            succ: [],
            service_id: serviceData.value.name,
        }
        const newNode = {
            id: nodeId,
            type: draggedType.value,
            position: position,
            style: { backgroundColor: randomColor() },
            data: nodeData
        }

        const { off } = onNodesInitialized(() => {
            updateNode(nodeId, (node) => ({
                position: { x: node.position.x - node.dimensions.width / 2, y: node.position.y - node.dimensions.height / 2 },
            }))

            off()
        })

        nodeMap[nodeId] = newNode
        nodeList.push(newNode)
    }

    return {
        draggedType,
        isDragOver,
        isDragging,
        onDragStart,
        onDragLeave,
        onDragOver,
        onDrop,
    }
}
