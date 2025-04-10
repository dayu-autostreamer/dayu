<template>
  <div class="topology-container">
    <div ref="chart" class="topology-chart"></div>

    <div v-if="showEmptyState" class="empty-state">
      <el-icon :size="40">
        <PieChart/>
      </el-icon>
      <p>{{ emptyMessage }}</p>
    </div>
  </div>
</template>

<script>
import { ref, watch, onMounted, onBeforeUnmount, computed } from 'vue'
import * as echarts from 'echarts'
import { PieChart } from '@element-plus/icons-vue'
import dagre from 'dagre'

export default {
  name: 'TopologyTemplate',
  components: { PieChart },
  props: {
    config: {
      type: Object,
      required: true,
      default: () => ({
        id: '',
        name: '',
        type: 'topology'
      })
    },
    data: {
      type: Array,
      required: true
    }
  },
  setup(props) {
    const chartRef = ref(null)
    const chartInstance = ref(null)
    const showEmptyState = ref(false)
    const emptyMessage = ref('No topology data available')

    // 处理拓扑数据
    const topologyData = computed(() => {
      try {
        const latestData = props.data[props.data.length - 1]?.topology
        if (!latestData) return null

        // 转换数据结构
        const nodes = []
        const edges = []
        const nodeMap = new Map()

        // 创建节点
        Object.entries(latestData).forEach(([nodeId, nodeInfo]) => {
          nodes.push({
            id: nodeId,
            name: nodeInfo.service.service_name,
            data: nodeInfo.service.data,
            symbolSize: 50,
            itemStyle: {
              color: getNodeColor(nodeInfo.service.data)
            }
          })
          nodeMap.set(nodeId, nodeInfo)
        })

        // 创建边
        Object.entries(latestData).forEach(([nodeId, nodeInfo]) => {
          nodeInfo.next_nodes.forEach(nextNodeId => {
            edges.push({
              source: nodeId,
              target: nextNodeId,
              lineStyle: {
                type: 'solid',
                curveness: 0.2
              },
              arrow: {
                type: 'triangle',
                width: 8
              }
            })
          })
        })

        // 使用dagre进行自动布局
        const g = new dagre.graphlib.Graph()
        g.setGraph({ rankdir: 'LR', align: 'UL', nodesep: 50, ranksep: 70 })
        g.setDefaultEdgeLabel(() => ({}))

        nodes.forEach(node => g.setNode(node.id, { width: 120, height: 60 }))
        edges.forEach(edge => g.setEdge(edge.source, edge.target))

        dagre.layout(g)

        // 应用布局坐标
        nodes.forEach(node => {
          const pos = g.node(node.id)
          node.x = pos.x
          node.y = pos.y
        })

        return { nodes, edges }
      } catch (e) {
        console.error('Process topology data failed:', e)
        return null
      }
    })

    const getNodeColor = (nodeData) => {
      // 根据节点数据设置不同颜色
      if (nodeData.includes('edge')) return '#91cc75'
      if (nodeData.includes('cloud')) return '#5470c6'
      return '#73c0de'
    }

    // 初始化图表
    const initChart = () => {
      if (!chartRef.value) return

      chartInstance.value = echarts.init(chartRef.value)
      window.addEventListener('resize', handleResize)
    }

    // 更新图表
    const updateChart = () => {
      if (!chartInstance.value || !topologyData.value) return

      const option = {
        tooltip: {
          formatter: params => {
            if (params.dataType === 'node') {
              return `${params.data.name}<br/>Location: ${params.data.data}`
            }
            return `${params.source} → ${params.target}`
          }
        },
        series: [{
          type: 'graph',
          layout: 'none',
          draggable: true,
          zoom: 0.8,
          roam: true,
          focusNodeAdjacency: true,
          edgeSymbol: ['none', 'arrow'],
          label: {
            show: true,
            formatter: ({ data }) =>
              `{name|${data.name}}\n{data|${data.data}}`,
            rich: {
              name: {
                fontWeight: 'bold',
                fontSize: 14,
                color: '#333'
              },
              data: {
                fontSize: 12,
                color: '#666',
                padding: [5, 0]
              }
            }
          },
          edgeLabel: {
            show: false
          },
          lineStyle: {
            color: '#aaa',
            width: 1.5
          },
          emphasis: {
            label: {
              show: true
            },
            lineStyle: {
              width: 2.5
            }
          },
          nodes: topologyData.value.nodes,
          edges: topologyData.value.edges
        }]
      }

      chartInstance.value.setOption(option)
      showEmptyState.value = false
    }

    const handleResize = () => {
      chartInstance.value?.resize()
    }

    // 空状态检查
    watch(topologyData, (newVal) => {
      showEmptyState.value = !newVal?.nodes?.length
      if (!showEmptyState.value) {
        nextTick(() => updateChart())
      }
    })

    onMounted(() => {
      initChart()
      updateChart()
    })

    onBeforeUnmount(() => {
      window.removeEventListener('resize', handleResize)
      chartInstance.value?.dispose()
    })

    return {
      chart: chartRef,
      showEmptyState,
      emptyMessage
    }
  }
}
</script>

<style scoped>
.topology-container {
  width: 100%;
  height: 100%;
  min-height: 500px;
  position: relative;
}

.topology-chart {
  width: 100%;
  height: 100%;
}

.empty-state {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  text-align: center;
  color: var(--el-text-color-secondary);
  z-index: 10;
}

.empty-state p {
  margin-top: 10px;
  font-size: 14px;
}
</style>