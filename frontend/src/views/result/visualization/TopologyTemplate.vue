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
import { ref, watch, onMounted, onBeforeUnmount, computed, nextTick } from 'vue'
import * as echarts from 'echarts'
import { PieChart } from '@element-plus/icons-vue'
import { graphlib, layout as dagreLayout } from '@dagrejs/dagre'

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
    const showEmptyState = ref(true)
    const emptyMessage = ref('No topology data available')
    const colorMap = ref(new Map())

    // 优化的配色方案（保证文字可读性）
    const COLOR_PALETTE = [
      '#E6F4FF', '#F0F5FF', '#F6FFED', '#FFF7E6',
      '#FFF2F0', '#F9F0FF', '#E6FFFB', '#F0F0F0'
    ]

    // 处理拓扑数据
    const topologyData = computed(() => {
      try {
        const latestData = props.data[props.data.length - 1]?.topology
        if (!latestData) return null

        colorMap.value.clear()
        const nodes = []
        const edges = []

        Object.entries(latestData).forEach(([nodeId, nodeInfo]) => {
          const { service_name, data } = nodeInfo.service
          const bgColor = COLOR_PALETTE[nodes.length % COLOR_PALETTE.length]

          nodes.push({
            id: nodeId,
            name: service_name,
            data: data,
            symbol: 'rect',
            symbolSize: [160, 80],
            itemStyle: {
              color: bgColor,
              borderColor: '#2c3e50',
              borderWidth: 2,
              borderRadius: 8
            },
            label: {
              formatter: `{title|${service_name}}\n{divider|─}\n{content|${data}}`,
              rich: {
                title: {
                  fontSize: 16,
                  fontWeight: 'bold',
                  color: '#2c3e50',
                  padding: [5, 0]
                },
                divider: {
                  fontSize: 18,
                  color: '#95a5a6',
                  lineHeight: 18
                },
                content: {
                  fontSize: 14,
                  color: '#34495e',
                  fontWeight: 500,
                  padding: [5, 0]
                }
              }
            }
          })

          // 创建边
          nodeInfo.next_nodes.forEach(nextNodeId => {
            edges.push({
              source: nodeId,
              target: nextNodeId,
              lineStyle: {
                color: '#95a5a6',
                width: 2,
                curveness: 0.2
              },
              arrow: {
                type: 'triangle',
                width: 10,
                length: 10
              }
            })
          })
        })

        // 自动布局
        const g = new graphlib.Graph()
        g.setGraph({
          rankdir: 'LR',
          nodesep: 80,
          ranksep: 60,
          marginx: 40,
          marginy: 40
        })
        g.setDefaultEdgeLabel(() => ({}))

        nodes.forEach(node => {
          g.setNode(node.id, {
            width: node.symbolSize[0],
            height: node.symbolSize[1]
          })
        })

        edges.forEach(edge => g.setEdge(edge.source, edge.target))
        dagreLayout(g)

        // 应用坐标
        nodes.forEach(node => {
          const pos = g.node(node.id)
          node.x = pos?.x || 0
          node.y = pos?.y || 0
        })

        return { nodes, edges }
      } catch (e) {
        console.error('Process topology data failed:', e)
        return null
      }
    })

    // 初始化图表
    const initChart = () => {
      if (!chartRef.value) return
      chartInstance.value = echarts.init(chartRef.value)
      window.addEventListener('resize', handleResize)
    }

    // 更新图表
    const updateChart = () => {
      if (!chartInstance.value || !topologyData.value) return

      chartInstance.value.setOption({
        tooltip: {
          backgroundColor: 'rgba(255,255,255,0.95)',
          formatter: params => {
            if (params.dataType === 'node') {
              return `
                <div style="max-width: 250px">
                  <div style="font-size:16px;font-weight:bold;color:#2c3e50;margin-bottom:8px">
                    ${params.data.name}
                  </div>
                  <div style="color:#7f8c8d">
                    Location: <span style="color:#2c3e50;font-weight:500">${params.data.data}</span>
                  </div>
                </div>
              `
            }
            return `
              <div style="color:#7f8c8d">
                Connection:
                <span style="color:#2c3e50;font-weight:500">
                  ${params.source} → ${params.target}
                </span>
              </div>
            `
          }
        },
        series: [{
          type: 'graph',
          layout: 'none',
          draggable: true,
          zoom: 0.9,
          roam: true,
          edgeSymbol: ['none', 'arrow'],
          edgeSymbolSize: [0, 10],
          label: {
            position: 'inside',
            show: true
          },
          lineStyle: {
            color: '#bdc3c7'
          },
          emphasis: {
            itemStyle: {
              borderColor: '#2c3e50',
              borderWidth: 3
            }
          },
          nodes: topologyData.value.nodes,
          edges: topologyData.value.edges
        }]
      })
    }

    const handleResize = () => chartInstance.value?.resize()

    // 空状态处理
    watch(topologyData, (newVal) => {
      showEmptyState.value = !newVal?.nodes?.length
      if (!showEmptyState.value) {
        nextTick(() => updateChart())
      }
    })

    onMounted(() => {
      initChart()
      // 初始数据加载检查
      if (props.data.length > 0) {
        updateChart()
      }
    })

    onBeforeUnmount(() => {
      window.removeEventListener('resize', handleResize)
      chartInstance.value?.dispose()
    })

    return { chart: chartRef, showEmptyState, emptyMessage }
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
  min-height: 500px;
}

.empty-state {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  text-align: center;
  z-index: 10;
}

.empty-state p {
  margin-top: 10px;
  font-size: 14px;
  color: var(--el-text-color-secondary);
}

.empty-state .el-icon {
  color: var(--el-text-color-secondary);
}
</style>