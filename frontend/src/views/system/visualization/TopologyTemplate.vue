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
import {ref, watch, onMounted, onBeforeUnmount, computed, nextTick} from 'vue'
import * as echarts from 'echarts'
import {PieChart} from '@element-plus/icons-vue'
import {graphlib, layout as dagreLayout} from '@dagrejs/dagre'

export default {
  name: 'TopologyTemplate',
  components: {PieChart},
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

    // 优化的颜色调色板（12种容易区分的颜色）
    const COLOR_PALETTE = [
      '#5470c6', '#91cc75', '#fac858', '#ee6666',
      '#73c0de', '#3ba272', '#fc8452', '#9a60b4',
      '#ea7ccc', '#17b3a3', '#b7a4f3', '#ff88aa'
    ]

    // 根据数据内容生成稳定颜色
    const generateColor = (data) => {
      if (!colorMap.value.has(data)) {
        // 通过哈希算法生成稳定颜色索引
        let hash = 0
        for (let i = 0; i < data.length; i++) {
          hash = data.charCodeAt(i) + ((hash << 5) - hash)
        }
        const index = Math.abs(hash) % COLOR_PALETTE.length
        colorMap.value.set(data, COLOR_PALETTE[index])
      }
      return colorMap.value.get(data)
    }

    // 计算文本所需尺寸
    const calculateNodeSize = (text) => {
      const lines = text.split('\n')
      const maxLineLength = Math.max(...lines.map(line => line.length))
      const width = Math.min(300, Math.max(140, maxLineLength * 10)) // 每个字符按12px估算
      const height = Math.max(80, lines.length * 28) // 每行按28px估算
      return [width , height] // 添加内边距
    }

    // 辅助函数：根据变量名列表，从 item 中取第一个非空值
    const getFirstNonEmptyValue = (obj, variables) => {
      if (!obj || typeof obj !== 'object' || !Array.isArray(variables)) return null
      for (const key of variables) {
        const val = obj[key]
        if (val !== null && val !== undefined && val !== '') {
          return val
        }
      }
      return null
    }

    // 处理拓扑数据
    const topologyData = computed(() => {
      try {
        const variables = props.config && props.config.variables
        if (!Array.isArray(variables)) return null

        // 取最新一条包含指定变量非空值的数据
        const latestData = [...(props.data || [])]
          .reverse()
          .map(item => getFirstNonEmptyValue(item, variables))
          .find(v => v !== null)

        if (!latestData || typeof latestData !== 'object') return null

        colorMap.value.clear()
        const nodes = []
        const edges = []

        Object.entries(latestData).forEach(([nodeId, nodeInfo]) => {
          const {service_name, data} = nodeInfo.service
          const labelText = `${service_name}\n${data}`
          const [width, height] = calculateNodeSize(labelText)
          const bgColor = generateColor(String(data))

          nodes.push({
            id: nodeId,
            name: service_name,
            data: data,
            symbol: 'Rect',
            symbolSize: [width, height],
            itemStyle: {
              color: bgColor,
              borderColor: '#2c3e50',
              borderWidth: 0,
              borderRadius: 0
            },
            label: {
              formatter: `{title|${service_name}}\n{divider|─}\n{content|${data}}`,
              rich: {
                title: {
                  fontSize: 14,
                  fontWeight: 'bold',
                  color: getContrastColor(bgColor),
                  padding: [5, 0]
                },
                divider: {
                  fontSize: 18,
                  color: getContrastColor(bgColor, 0.6),
                  lineHeight: 12
                },
                content: {
                  fontSize: 14,
                  color: getContrastColor(bgColor),
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
          nodesep: 40,
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

        return {nodes, edges}
      } catch (e) {
        console.error('Process topology data failed:', e)
        return null
      }
    })

    // 根据背景色获取对比色
    const getContrastColor = (hex, opacity = 1) => {
      const r = parseInt(hex.slice(1, 3), 16)
      const g = parseInt(hex.slice(3, 5), 16)
      const b = parseInt(hex.slice(5, 7), 16)
      const brightness = (r * 299 + g * 587 + b * 114) / 1000
      return brightness > 150
          ? `rgba(44, 62, 80, ${opacity})`  // 深色文字
          : `rgba(236, 240, 241, ${opacity})` // 浅色文字
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

      chartInstance.value.setOption({
        tooltip: {
          backgroundColor: 'rgba(255,255,255,0.95)',
          formatter: params => {
            if (params.dataType === 'node') {
              return `
                <div style="max-width: 300px">
                  <div style="font-size:16px;font-weight:bold;color:#2c3e50;margin-bottom:8px">
                    ${params.data.name}
                  </div>
                  <div style="color:#7f8c8d">
                    Data:
                    <span style="color:${params.data.itemStyle.color};font-weight:500">
                      ${params.data.data}
                    </span>
                  </div>
                </div>
              `
            }
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
              borderWidth: 0,
              borderRadius: 0,
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
      if (props.data.length > 0) updateChart()
    })

    onBeforeUnmount(() => {
      window.removeEventListener('resize', handleResize)
      chartInstance.value?.dispose()
    })

    return {chart: chartRef, showEmptyState, emptyMessage}
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
  color: #909399; /* fallback color (Element Plus text secondary) */
}

.empty-state .el-icon {
  color: #909399; /* fallback color (Element Plus text secondary) */
}
</style>