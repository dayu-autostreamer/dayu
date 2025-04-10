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
import {ref, watch, onMounted, onBeforeUnmount, computed} from 'vue'
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
    const showEmptyState = ref(false)
    const emptyMessage = ref('No topology data available')
    const colorMap = ref(new Map())

    // 现代感配色方案
    const COLOR_PALETTE = [
      '#6C5B7B', '#C06C84', '#F67280', '#F8B195', // 紫色系
      '#4B8BBE', '#306998', '#FFE873', '#646464', // 蓝色/金色系
      '#2F9599', '#F7DB4F', '#FF6F61', '#A7226E'  // 绿/黄/红系
    ]

    // 生成协调颜色
    const generateColor = (str) => {
      let hash = 0
      for (let i = 0; i < str.length; i++) {
        hash = str.charCodeAt(i) + ((hash << 5) - hash)
      }
      return COLOR_PALETTE[Math.abs(hash) % COLOR_PALETTE.length]
    }

    // 计算文本尺寸
    const calculateTextSize = (text, fontSize = 12) => {
      const lineCount = text.split('\n').length
      const maxLength = Math.max(...text.split('\n').map(l => l.length))
      return {
        width: Math.min(200, maxLength * fontSize * 0.6),
        height: lineCount * fontSize * 1.5
      }
    }

    // 处理拓扑数据
    const topologyData = computed(() => {
      try {
        const latestData = props.data[props.data.length - 1]?.topology
        if (!latestData) return null

        colorMap.value.clear()
        const nodes = []
        const edges = []

        // 创建节点
        Object.entries(latestData).forEach(([nodeId, nodeInfo]) => {
          const {service_name, data} = nodeInfo.service
          const labelText = `{title|${service_name}}\n{divider|─}\n{content|${data}}`

          // 计算节点尺寸
          const {width, height} = calculateTextSize(`${service_name}\n${data}`, 14)
          const nodeWidth = Math.max(120, width + 40)
          const nodeHeight = Math.max(80, height + 30)

          // 生成颜色
          if (!colorMap.value.has(data)) {
            colorMap.value.set(data, generateColor(data))
          }

          nodes.push({
            id: nodeId,
            name: service_name,
            data: data,
            symbol: 'rect',
            symbolSize: [nodeWidth, nodeHeight],
            itemStyle: {
              color: colorMap.value.get(data),
              borderColor: '#2c3e50',
              borderWidth: 2,
              borderRadius: 8
            },
            label: {
              formatter: labelText,
              rich: {
                title: {
                  fontSize: 16,
                  fontWeight: 'bold',
                  color: '#2c3e50',
                  padding: [5, 0],
                  align: 'center'
                },
                divider: {
                  fontSize: 18,
                  color: '#2c3e50',
                  align: 'center',
                  lineHeight: 18
                },
                content: {
                  fontSize: 14,
                  color: '#34495e',
                  fontWeight: 500,
                  align: 'center',
                  padding: [5, 0]
                }
              }
            }
          })
        })

        // 创建边
        Object.entries(latestData).forEach(([nodeId, nodeInfo]) => {
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
                length: 10,
                itemStyle: {
                  color: '#95a5a6'
                }
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

        return {nodes, edges}
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

      const option = {
        tooltip: {
          backgroundColor: 'rgba(255,255,255,0.95)',
          borderColor: '#ddd',
          borderWidth: 1,
          textStyle: {
            color: '#2c3e50',
            fontSize: 14
          },
          formatter: params => {
            if (params.dataType === 'node') {
              return `
                <div style="max-width: 250px">
                  <div style="font-size:16px;font-weight:bold;color:#2c3e50;margin-bottom:8px">
                    ${params.data.name}
                  </div>
                  <div style="display:flex;align-items:center;margin-bottom:6px">
                    <div style="width:12px;height:12px;background:${params.data.itemStyle.color};
                      margin-right:8px;border-radius:2px"></div>
                    <span style="color:#7f8c8d">Location:</span>
                    <span style="margin-left:6px;font-weight:500">${params.data.data}</span>
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
          focusNodeAdjacency: true,
          edgeSymbol: ['none', 'arrow'],
          edgeSymbolSize: [0, 10],
          label: {
            position: 'inside',
            show: true,
            fontSize: 14
          },
          lineStyle: {
            color: '#bdc3c7'
          },
          emphasis: {
            label: {
              show: true
            },
            itemStyle: {
              borderColor: '#2c3e50',
              borderWidth: 3
            }
          },
          nodes: topologyData.value.nodes,
          edges: topologyData.value.edges
        }]
      }

      chartInstance.value.setOption(option)
      showEmptyState.value = false
    }

    const handleResize = () => chartInstance.value?.resize()

    watch(topologyData, (newVal) => {
      showEmptyState.value = !newVal?.nodes?.length
      if (!showEmptyState.value) updateChart()
    })

    onMounted(() => {
      initChart()
      updateChart()
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
  min-height: 600px;
  position: relative;
  background: #f5f6fa;
  border-radius: 12px;
  overflow: hidden;
}

.topology-chart {
  width: 100%;
  height: 100%;
  min-height: 600px;
}

.empty-state {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  text-align: center;
  background: rgba(255, 255, 255, 0.9);
  padding: 20px 40px;
  border-radius: 8px;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.1);
}

.empty-state p {
  margin-top: 10px;
  font-size: 16px;
  color: #7f8c8d;
}
</style>