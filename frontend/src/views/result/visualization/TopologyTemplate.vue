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
    const showEmptyState = ref(true) // 默认显示空状态
    const emptyMessage = ref('No topology data available')
    const colorMap = ref(new Map())

    // 优化后的配色方案（保证对比度）
    const COLOR_PALETTE = [
      '#2C3E50', '#34495E', '#16A085', '#27AE60',
      '#2980B9', '#8E44AD', '#2C3E50', '#D35400',
      '#C0392B', '#7F8C8D', '#F39C12', '#BDC3C7'
    ]

    // 根据背景色计算最佳文字颜色（YIQ对比度算法）
    const getContrastColor = (hexColor) => {
      const r = parseInt(hexColor.substr(1, 2), 16)
      const g = parseInt(hexColor.substr(3, 2), 16)
      const b = parseInt(hexColor.substr(5, 2), 16)
      const yiq = ((r * 299) + (g * 587) + (b * 114)) / 1000
      return yiq >= 128 ? '#2c3e50' : '#ecf0f1'
    }

    // 生成颜色并自动计算文字颜色
    const generateStyle = (data) => {
      if (!colorMap.value.has(data)) {
        const color = COLOR_PALETTE[colorMap.value.size % COLOR_PALETTE.length]
        colorMap.value.set(data, {
          bgColor: color,
          textColor: getContrastColor(color)
        })
      }
      return colorMap.value.get(data)
    }

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
          const { bgColor, textColor } = generateStyle(data)

          const labelText = `{title|${service_name}}\n{divider|─}\n{content|${data}}`
          const { width, height } = calculateTextSize(`${service_name}\n${data}`)

          nodes.push({
            id: nodeId,
            name: service_name,
            data: data,
            symbol: 'rect',
            symbolSize: [Math.max(140, width + 40), Math.max(90, height + 30)],
            itemStyle: {
              color: bgColor,
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
                  color: textColor,
                  padding: [5, 0]
                },
                divider: {
                  fontSize: 18,
                  color: textColor,
                  lineHeight: 18
                },
                content: {
                  fontSize: 14,
                  color: textColor,
                  fontWeight: 500,
                  padding: [5, 0]
                }
              }
            }
          })
        })

        // 边创建逻辑保持不变...
        return { nodes, edges }
      } catch (e) {
        return null
      }
    })

    // 计算文本尺寸（保持不变）...

    // 初始化图表
    const initChart = () => {
      if (!chartRef.value) return
      chartInstance.value = echarts.init(chartRef.value)
      window.addEventListener('resize', handleResize)
      updateChart() // 初始化时立即更新
    }

    // 更新图表逻辑保持不变...

    // 空状态处理增强
    watch(topologyData, (newVal) => {
      showEmptyState.value = !newVal?.nodes?.length
      if (!showEmptyState.value) {
        nextTick(() => {
          updateChart()
          chartRef.value.style.background = 'none' // 移除默认背景
        })
      } else {
        chartRef.value.style.background = '#f5f6fa'
      }
    })

    // 生命周期和样式保持不变...

    return { chart: chartRef, showEmptyState, emptyMessage }
  }
}
</script>

<style scoped>
.topology-container {
  width: 100%;
  height: 100%;
  min-height: 600px;
  position: relative;
}

.topology-chart {
  width: 100%;
  height: 100%;
  min-height: 600px;
  background: #fff; /* 改为白色背景 */
}

.empty-state {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  text-align: center;
  z-index: 10; /* 确保在最上层 */
  background: rgba(255,255,255,0.9);
  padding: 30px 50px;
  border-radius: 12px;
  box-shadow: 0 4px 20px rgba(0,0,0,0.1);
}

.empty-state p {
  margin-top: 15px;
  font-size: 16px;
  color: #7f8c8d;
  font-weight: 500;
}

.empty-state .el-icon {
  color: #bdc3c7;
}
</style>