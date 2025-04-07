<template>
  <div class="chart-container">
    <div ref="container" class="chart-wrapper"></div>
    <div v-if="!props.data?.length" class="empty-state">
      <el-icon :size="40"><PieChart /></el-icon>
      <p>No data available</p>
    </div>
  </div>
</template>

<script>
import { ref, onMounted, onBeforeUnmount, watch, nextTick } from 'vue'
import * as echarts from 'echarts'
import { PieChart } from '@element-plus/icons-vue'
import mitt from 'mitt'

const emitter = mitt()

export default {
  components: { PieChart },
  props: ['config', 'data', 'variableStates'],

  setup(props) {
    const chart = ref(null)
    const container = ref(null)
    const initialized = ref(false)
    const resizeObserver = ref(null)

    const initChart = async () => {
      await nextTick()

      if (!container.value) {
        console.error('Chart container not found')
        return
      }

      // 销毁旧实例
      if (chart.value) {
        chart.value.dispose()
        chart.value = null
      }

      try {
        chart.value = echarts.init(container.value)
        initialized.value = true
        console.log('ECharts initialized for', props.config.id)
        renderChart()

        // 添加响应式尺寸监听
        resizeObserver.value = new ResizeObserver(() => {
          chart.value?.resize()
        })
        resizeObserver.value.observe(container.value)
      } catch (e) {
        console.error('ECharts init failed:', e)
        setTimeout(initChart, 300) // 自动重试
      }
    }

    const renderChart = () => {
      if (!chart.value || !props.data?.length) return

      try {
        const option = {
          animation: false,
          xAxis: {
            type: 'category',
            data: props.data.map(d => d.taskId),
            axisLabel: {
              rotate: 45
            }
          },
          yAxis: {
            type: 'value',
            axisLabel: {
              formatter: value => Number(value).toFixed(2)
            }
          },
          series: Object.keys(props.data[0])
            .filter(k => k !== 'taskId' && props.variableStates[k])
            .map(varName => ({
              name: varName,
              type: 'line',
              data: props.data.map(d => d[varName]),
              smooth: true,
              showSymbol: props.data.length < 20,
              lineStyle: {
                width: 2
              }
            })),
          tooltip: {
            trigger: 'axis',
            formatter: params => {
              return `${params[0].axisValue}<br/>` +
                params.map(p =>
                  `${p.marker} ${p.seriesName}: ${p.value.toFixed(2)}`
                ).join('<br>')
            }
          },
          grid: {
            left: '3%',
            right: '4%',
            bottom: '10%',
            containLabel: true
          }
        }

        chart.value.setOption(option, { notMerge: true })
        console.log('Chart rendered with', props.data.length, 'data points')
      } catch (e) {
        console.error('Render error:', e)
      }
    }

    onMounted(async () => {
      await initChart()
      emitter.on('force-update-charts', renderChart)
    })

    onBeforeUnmount(() => {
      emitter.off('force-update-charts', renderChart)
      resizeObserver.value?.disconnect()
      chart.value?.dispose()
    })

    watch(() => props.data, () => {
      if (initialized.value) renderChart()
    }, { deep: true })

    watch(() => props.variableStates, () => {
      if (initialized.value) renderChart()
    }, { deep: true })

    return { container }
  }
}
</script>

<style scoped>
.chart-container {
  position: relative;
  width: 100%;
  height: 100%;
  min-height: 400px;
}

.chart-wrapper {
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
}
</style>