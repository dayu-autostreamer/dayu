<template>
  <div class="chart-container">
    <div ref="container" class="chart-wrapper"></div>
    <div v-if="showEmptyState" class="empty-state">
      <el-icon :size="40"><PieChart /></el-icon>
      <p>{{ emptyMessage }}</p>
    </div>
  </div>
</template>

<script>
import { ref, computed, onMounted, onBeforeUnmount, watch, nextTick } from 'vue'
import * as echarts from 'echarts'
import { PieChart } from '@element-plus/icons-vue'
import mitt from 'mitt'

const emitter = mitt()

export default {
  name: 'CurveTemplate',
  components: { PieChart },
  props: {
    config: {
      type: Object,
      default: () => ({})
    },
    data: {
      type: Array,
      required: true,
      default: () => []
    },
    variableStates: {
      type: Object,
      default: () => ({})
    }
  },

  setup(props) {
    // Refs
    const chart = ref(null)
    const container = ref(null)
    const initialized = ref(false)
    const resizeObserver = ref(null)
    const lastRenderData = ref(null)

    // Computed Properties
    const safeData = computed(() => {
      try {
        return Array.isArray(props.data) ? props.data : []
      } catch (e) {
        console.error('Data format error:', e)
        return []
      }
    })

    const availableVariables = computed(() => {
      return safeData.value[0]
        ? Object.keys(safeData.value[0]).filter(k => k !== 'taskId')
        : []
    })

    const activeVariables = computed(() => {
      return availableVariables.value.filter(
        k => props.variableStates?.[k] !== false
      )
    })

    const showEmptyState = computed(() => {
      return !safeData.value.length || !activeVariables.value.length
    })

    const emptyMessage = computed(() => {
      if (!safeData.value.length) return 'No data available'
      if (!activeVariables.value.length) return 'No active variables selected'
      return ''
    })

    // Methods
    const initChart = async () => {
      try {
        await nextTick()
        if (!container.value) {
          console.warn('Chart container not found')
          return
        }

        if (chart.value) {
          chart.value.dispose()
          chart.value = null
        }

        chart.value = echarts.init(container.value)
        initialized.value = true
        console.log(`[ECharts] Initialized chart for ${props.config.id || 'unknown'}`)

        // Setup resize observer
        resizeObserver.value = new ResizeObserver(() => {
          chart.value?.resize()
        })
        resizeObserver.value.observe(container.value)

        renderChart()
      } catch (e) {
        console.error('Chart initialization failed:', e)
        initialized.value = false
      }
    }

    const renderChart = () => {
      if (!chart.value || !initialized.value) return
      if (showEmptyState.value) {
        chart.value.clear()
        return
      }

      try {
        const currentDataKey = JSON.stringify({
          dataLength: safeData.value.length,
          variables: activeVariables.value
        })

        // Skip render if data hasn't changed
        if (lastRenderData.value === currentDataKey) return
        lastRenderData.value = currentDataKey

        const option = {
          animation: false,
          tooltip: {
            trigger: 'axis',
            formatter: params => {
              return `${params[0].axisValue}<br/>` +
                params.map(p =>
                  `${p.marker} ${p.seriesName}: ${Number(p.value).toFixed(2)}`
                ).join('<br>')
            }
          },
          legend: {
            data: activeVariables.value
          },
          grid: {
            left: '3%',
            right: '4%',
            bottom: '10%',
            containLabel: true
          },
          xAxis: {
            type: 'category',
            data: safeData.value.map(d => String(d.taskId || '')),
            axisLabel: {
              rotate: 45,
              formatter: value => value.length > 6 ? `${value.slice(0,6)}...` : value
            }
          },
          yAxis: {
            type: 'value',
            axisLabel: {
              formatter: value => Number(value).toFixed(2)
            }
          },
          series: activeVariables.value.map(varName => ({
            name: varName,
            type: 'line',
            data: safeData.value.map(d => {
              const val = d[varName]
              return typeof val === 'number' ? val :
                typeof val === 'string' ? parseFloat(val) || 0 : 0
            }),
            smooth: true,
            showSymbol: safeData.value.length < 20,
            lineStyle: {
              width: 2
            },
            emphasis: {
              lineStyle: {
                width: 3
              }
            }
          }))
        }

        chart.value.setOption(option, {
          notMerge: true,
          lazyUpdate: true
        })
      } catch (e) {
        console.error('Chart render error:', e)
      }
    }

    // Lifecycle Hooks
    onMounted(async () => {
      await initChart()
      emitter.on('force-update-charts', renderChart)
    })

    onBeforeUnmount(() => {
      emitter.off('force-update-charts', renderChart)
      resizeObserver.value?.disconnect()
      if (chart.value) {
        chart.value.dispose()
        chart.value = null
      }
    })

    // Watchers
    watch(
      () => [safeData.value, activeVariables.value],
      () => {
        if (initialized.value) {
          renderChart()
        } else {
          initChart()
        }
      },
      { deep: true }
    )

    return {
      container,
      showEmptyState,
      emptyMessage
    }
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
  min-height: 300px;
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
}
</style>