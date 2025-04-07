<template>
  <div class="chart-container">
    <div ref="container" class="chart-wrapper"></div>
    <div v-if="showEmptyState" class="empty-state">
      <el-icon :size="40">
        <PieChart/>
      </el-icon>
      <p>{{ emptyMessage }}</p>
    </div>
  </div>
</template>

<script>
import {ref, computed, onMounted, onBeforeUnmount, watch, nextTick} from 'vue'
import * as echarts from 'echarts'
import {PieChart} from '@element-plus/icons-vue'

export default {
  name: 'CurveTemplate',
  components: {PieChart},
  props: {
    config: {
      type: Object,
      required: true,
      default: () => ({
        id: '',
        name: '',
        type: 'curve',
        variables: []
      })
    },
    data: {
      type: Array,
      required: true,
      default: () => [],
      validator: value => Array.isArray(value) && value.every(item =>
          typeof item?.taskId !== 'undefined'
      )
    },
    variableStates: {
      type: Object,
      required: true,
      default: () => ({})
    }
  },

  setup(props) {
    // Refs
    const chart = ref(null)
    const container = ref(null)
    const initialized = ref(false)
    const lastRenderTime = ref(0)
    const resizeObserver = ref(null)

    // Computed Properties
    const safeData = computed(() => {
      try {
        return (props.data || []).map(item => ({
          ...item,
          taskId: String(item.taskId || '')
        }))
      } catch (e) {
        console.error('Data format error:', e)
        return []
      }
    })

    const availableVariables = computed(() => {
      if (!safeData.value.length) return []
      return Object.keys(safeData.value[0])
          .filter(k => k !== 'taskId' && props.config.variables?.includes(k))
    })

    const activeVariables = computed(() => {
      return availableVariables.value.filter(
          k => props.variableStates[k] !== false
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
          console.warn('Chart container element not found')
          return false
        }

        // Dispose previous instance
        if (chart.value) {
          chart.value.dispose()
          chart.value = null
        }

        // Initialize new instance
        chart.value = echarts.init(container.value)
        initialized.value = true

        // Setup resize observer
        resizeObserver.value = new ResizeObserver(() => {
          if (Date.now() - lastRenderTime.value > 200) {
            chart.value?.resize()
          }
        })
        resizeObserver.value.observe(container.value)

        return true
      } catch (e) {
        console.error('ECharts initialization failed:', e)
        initialized.value = false
        return false
      }
    }

    const renderChart = async () => {
      if (!chart.value || !initialized.value) {
        if (!await initChart()) return
      }

      if (showEmptyState.value) {
        chart.value.clear()
        return
      }

      try {
        const option = getChartOption()
        chart.value.setOption(option, {
          notMerge: true,
          lazyUpdate: true
        })
        lastRenderTime.value = Date.now()
      } catch (e) {
        console.error('Chart render error:', e)
      }
    }

    const getChartOption = () => {
      return {
        animation: false,
        tooltip: {
          trigger: 'axis',
          formatter: params => {
            if (!params || !params.length) return ''
            return `${params[0].axisValue}<br/>` +
                params.map(p =>
                    `${p.marker} ${p.seriesName}: ${Number(p.value).toFixed(2)}`
                ).join('<br>')
          }
        },
        legend: {
          data: activeVariables.value,
          type: 'scroll'
        },
        grid: {
          left: '3%',
          right: '4%',
          bottom: '15%',
          containLabel: true
        },
        xAxis: {
          type: 'category',
          data: safeData.value.map(d => d.taskId),
          axisLabel: {
            rotate: 45,
            formatter: value => value.length > 8 ? `${value.slice(0, 8)}...` : value
          },
          axisPointer: {
            type: 'shadow'
          },
          axisLine: {show: true},
          axisTick: {show: true}
        },
        yAxis: {
          type: 'value',
          axisLabel: {
            formatter: value => Number(value).toFixed(2)
          },
          axisLine: {show: true},
          axisTick: {show: true},
          splitLine: {
            lineStyle: {
              type: 'dashed'
            }
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
          symbol: 'circle',
          symbolSize: 6,
          connectNulls: true,
          lineStyle: {
            width: 2,
            type: 'solid'
          },
          emphasis: {
            lineStyle: {
              width: 3
            }
          }
        }))
      }
    }

    // Lifecycle Hooks
    onMounted(async () => {
      await initChart()
      renderChart()
    })

    onBeforeUnmount(() => {
      if (resizeObserver.value) {
        resizeObserver.value.disconnect()
      }
      if (chart.value) {
        chart.value.dispose()
        chart.value = null
      }
    })

    // Watchers
    watch(
        [safeData, activeVariables],
        () => {
          if (initialized.value) {
            renderChart()
          } else {
            initChart().then(renderChart)
          }
        },
        {deep: true}
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
  font-size: 14px;
}
</style>