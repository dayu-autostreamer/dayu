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
        type: 'cdf',
        variables: [],
        x_axis: '',
        y_axis: ''
      })
    },
    data: {
      type: Array,
      required: true,
      validator: value => {
        return Array.isArray(value) && value.every(item =>
            item.taskId !== undefined
        )
      }
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
    const resizeObserver = ref(null)
    const isMounted = ref(true)

    const cleanChart = () => {
      if (chart.value) {
        chart.value.dispose()
        chart.value = null
      }

      if (container.value) {
        container.value.innerHTML = ''
      }
    }

    // Computed Properties
    const safeData = computed(() => {
      const result = {}
      const currentVariableStates = props.variableStates || {}
      if (!props.config.variables?.length) {
        console.warn('No variables defined in config')
        return result
      }

      props.config.variables?.forEach(varName => {
        if (currentVariableStates[varName] !== true) return
        const allValues = props.data
            .map(d => d[varName])
            .filter(v => v !== undefined && v !== null && !isNaN(v))
            .map(Number)
            .sort((a, b) => a - b)
        if (allValues.length === 0) return

        const n = allValues.length
        const uniqueValues = [...new Set(allValues)]
        result[varName] = uniqueValues.map(value => ({
          value,
          probability: allValues.filter(v => v <= value).length / n
        }))
      })

      return result
    })

    const activeVariables = computed(() => {
      return props.config.variables?.filter(varName =>
          props.variableStates[varName] === true
      ) || []
    })

    const showEmptyState = computed(() => {
      const hasData = Object.values(safeData.value).some(arr => arr?.length > 0)
      const anyActive = activeVariables.value.length > 0
      return !(hasData && anyActive)
    })

    const emptyMessage = computed(() => {
      if (props.data.length === 0) return 'No data available'
      if (activeVariables.value.length === 0) return 'No active variables selected'
      const hasInvalidData = Object.values(safeData.value).every(arr => (arr?.length || 0) === 0)
      return hasInvalidData ? 'No valid numeric data available' : ''
    })

    // Methods
    const initChart = async () => {
      try {
        await nextTick()
        if (!container.value) return false
        const isVisible = () => {
          const rect = container.value.getBoundingClientRect()
          return !(rect.width === 0 || rect.height === 0)
        }
        let checks = 0
        while (checks < 10) {
          if (isVisible()) break
          await new Promise(r => setTimeout(r, 50))
          checks++
        }
        if (!isVisible()) {
          console.warn('Container remains invisible after retries')
          return false
        }
        const style = window.getComputedStyle(container.value)
        if (style.display === 'none' || style.visibility === 'hidden') {
          console.warn('Chart container is hidden')
          return false
        }
        if (chart.value) {
          chart.value.dispose()
          chart.value = null
        }
        chart.value = echarts.init(container.value, null, {
          renderer: 'canvas',
          useDirtyRect: true
        })
        container.value.dataset.chartReady = 'true'
        return true
      } catch (e) {
        console.error('Chart init failed:', e)
        return false
      }
    }

    const getChartOption = () => {
      const hasData = Object.values(safeData.value).some(arr => arr?.length > 0)
      if (!hasData || activeVariables.value.length === 0 || Object.keys(safeData.value).length === 0) {
        return {
          xAxis: { type: 'value', show: false, splitLine: { show: false } },
          yAxis: { type: 'value', show: false, splitLine: { show: false } },
          series: [],
          grid: { containLabel: true }
        }
      }
      const series = []
      Object.entries(safeData.value).forEach(([varName, points]) => {
        series.push({
          name: varName,
          type: 'line',
          data: points.map(p => [p.value, p.probability]),
          smooth: true,
          areaStyle: {opacity: 0.1}
        })
      })
      return {
        tooltip: {
          trigger: 'item',
          formatter: params => {
            return `${params.seriesName}<br/>
            数值: ${params.value[0].toFixed(2)}<br/>
            概率: ${(params.value[1] * 100).toFixed(1)}%`
          }
        },
        xAxis: {
          name: props.config.x_axis,
          nameLocation: 'center',
          nameGap: 25,
          type: 'value',
          min: 'dataMin',
          max: 'dataMax',
          show: true
        },
        yAxis: {
          name: props.config.y_axis,
          type: 'value',
          min: 0,
          max: 1,
          axisLabel: {formatter: value => `${(value * 100).toFixed(0)}%`},
          show: true
        },
        grid: { containLabel: true },
        series,
        legend: {data: Object.keys(safeData.value)}
      }
    }

    const renderChart = async () => {
      try {
        // 空状态不渲染图表，并确保已清理
        if (showEmptyState.value) {
          cleanChart()
          return
        }
        if (!chart.value) {
          const success = await initChart()
          if (!success) return
        }
        // 使用 notMerge 强制覆盖，避免首次空图隐藏坐标轴导致后续保持隐藏
        chart.value.setOption(getChartOption(), true)
        chart.value.dispatchAction({
          type: 'downplay',
          seriesIndex: 'all'
        })
        chart.value.dispatchAction({
          type: 'highlight',
          seriesIndex: 0
        })
      } catch (e) {
        console.error('Render failed:', e)
      }
    }

    // 在样式变化/尺寸变化时触发 resize
    const mutationObserver = new MutationObserver(() => {
      if (chart.value) {
        chart.value.resize()
      }
    })

    const setupResizeHandling = () => {
      if (!container.value) return
      // ResizeObserver 监听容器尺寸
      try {
        resizeObserver.value = new ResizeObserver(() => {
          if (chart.value) {
            chart.value.resize()
          }
        })
        resizeObserver.value.observe(container.value)
      } catch (e) {
        // 某些环境可能没有 ResizeObserver
        window.addEventListener('resize', resizeChart)
      }
      // 监听样式/类变化
      mutationObserver.observe(container.value, {
        attributes: true,
        attributeFilter: ['style', 'class']
      })
    }

    const resizeChart = () => {
      if (chart.value) chart.value.resize()
    }

    // Lifecycle Hooks
    onMounted(() => {
      if (!showEmptyState.value) {
        renderChart()
      }
      if (container.value) {
        setupResizeHandling()
      }
      // 仅在非空状态下尝试延迟渲染
      setTimeout(() => {
        if (!showEmptyState.value) {
          renderChart()
        }
      }, 300)
    })

    onBeforeUnmount(() => {
      isMounted.value = false
      if (chart.value) {
        chart.value.dispose()
        chart.value = null
      }
      if (resizeObserver.value) {
        try { resizeObserver.value.disconnect() } catch {}
      }
      try { mutationObserver.disconnect() } catch {}
      window.removeEventListener('resize', resizeChart)
    })

    // Watchers
    watch(showEmptyState, (newVal) => {
      if (newVal) {
        cleanChart()
      } else {
        renderChart()
      }
    })

    watch(() => props.data, () => {
      if (isMounted.value && !showEmptyState.value) {
        renderChart()
      } else if (isMounted.value && showEmptyState.value) {
        cleanChart()
      }
    }, {deep: true, flush: 'post'})

    // 当变量选择状态变化时也需要重绘
    watch(() => props.variableStates, () => {
      if (isMounted.value && !showEmptyState.value) {
        renderChart()
      } else if (isMounted.value && showEmptyState.value) {
        cleanChart()
      }
    }, {deep: true})

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
  color: var(--el-text-color-secondary, #909399);
  color: var(--el-text-color-secondary);
}

.empty-state p {
  margin-top: 10px;
  font-size: 14px;
}
</style>