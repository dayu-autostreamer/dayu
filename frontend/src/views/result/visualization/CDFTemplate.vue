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
import {ref, computed, onMounted, onBeforeUnmount, watch, nextTick, toRaw} from 'vue'
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
    const forceUpdate = ref(0)

    let renderRetryCount = 0

    const animationConfig = {
      duration: 800,
      easing: 'quarticInOut'
    }

    const cleanChart = () => {
      if (chart.value) {
        chart.value.dispose()
        chart.value = null
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

        // 收集所有有效数值
        const allValues = props.data
            .map(d => d[varName])
            .filter(v => v !== undefined && v !== null && !isNaN(v))
            .map(Number)
            .sort((a, b) => a - b)
        if (allValues.length === 0) return

        // 生成CDF点

        const n = allValues.length

        const uniqueValues = [...new Set(allValues)]
        const cdfPoints = uniqueValues.map(value => ({
          value,
          probability: allValues.filter(v => v <= value).length / n
        }))

        result[varName] = cdfPoints
      })

      return result
    })

    const availableVariables = computed(() => {
      if (!safeData.value.length) return []
      return Object.keys(safeData.value[0])
          .filter(k => k !== 'taskId' && props.config.variables?.includes(k))
    })

    const activeVariables = computed(() => {
      return props.config.variables?.filter(varName =>
          props.variableStates[varName] !== false
      ) || []
    })


    const showEmptyState = computed(() => {
      const hasData = Object.values(safeData.value).some(arr => arr?.length > 0)
      const hasActiveVars = activeVariables.value.length > 0
      const hasValidData = hasData && activeVariables

      return !hasValidData
    })

    const emptyMessage = computed(() => {
      if (props.data.length === 0) return 'No data available'
      if (activeVariables.value.length === 0) return 'No active variables selected'

      const hasInvalidData = Object.values(safeData.value).every(arr => arr.length === 0)
      return hasInvalidData ? 'No valid numeric data available' : ''
    })

    // Methods
    const initChart = async () => {
      try {
        // 三重等待确保 DOM 就绪
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

        // 检查容器可见性
        const style = window.getComputedStyle(container.value)
        if (style.display === 'none' || style.visibility === 'hidden') {
          console.warn('Chart container is hidden')
          return false
        }

        // 清理旧实例
        if (chart.value) {
          chart.value.dispose()
          chart.value = null
        }

        chart.value = echarts.init(container.value, null, {
          renderer: 'canvas',
          useDirtyRect: true
        })

        // 标记容器状态
        container.value.dataset.chartReady = 'true'
        return true
      } catch (e) {
        console.error('Chart init failed:', e)
        return false
      }
    }


    const renderChart = async () => {
      try {

        if (!chart.value) {
          const success = await initChart()
          if (!success) return
        }

        chart.value.setOption(getChartOption())

        // 添加视觉连续性
        chart.value.dispatchAction({
          type: 'downplay',
          seriesIndex: 'all'
        })
        chart.value.dispatchAction({
          type: 'highlight',
          seriesIndex: 0
        })

        renderRetryCount = 0 // 重置计数器
      } catch (e) {
        console.error('Render failed:', e)
      }
    }

    const observer = new MutationObserver(() => {
      forceUpdate.value++
    })

    const discreteValueMap = ref({})
    const getDiscreteValue = (varName, value) => {
      if (!discreteValueMap.value[varName]) {
        discreteValueMap.value[varName] = {}
      }
      const map = discreteValueMap.value[varName]
      if (!(value in map)) {
        map[value] = Object.keys(map).length
      }
      return map[value]
    }

    const getChartOption = () => {
      const hasData = Object.values(safeData.value).some(arr => arr?.length > 0)
      if (!hasData) {
        // 返回完全空配置清除坐标轴
        return {
          xAxis: {show: false},
          yAxis: {show: false},
          series: []
        }
      }
      if (activeVariables.value.length === 0 || safeData.value.length === 0) {
        return {
          xAxis: {show: false},
          yAxis: {show: false},
          series: []
        }
      }

      const series = []

      Object.entries(safeData.value).forEach(([varName, points]) => {
        series.push({
          name: varName,
          type: 'line',
          data: points.map(p => [p.value, p.probability]),
          smooth: true,
          areaStyle: {
            opacity: 0.1
          }
        })
      })

      return {
        tooltip: {
          trigger: 'item',
          formatter: params => {
            return `${params.seriesName}<br/>
            Value: ${params.value[0].toFixed(2)}<br/>
            Probability: ${(params.value[1] * 100).toFixed(1)}%`
          }
        },
        xAxis: {
          name: props.config.x_axis,
          nameLocation: 'center',
          nameGap: 25,
          type: 'value',
          min: 'dataMin',
          max: 'dataMax'
        },
        yAxis: {
          name: props.config.y_axis,
          type: 'value',
          min: 0,
          max: 1,
          axisLabel: {
            formatter: value => `${(value * 100).toFixed(0)}%`
          }
        },
        series,
        legend: {
          data: Object.keys(safeData.value)
        }
      }
    }

    const smoothUpdate = () => {
      if (!chart.value) return
      chart.value.setOption({
        series: activeVariables.value.map(varName => ({
          data: safeData.value.map(d => d[varName])
        }))
      }, {
        replaceMerge: ['series'],
        notMerge: false
      })
    }

    // Lifecycle Hooks
    onMounted(() => {
      if (!showEmptyState.value) {
        renderChart()
      }
      if (container.value) {
        observer.observe(container.value, {
          attributes: true,
          attributeFilter: ['style', 'class']
        })
      }
      setTimeout(renderChart, 300)
    })

    onBeforeUnmount(() => {
      isMounted.value = false
      if (chart.value) {
        chart.value.dispose()
        chart.value = null
      }
      if (resizeObserver.value) {
        resizeObserver.value.disconnect()
      }
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
      }
    }, {deep: true, flush: 'post'})

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