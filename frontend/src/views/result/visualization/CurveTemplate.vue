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
        type: 'curve',
        variables: []
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

    console.log('CurveTemplate Mounted:', {
      config: toRaw(props.config),
      variables: toRaw(props.variableStates),
      initialData: toRaw(props.data)
    })

    // Refs
    const chart = ref(null)
    const container = ref(null)
    const initialized = ref(false)
    const lastRenderTime = ref(0)
    const resizeObserver = ref(null)
    const isMounted = ref(true)
    const forceUpdate = ref(0)

    let renderRetryCount = 0
    const MAX_RETRIES = 8

    const animationConfig = {
      duration: 800,
      easing: 'quarticInOut'
    }

    // Computed Properties
    const safeData = computed(() => {
      return (props.data || []).map(item => {
        const cleanItem = {taskId: item.taskId}

        props.config.variables?.forEach(varName => {
          const rawValue = item[varName]
          // 转换无效值为0
          cleanItem[varName] = rawValue !== null && rawValue !== undefined ?
              rawValue : 0
        })

        return cleanItem
      })
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
      const hasData = safeData.value.length > 0
      const hasActiveVars = activeVariables.value.length > 0
      const hasValidData = hasData && activeVariables.value.some(v =>
          safeData.value.some(d => d[v] !== undefined)
      )

      console.log('Empty State Check:', {
        hasData,
        hasActiveVars,
        hasValidData
      })

      return !hasValidData
    })

    const emptyMessage = computed(() => {
      if (!safeData.value.length) return 'No data available'
      if (!activeVariables.value.length) return 'No active variables selected'
      return ''
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
        if (renderRetryCount++ > MAX_RETRIES) {
          console.warn('Max retries reached')
          return
        }

        if (!(await initChart())) {
          setTimeout(renderChart, 300 * Math.pow(2, renderRetryCount)) // 指数退避
          return
        }

        // 强制清除旧实例
        if (chart.value) {
          chart.value.dispose()
          chart.value = null
        }

        chart.value = echarts.init(container.value)
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


    const valueTypes = computed(() => {
      const types = {}
      activeVariables.value.forEach(varName => {
        const sampleValue = safeData.value[0]?.[varName]
        types[varName] = typeof sampleValue
      })
      return types
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
      if (activeVariables.value.length === 0 || safeData.value.length === 0) {
        return {}
      }

      const inferAxisType = (values) => {
        const sample = values[0]
        return typeof sample === 'string' ? 'category' : 'value'
      }

      const yAxisConfig = activeVariables.value.map(varName => ({
        type: inferAxisType(safeData.value.map(d => d[varName])),
        name: varName,
        alignTicks: true,
        axisLabel: {
          formatter: value => {
            if (valueTypes.value[varName] === 'string') {
              const entry = Object.entries(discreteValueMap.value[varName])
                  .find(([k, v]) => v === value)
              return entry ? entry[0] : value
            }
            return Number(value).toFixed(2)
          }
        }
      }))

      const seriesConfig = activeVariables.value.map(varName => {
        const values = safeData.value.map(d => d[varName])

        return {
          name: varName,
          type: 'line',
          yAxisIndex: activeVariables.value.indexOf(varName),
          data: values.map(v => {
            if (v === undefined) return null
            return valueTypes.value[varName] === 'string'
                ? getDiscreteValue(varName, v)
                : Number(v)
          }),
          // 新增：空数据跳过渲染
          connectNulls: false,
          // 新增：优化渲染性能
          progressive: 200,
          animation: values.length < 100
        }
      })


      return {
        // 新增动画配置
        animation: true,
        animationDuration: animationConfig.duration,
        animationEasing: animationConfig.easing,
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
        yAxis: yAxisConfig,
        series: seriesConfig
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
    watch(() => props.data, () => {
      if (isMounted.value) {
        renderChart()
      }
    }, {deep: true, flush: 'post'})

    watch(activeVariables, (newVal) => {
      console.log('Active Variables Changed:', newVal)
      console.log('Current Variable States:', toRaw(props.variableStates))
    }, {deep: true})

    watch(safeData, (newVal) => {
      console.log('Chart Data Updated:', {
        data: newVal,
        keys: newVal.length > 0 ? Object.keys(newVal[0]) : []
      })
    }, {deep: true})

    watch(() => props.variableStates, () => {
      console.log('VariableStates Changed:', toRaw(props.variableStates))
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
  color: var(--el-text-color-secondary);
  z-index: 10;
}

.empty-state p {
  margin-top: 10px;
  font-size: 14px;
}
</style>