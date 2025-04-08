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
      validator: value => {
        const isValid = Array.isArray(value) && value.every(item => {
          return typeof item.taskId !== 'undefined' &&
              Object.values(item).some(v => typeof v === 'number')
        })
        console.log('[CURVE] Data validation:', isValid, value)
        return isValid
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
    const initialized = ref(false)
    const lastRenderTime = ref(0)
    const resizeObserver = ref(null)
    const isMounted = ref(true)

    const animationConfig = {
      duration: 800,
      easing: 'cubicInOut'
    }

    // Computed Properties
    const safeData = computed(() => {
      console.log('[DEBUG] Curve received data:', props.data)
      return (props.data || []).map(item => ({
        taskId: item.taskId || 'unknown',
        ...Object.fromEntries(
            Object.entries(item).filter(([k]) => k !== 'taskId')
        )
      }))
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
      // 显示调试信息
      console.log('[DEBUG] Active variables:', activeVariables.value)
      console.log('[DEBUG] Data validity:', safeData.value.length > 0)

      return !safeData.value.length ||
          activeVariables.value.length === 0 ||
          !activeVariables.value.some(v =>
              safeData.value.some(d => d[v] !== undefined)
          )
    })

    const emptyMessage = computed(() => {
      if (!safeData.value.length) return 'No data available'
      if (!activeVariables.value.length) return 'No active variables selected'
      return ''
    })

    // Methods
    const initChart = async () => {
      try {
        // 双重等待确保 DOM 就绪
        await nextTick()
        await new Promise(resolve => requestAnimationFrame(resolve))

        if (!container.value || !container.value.offsetParent) {
          console.warn('Chart container not ready')
          return false
        }

        // 清理旧实例
        if (chart.value) {
          chart.value.dispose()
          chart.value = null
        }

        // 防止重复初始化
        if (container.value._echarts_instance) {
          console.log('Reusing existing chart instance')
          chart.value = echarts.getInstanceByDom(container.value)
          return true
        }

        chart.value = echarts.init(container.value, null, {
          renderer: 'canvas',
          useDirtyRect: true // 启用脏矩形优化
        })
        initialized.value = true
        return true
      } catch (e) {
        console.error('Init failed:', e)
        return false
      }
    }

    const renderChart = async () => {
      if (!container.value) return

      try {
        let retries = 0
        while (retries < 3) {
          if (await initChart()) break
          await new Promise(r => setTimeout(r, 300 * (retries + 1)))
          retries++
        }

        if (!chart.value) return

        // 强制清理旧配置
        chart.value.clear()
        chart.value.setOption(getChartOption())
      } catch (e) {
        console.error('Render failed:', e)
      }
    }

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
        yAxis: {
          type: 'value',
          axisLabel: {
            formatter: value => {
              // 查找反向映射
              const varName = activeVariables.value.find(v =>
                  valueTypes.value[v] === 'string'
              )
              if (!varName) return Number(value).toFixed(2)

              const map = discreteValueMap.value[varName]
              const entry = Object.entries(map).find(([k, v]) => v === value)
              return entry ? entry[0] : value
            }
          }
        },
        series: activeVariables.value.map(varName => ({
          name: varName,
          type: 'line',
          data: safeData.value.map(d => {
            const rawValue = d[varName]
            if (valueTypes.value[varName] === 'string') {
              return getDiscreteValue(varName, rawValue)
            }
            return typeof rawValue === 'number' ? rawValue :
                typeof rawValue === 'string' ? parseFloat(rawValue) || 0 : 0
          }),
          // 新增平滑配置
          smooth: true,
          symbol: 'emptyCircle',
          symbolSize: 8,
          showSymbol: safeData.value.length < 50,
          connectNulls: true,
          lineStyle: {
            width: 2,
            shadowColor: 'rgba(0, 0, 0, 0.2)',
            shadowBlur: 8,
            shadowOffsetY: 5
          },
          areaStyle: {
            color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
              {offset: 0, color: 'rgba(64, 158, 255, 0.6)'},
              {offset: 1, color: 'rgba(64, 158, 255, 0.02)'}
            ])
          },
          emphasis: {
            lineStyle: {
              width: 3
            }
          },
          animationEasing: 'cubicOut',
          animationDurationUpdate: animationConfig.duration
        }))
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
      isMounted.value = true
      // 延迟初始化确保容器存在
      setTimeout(renderChart, 100)
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