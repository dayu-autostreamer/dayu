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

    const animationConfig = {
      duration: 800,
      easing: 'cubicInOut'
    }

    // Computed Properties
    const safeData = computed(() => {
      try {
        return (props.data || []).map(item => {
          const processedItem = {taskId: ''};
          Object.entries(item).forEach(([key, value]) => {
            processedItem[key] = typeof value === 'number' ? value :
                !isNaN(value) ? parseFloat(value) : 0;
          });
          return processedItem;
        });
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
        } else {
          container.value.style.height = '100%'
          container.value.style.width = '100%'
        }

        if (chart.value) {
          chart.value.dispose()
          chart.value = null
        }

        chart.value = echarts.init(container.value)
        initialized.value = true

        // 修改：添加带动画的resize
        resizeObserver.value = new ResizeObserver(() => {
          chart.value?.resize({animation: animationConfig})
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
        // 修改：优化setOption参数
        if (chart.value.getOption()) {
          chart.value.setOption(option, {
            replaceMerge: ['series', 'xAxis', 'yAxis'],
            notMerge: false
          })
        } else {
          chart.value.setOption(option)
        }
        lastRenderTime.value = Date.now()
      } catch (e) {
        console.error('Chart render error:', e)
      }
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
        () => props.data,
        (newVal, oldVal) => {
          if (JSON.stringify(newVal) !== JSON.stringify(oldVal)) {
            console.log('Curve data changed:', newVal)
            nextTick().then(() => {
              renderChart()
            })
          }
        },
        {deep: true, immediate: true}
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