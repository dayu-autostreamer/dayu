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
import mitt from 'mitt'

const emitter = mitt()

export default {
  name: 'CurveTemplate',
  components: {PieChart},
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
      await nextTick();

      if (!container.value || !props.data?.length) {
        setTimeout(initChart, 100); // 自动重试机制
        return;
      }

      // 销毁旧实例
      if (chart.value) {
        chart.value.dispose();
        chart.value = null;
      }

      try {
        chart.value = echarts.init(container.value);
        chart.value.setOption(option);
      } catch (e) {
        console.error('ECharts init retrying...');
        console.log(e)
        setTimeout(initChart, 300);
      }
    };

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

// CurveTemplate.vue 中的 option 配置
        const option = {
          // 强制定义坐标轴类型
          xAxis: {
            type: 'category',
            axisPointer: {
              type: 'shadow' // 添加默认指针类型
            },
            data: safeData.value.map(d => d.taskId)
          },
          yAxis: {
            type: 'value',
            axisLine: {show: true}, // 显式定义轴线
            axisTick: {show: true}  // 显式定义刻度
          },
          series: activeVariables.value.map(varName => ({
            name: varName,
            type: 'line',
            connectNulls: true, // 处理空值
            symbol: 'circle',   // 明确符号类型
            data: safeData.value.map(d => d[varName]),
            // 添加空数据保护
            lineStyle: {
              type: 'solid'
            },
            emphasis: {
              disabled: false // 启用高亮状态
            }
          }))
        };

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
}
</style>