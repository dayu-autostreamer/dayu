<template>
  <div class="chart-container">
    <div ref="container" class="chart-wrapper"></div>
    <div v-if="!props.data?.length" class="empty-state">
      <el-icon :size="40">
        <PieChart/>
      </el-icon>
      <p>No data available</p>
    </div>
  </div>
</template>

<script>
import {onMounted, onBeforeUnmount, watch, ref, nextTick} from 'vue'
import * as echarts from 'echarts'

export default {
  props: ['config', 'data', 'variableStates'],

  setup(props) {
    const chart = ref(null)
    const container = ref(null)
    const initialized = ref(false)


    const initChart = async () => {
      await nextTick()
      if (!container.value) {
        console.error('Chart container not found')
        return
      }

      if (chart.value) {
        chart.value.dispose()
        chart.value = null
      }

      try {
        chart.value = echarts.init(container.value)
        initialized.value = true
        console.log('ECharts initialized for', props.config.id)
        renderChart()
      } catch (e) {
        console.error('ECharts init failed:', e)
      }
    }


    const renderChart = () => {
      if (!chart.value || !props.data?.length) return

      try {
        const option = {
          xAxis: {
            type: 'category',
            data: props.data.map(d => d.taskId.toString())
          },
          yAxis: {type: 'value'},
          series: Object.keys(props.data[0])
              .filter(k => k !== 'taskId' && props.variableStates[k])
              .map(varName => ({
                name: varName,
                type: 'line',
                data: props.data.map(d => Number(d[varName])),
                smooth: true,
                showSymbol: props.data.length < 20
              })),
          tooltip: {
            trigger: 'axis',
            formatter: params => {
              return `${params[0].name}<br/>` +
                  params.map(p =>
                      `${p.marker} ${p.seriesName}: ${p.value.toFixed(2)}`
                  ).join('<br>')
            }
          }
        }

        chart.value.setOption(option, {notMerge: true})
        console.log('Chart rendered with', props.data.length, 'data points')
      } catch (e) {
        console.error('Render error:', e)
      }
    }

    onMounted(async () => {
      await initChart()
      if (!initialized.value) {
        setTimeout(initChart, 300)
      }
    })

    onBeforeUnmount(() => {
      chart.value?.dispose()
    })


    watch(() => [...props.data, props.variableStates], () => {
      if (initialized.value) renderChart()
    }, {deep: true})

    return {container}
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

.chart-container::after {
  content: '';
  display: block;
  padding-bottom: 56.25%;
}

.chart-wrapper {
  position: absolute;
  top: 0;
  left: 0;
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