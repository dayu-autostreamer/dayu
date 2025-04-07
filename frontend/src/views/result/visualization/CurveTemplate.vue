<template>
  <div class="chart-container">
    <div ref="containerRef" :style="{ width: '100%', height: '100%' }"></div>
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
    let observer = null


    const initChart = async () => {
      await nextTick()

      if (!container.value) return
      if (chart.value) chart.value.dispose()

      chart.value = echarts.init(container.value)
      renderChart()

      observer = new ResizeObserver(() => {
        chart.value?.resize()
      })
      observer.observe(container.value)
    }


    const renderChart = () => {
      if (!chart.value || !props.data?.length) return

      try {
        const option = {
          xAxis: {
            type: 'category',
            data: props.data.map(d => d.taskId)
          },
          yAxis: {type: 'value'},
          series: Object.keys(props.data[0])
              .filter(k => k !== 'taskId')
              .map(varName => ({
                name: varName,
                type: 'line',
                data: props.data.map(d => d[varName]),
                showSymbol: false
              }))
        }

        chart.value.setOption(option, true)
        console.log('[ECharts] Render success')
      } catch (e) {
        console.error('[ECharts] Render error:', e)
      }
    }

    onMounted(async () => {
      await initChart()
      if (!chart.value) setTimeout(initChart, 100)
    })

    onBeforeUnmount(() => {
      observer?.disconnect()
      chart.value?.dispose()
    })

    watch(() => props.data, () => {
      renderChart()
    }, {deep: true})

    return {container}
  }
}
</script>

<style scoped>
.chart-container {
  width: 100%;
  height: calc(100% - 40px);
  padding: 12px;
}

.chart-wrapper {
  width: 100%;
  height: 100%;
  min-height: 300px;
}
</style>