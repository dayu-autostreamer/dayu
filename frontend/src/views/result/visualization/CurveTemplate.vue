<template>
  <div class="chart-container">
    <div ref="containerRef" :style="{ width: '100%', height: '100%' }"></div>
  </div>
</template>

<script>
import {onMounted, onBeforeUnmount, watch, ref} from 'vue'
import * as echarts from 'echarts'

export default {
  props: ['config', 'data', 'variableStates'],

  setup(props) {
    let chart = null

    const containerRef = ref(null)

    const initChart = async () => {
      await nextTick()

      if (!containerRef.value) return
      if (chart) chart.dispose()

      chart = echarts.init(containerRef.value)
      renderChart()

      window.addEventListener('resize', () => {
        chart?.resize()
      })
    }


    const renderChart = () => {
      if (!chart || !props.data?.length) return

      console.log('ECharts Render Data:', {
        taskIds: props.data?.map(d => d.taskId),
        seriesData: props.data?.map(d =>
            Object.fromEntries(
                Object.entries(d).filter(([k]) => k !== 'taskId')
            ))
      })

      const option = {
        tooltip: {
          trigger: 'axis'
        },
        legend: {
          data: Object.keys(props.data[0]).filter(k => k !== 'taskId')
        },
        grid: {
          left: '3%',
          right: '4%',
          bottom: '3%',
          containLabel: true
        },
        xAxis: {
          type: 'category',
          boundaryGap: false,
          data: props.data.map(d => d.taskId)
        },
        yAxis: {
          type: 'value'
        },

        series: props.data[0]
            ? Object.keys(props.data[0])
                .filter(k => k !== 'taskId')
                .map(varName => ({
                  name: varName,
                  type: 'line',
                  data: props.data.map(d => d[varName]),
                }))
            : []
      }

      chart?.setOption(option, true)
    }

    onMounted(initChart)
    onBeforeUnmount(() => {
      window.removeEventListener('resize', () => chart?.resize())
      chart?.dispose()
    })

    watch(() => [props.data, props.variableStates], () => {
      renderChart()
    }, {deep: true})

    return {containerRef}
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