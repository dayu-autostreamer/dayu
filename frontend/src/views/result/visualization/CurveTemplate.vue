<template>
  <div :id="'chart-'+config.id" class="chart-container"></div>
</template>

<script>
import { onMounted, onBeforeUnmount, watch } from 'vue'
import * as echarts from 'echarts'

export default {
  extends: {
    setup(props) {
      let chart = null

      const renderChart = () => {
        if (!props.data?.length) return

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
          series: Object.keys(props.data[0])
            .filter(k => k !== 'taskId' && props.variableStates[k])
            .map(varName => ({
              name: varName,
              type: 'line',
              data: props.data.map(d => parseFloat(d[varName]) || 0,
              smooth: true
            }))
        }

        chart?.setOption(option, true)
      }

      onMounted(() => {
        const container = document.getElementById(`chart-${props.config.id}`)
        if (container) {
          chart = echarts.init(container)
          renderChart()
          window.addEventListener('resize', () => chart?.resize())
        }
      })

      onBeforeUnmount(() => {
        if (chart) {
          window.removeEventListener('resize', () => chart.resize())
          chart.dispose()
        }
      })

      watch(() => props.data, renderChart, { deep: true })
      watch(() => props.variableStates, renderChart, { deep: true })
    }
  }
}
</script>

<style scoped>
.chart-container {
  width: 100%;
  height: 100%;
  min-height: 300px;
}
</style>