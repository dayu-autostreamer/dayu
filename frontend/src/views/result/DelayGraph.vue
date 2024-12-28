<template>
  <div>
    <h3>End-to-End Task Delay</h3>
  </div>

  <div id="line-chart-delay" style="width: 24vw; height: 32vh;"></div>
<!--  <br>-->
<!--  <div>-->
<!--    <div style="text-align: center; font-size: medium;">-->
<!--      <p>{{ prompt }}</p>-->
<!--    </div>-->

<!--  </div>-->
</template>

<script>

import * as echarts from 'echarts';

export default {
  props: {
    data: {
      type: Object,
    },
    prompt: {
      type: String,
      default: '时延说明'
    }
  },
  data() {
    return {
      chartData: [],
      chart: null
    }
  },
  mounted() {

    // 初始化echarts实例
    this.chart = echarts.init(document.getElementById('line-chart-delay'));
    // 调用加载数据的方法
    this.loadData();
    // 定时更新数据
    setInterval(this.loadData, 5000); // 每5秒更新一次
  },
  watch: {
    data: {
      handler: function (newVal, oldVal) {
        this.loadData();
      },
      deep: true
    }
  },
  methods: {
    loadData() {
      this.chartData = this.data;
      if (!this.chartData) {
        return;
      }
      // 调用渲染图表的方法
      this.renderChart();
    },

    renderChart() {
      const xAxisData = this.chartData.map(item => item.taskId);
      const seriesData = this.chartData.map(item => item.delay);

      const option = {
        // title: {
        //     text: '处理时延CDF图'
        // },
        xAxis: {
          type: 'category',
          name: 'Task Index',
          data: xAxisData,
          nameLocation: 'middle', // 使轴名称居中显示
          nameGap: 25, // 与轴的距离
          axisLine: {
            lineStyle: {
              color: '#999' // x轴线颜色
            }
          },
          splitLine: {
            lineStyle: {
              type: 'dashed' // 分隔线类型
            }
          }
        },
        yAxis: {
          type: 'value',
          name: 'Task Delay',
          nameLocation: 'middle', // 使轴名称居中显示
          nameGap: 30, // 与轴的距离
          axisLine: {
            lineStyle: {
              color: '#999' // x轴线颜色
            }
          },
          splitLine: {
            lineStyle: {
              type: 'dashed' // 分隔线类型
            }
          }
        },
        grid: {
          left: '5%', // 调整左边距
          right: '5%', // 调整右边距
          bottom: '10%', // 调整底边距
          containLabel: true // 包含标签在内
        },
        series: [{
          data: seriesData,
          type: 'line',
          width: 2 // 折线宽度
        }]
      };

      this.chart.setOption(option);
    }
  }
};
</script>


<style scoped>
#line-chart {
  margin: 0 auto;
  width: 100%;
  height: 100%;
}
</style>
  