<template>
  <div class="home-container layout-pd">
    <el-row :gutter="15" class="home-card-two mb15 toggleSource">
      <el-col :xs="24" :sm="24" :md="20" :lg="20" :xl="20">
        <div class="home-card-item">
          <div style="height: 100%">
            <div class="flex-margin flex w100">
              <div class="flex-auto" style="font-weight: bold">

                Choose Datasource: &nbsp; &nbsp;

                <el-select v-model="selectedDataSource" placeholder="Please choose datasource"
                           style="width: 40%; font-weight: normal">
                  <el-option v-for="item in dataSourceList" :key="item.id" :label="item.label"
                             :value="item.id">
                  </el-option>
                </el-select>

              </div>
            </div>
          </div>
        </div>
      </el-col>
      <el-col :xs="24" :sm="24" :md="4" :lg="4" :xl="4">
        <div class="home-card-item">
          <div style="height: 100%">
            <div class="flex-margin flex w100">
              <div class="flex-auto">
                <div style="display: flex; justify-content: center; align-items: center;">
                  <el-button type="primary" class="export-button" @click="exportTaskLog" style="font-weight: bold">
                    Export Log
                  </el-button>
                </div>

              </div>
            </div>
          </div>
        </div>
      </el-col>
    </el-row>
    <el-row :gutter="15" class="home-card-two mb15">
      <el-col :xs="24" :sm="24" :md="8" :lg="8" :xl="8">
        <div class="home-card-item">
          <div style="height: 100%">
            <div class="flex-margin flex w100">
              <div class="flex-auto">
                <DataVisualize :data="filteredVisualizeData"/>
              </div>
            </div>
          </div>
        </div>
      </el-col>
      <el-col :xs="24" :sm="24" :md="8" :lg="8" :xl="8">
        <div class="home-card-item">
          <div style="height: 100%">
            <div class="flex-margin flex w100">
              <div class="flex-auto">
                <ResultGraph :data="filteredResultData"/>
              </div>
            </div>
          </div>
        </div>
      </el-col>
      <el-col :xs="24" :sm="24" :md="8" :lg="8" :xl="8">
        <div class="home-card-item">
          <div style="height: 100%">
            <div class="flex-margin flex w100">
              <div class="flex-auto">
                <DelayGraph :data="filteredDelayData"/>
              </div>
            </div>
          </div>
        </div>
      </el-col>
    </el-row>

  </div>
</template>

<script>


import {reactive} from 'vue'
import DataVisualize from './DataVisualize.vue'
import ResultGraph from './ResultGraph.vue'
import DelayGraph from './DelayGraph.vue'


export default {
  components: {
    DataVisualize,
    ResultGraph,
    DelayGraph,

  },
  data() {
    return {
      // 子组件注入的数据
      visualizeData: null,
      resultData: null,
      delayData: null,

      // 数据源列表
      dataSourceList: null,
      // [
      //     {
      //         id: '1',
      //         label: '数据源1'
      //     }, {
      //         id: '2',
      //         label: '数据源2'
      //     }
      // ],

      // 选择的数据源
      selectedDataSource: null,

      visualizing_prompt: null,
      result_title_prompt: null,
      result_text_prompt: null,
      delay_text_prompt: null,

      bufferedTaskCache: {},
      // {
      // 测试用fake数据
      // "1": [
      //             {
      //                 taskId: "1",
      //                 result: "8",
      //                 delay: "0.5"
      //             },
      //             {
      //                 taskId: "2",
      //                 result: "9",
      //                 delay: "0.6"
      //             },
      //             {
      //                 taskId: "3",
      //                 result: "10",
      //                 delay: "0.7"
      //             },
      //             {
      //                 taskId: "4",
      //                 result: "11",
      //                 delay: "0.8"
      //             },
      //             {
      //                 taskId: "5",
      //                 result: "12",
      //                 delay: "0.9"
      //             },
      //             {
      //                 taskId: "6",
      //                 result: "13",
      //                 delay: "1.0"
      //             },
      //             {
      //                 taskId: "7",
      //                 result: "14",
      //                 delay: "1.1"
      //             },
      //             {
      //                 taskId: "8",
      //                 result: "15",
      //                 delay: "1.2"
      //             },
      //             {
      //                 taskId: "9",
      //                 result: "16",
      //                 delay: "1.3"
      //             },
      //             {
      //                 taskId: "10",
      //                 result: "17",
      //                 delay: "1.4"
      //             }
      //         ],
      //     "2": [
      //             {
      //                 taskId: "1",
      //                 result: "8",
      //                 delay: "0.5"
      //             },
      //             {
      //                 taskId: "2",
      //                 result: "9",
      //                 delay: "0.6"
      //             },
      //             {
      //                 taskId: "3",
      //                 result: "10",
      //                 delay: "0.7"
      //             },
      //             {
      //                 taskId: "4",
      //                 result: "11",
      //                 delay: "0.8"
      //             },
      //             {
      //                 taskId: "5",
      //                 result: "12",
      //                 delay: "0.9"
      //             },
      //             {
      //                 taskId: "6",
      //                 result: "13",
      //                 delay: "1.0"
      //             },
      //             {
      //                 taskId: "7",
      //                 result: "14",
      //                 delay: "1.1"
      //             },
      //             {
      //                 taskId: "8",
      //                 result: "15",
      //                 delay: "1.2"
      //             },
      //             {
      //                 taskId: "9",
      //                 result: "16",
      //                 delay: "1.3"
      //             },
      //             {
      //                 taskId: "10",
      //                 result: "17",
      //                 delay: "1.4"
      //             }
      //         ]
      // },

      maxBufferedTaskCacheSize: 20,
    }
  },

  methods: {


    generateRandomJPG() {
      if (!this.selectedDataSource) {
        return;
      }
      // 创建一个 canvas 元素
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');

      // 设置 canvas 尺寸
      canvas.width = 320;
      canvas.height = 240;

      // 生成随机颜色
      const randomColor = `rgb(${Math.floor(Math.random() * 256)}, ${Math.floor(Math.random() * 256)}, ${Math.floor(Math.random() * 256)})`;

      // 绘制随机颜色的矩形
      ctx.fillStyle = randomColor;
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      // 将 canvas 转换为 data URL
      const dataURL = canvas.toDataURL('image/jpeg');

      // 更新组件数据，渲染随机图像
      // this.visualizeData = dataURL;
      return dataURL;
    },

    generateRandomData(sourceId) {
      if (!sourceId) {
        return;
      }
      if (!this.bufferedTaskCache[sourceId] || this.bufferedTaskCache[sourceId].length === 0) {
        this.bufferedTaskCache[sourceId] = [];
        // generate first 1 task
        this.bufferedTaskCache[sourceId].push({
          taskId: "1",
          result: Math.floor(Math.random() * 100),
          delay: Math.random().toFixed(2),
          visualize: this.generateRandomJPG()
        });
      } else {
        // generate next task
        const nextTaskId = (parseInt(this.bufferedTaskCache[sourceId][this.bufferedTaskCache[sourceId].length - 1].taskId) + 2).toString();
        this.bufferedTaskCache[sourceId].push({
          taskId: nextTaskId,
          result: Math.floor(Math.random() * 100),
          delay: Math.random().toFixed(2),
          visualize: this.generateRandomJPG()
        });
        if (this.bufferedTaskCache[sourceId].length > this.maxBufferedTaskCacheSize) {
          this.bufferedTaskCache[sourceId].shift();
        }
      }
    },
    generateFakeDataForAllSources() {
      this.dataSourceList.forEach((item) => {
        this.generateRandomData(item.id);
      });
    },

    // =========================== 与后端对接
    // 1. 获取数据源，仅初始化时获取一次
    getDataSourceList() {

      // for test
      // this.dataSourceList = [
      //     {
      //         id: 'datasource1',
      //         label: '数据源1'
      //     },
      //     {
      //         id: 'datasource2',
      //         label: '数据源2'
      //     }
      // ];
      // return;

      // "/source_list"
      // 后端返回json格式
      // return [
      //     {
      //         id: 'datasource1',
      //         label: '数据源1'
      //     },
      //     {
      //         id: 'datasource2',
      //         label: '数据源2'
      //     }
      // ];

      //  如果没有任务，返回空列表，表示现在没有开启的数据源

      fetch('/api/source_list')
          .then(response => response.json())
          .then(data => {
            if (!data || data.length === 0) {
              return;
            }
            this.dataSourceList = data;
            console.log(data);
            // initialize buffered cache
            this.dataSourceList.forEach((item) => {
              this.bufferedTaskCache[item.id] = [];
            });
          });

    },


    // 2. 获取最新结果数据

    // "/task_result"
    // 后端返回json格式
    // return {
    //     'datasource1': [
    //         {
    //             taskId: "1",
    //             result: "8",
    //             delay: "0.5",
    //             visualize: "data:image/jpg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/4gIoSUNDX1BST0ZJTEUAAQEAAAIYAAAAAAQwAABtbnRyUkdCIFhZWiAAAAAAAAAAAAAAAABhY3NwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAA9tYAAQAAAADTLQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAlkZXNjAAAA8AAAAHRyWFlaAAPsxpwBwiKNIuGxlOz0WhhpoWCMcALb/iyiLCTt8l10f//Z",
    //         },
    //         {
    //             taskId: "2",
    //             result: "9",
    //             delay: "0.6",
    //             visualize: "data:image/jpg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/4gIoSUNDX1BST0ZJTEUAAQEAAAIYAAAAAAQwAABtbnRyUkdCIFhZWiAAAAAAAAAAAAAAAABhY3NwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAA9tYAAQAAAADTLQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAlkZXNjAAAA8AAAAHRyWFlaAAPsxpwBwiKNIuGxlOz0WhhpoWCMcALb/iyiLCTt8l10f//Z",
    //         }],
    //     'datasource2': [
    //         {
    //             taskId: "1",
    //             result: "8",
    //             delay: "0.5",
    //             visualize: "data:image/jpg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/4gIoSUNDX1BST0ZJTEUAAQEAAAIYAAAAAAQwAABtbnRyUkdCIFhZWiAAAAAAAAAAAAAAAABhY3NwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAA9tYAAQAAAADTLQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAlkZXNjAAAA8AAAAHRyWFlaAAPsxpwBwiKNIuGxlOz0WhhpoWCMcALb/iyiLCTt8l10f//Z",
    //         },
    //         {
    //             taskId: "2",
    //             result: "9",
    //             delay: "0.6",
    //             visualize: "data:image/jpg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/4gIoSUNDX1BST0ZJTEUAAQEAAAIYAAAAAAQwAABtbnRyUkdCIFhZWiAAAAAAAAAAAAAAAABhY3NwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAA9tYAAQAAAADTLQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAlkZXNjAAAA8AAAAHRyWFlaAAPsxpwBwiKNIuGxlOz0WhhpoWCMcALb/iyiLCTt8l10f//Z",
    //         }]
    // };

    //如果是空字典就停住（可能没有key）
    getLatestResultData() {

      // for test
      // this.bufferedTaskCache = {
      //     'datasource1': [
      //         {
      //             taskId: "1",
      //             result: "8",
      //             delay: "0.5",
      //             visualize: "data:image/jpg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/4gIoSUNDX1BST0ZJTEUAAQEAAAIYAAAAAAQwAABtbnRyUkdCIFhZWiAAAAAAAAAAAAAAAABhY3NwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAA9tYAAQAAAADTLQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAlkZXNjAAAA8AAAAHRyWFlaAAPsxpwBwiKNIuGxlOz0WhhpoWCMcALb/iyiLCTt8l10f//Z",
      //         },
      //         {
      //             taskId: "2",
      //             result: "9",
      //             delay: "0.6",
      //             visualize: "data:image/jpg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/4gIoSUNDX1BST0ZJTEUAAQEAAAIYAAAAAAQwAABtbnRyUkdCIFhZWiAAAAAAAAAAAAAAAABhY3NwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAA9tYAAQAAAADTLQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAlkZXNjAAAA8AAAAHRyWFlaAAPsxpwBwiKNIuGxlOz0WhhpoWCMcALb/iyiLCTt8l10f//Z",
      //         },
      //         {
      //             taskId: "3",
      //             result: "10",
      //             delay: "0.7",
      //             visualize: "data:image/jpg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/4gIoSUNDX1BST0ZJTEUAAQEAAAIYAAAAAAQwAABtbnRyUkdCIFhZWiAAAAAAAAAAAAAAAABhY3NwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAA9tYAAQAAAADTLQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAlkZXNjAAAA8AAAAHRyWFlaAAPsxpwBwiKNIuGxlOz0WhhpoWCMcALb/iyiLCTt8l10f//Z",
      //         },
      //         {
      //             taskId: "4",
      //             result: "11",
      //             delay: "0.8",
      //             visualize: "data:image/jpg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/4gIoSUNDX1BST0ZJTEUAAQEAAAIYAAAAAAQwAABtbnRyUkdCIFhZWiAAAAAAAAAAAAAAAABhY3NwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAA9tYAAQAAAADTLQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAlkZXNjAAAA8AAAAHRyWFlaAAPsxpwBwiKNIuGxlOz0WhhpoWCMcALb/iyiLCTt8l10f//Z",
      //         },
      //         {
      //             taskId: "5",
      //             result: "12",
      //             delay: "0.9",
      //             visualize: "data:image/jpg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/4gIoSUNDX1BST0ZJTEUAAQEAAAIYAAAAAAQwAABtbnRyUkdCIFhZWiAAAAAAAAAAAAAAAABhY3NwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAA9tYAAQAAAADTLQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAlkZXNjAAAA8AAAAHRyWFlaAAPsxpwBwiKNIuGxlOz0WhhpoWCMcALb/iyiLCTt8l10f//Z",
      //         },
      //         {
      //             taskId: "6",
      //             result: "13",
      //             delay: "1.0",
      //             visualize: "data:image/jpg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/4gIoSUNDX1BST0ZJTEUAAQEAAAIYAAAAAAQwAABtbnRyUkdCIFhZWiAAAAAAAAAAAAAAAABhY3NwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAA9tYAAQAAAADTLQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAlkZXNjAAAA8AAAAHRyWFlaAAPsxpwBwiKNIuGxlOz0WhhpoWCMcALb/iyiLCTt8l10f//Z",
      //         },
      //         {
      //             taskId: "7",
      //             result: "14",
      //             delay: "1.1",
      //             visualize: "data:image/jpg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/4gIoSUNDX1BST0ZJTEUAAQEAAAIYAAAAAAQwAABtbnRyUkdCIFhZWiAAAAAAAAAAAAAAAABhY3NwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAA9tYAAQAAAADTLQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAlkZXNjAAAA8AAAAHRyWFlaAAPsxpwBwiKNIuGxlOz0WhhpoWCMcALb/iyiLCTt8l10f//Z",
      //         },
      //         {
      //             taskId: "8",
      //             result: "15",
      //             delay: "1.2",
      //             visualize: "data:image/jpg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/4gIoSUNDX1BST0ZJTEUAAQEAAAIYAAAAAAQwAABtbnRyUkdCIFhZWiAAAAAAAAAAAAAAAABhY3NwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAA9tYAAQAAAADTLQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAlkZXNjAAAA8AAAAHRyWFlaAAPsxpwBwiKNIuGxlOz0WhhpoWCMcALb/iyiLCTt8l10f//Z",
      //         },
      //         {
      //             taskId: "9",
      //             result: "16",
      //             delay: "1.3",
      //             visualize: "data:image/jpg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/4gIoSUNDX1BST0ZJTEUAAQEAAAIYAAAAAAQwAABtbnRyUkdCIFhZWiAAAAAAAAAAAAAAAABhY3NwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAA9tYAAQAAAADTLQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAlkZXNjAAAA8AAAAHRyWFlaAAPsxpwBwiKNIuGxlOz0WhhpoWCMcALb/iyiLCTt8l10f//Z",
      //         },
      //         {
      //             taskId: "10",
      //             result: "17",
      //             delay: "1.4",
      //             visualize: "data:image/jpg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/4gIoSUNDX1BST0ZJTEUAAQEAAAIYAAAAAAQwAABtbnRyUkdCIFhZWiAAAAAAAAAAAAAAAABhY3NwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAA9tYAAQAAAADTLQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAlkZXNjAAAA8AAAAHRyWFlaAAPsxpwBwiKNIuGxlOz0WhhpoWCMcALb/iyiLCTt8l10f//Z",
      //         }
      //     ],
      //     'datasource2': [
      //         {
      //             taskId: "1",
      //             result: "8",
      //             delay: "0.5",
      //             visualize: "data:image/jpg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/4gIoSUNDX1BST0ZJTEUAAQEAAAIYAAAAAAQwAABtbnRyUkdCIFhZWiAAAAAAAAAAAAAAAABhY3NwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAA9tYAAQAAAADTLQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAlkZXNjAAAA8AAAAHRyWFlaAAPsxpwBwiKNIuGxlOz0WhhpoWCMcALb/iyiLCTt8l10f//Z",
      //         },
      //         {
      //             taskId: "2",
      //             result: "9",
      //             delay: "0.6",
      //             visualize: "data:image/jpg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/4gIoSUNDX1BST0ZJTEUAAQEAAAIYAAAAAAQwAABtbnRyUkdCIFhZWiAAAAAAAAAAAAAAAABhY3NwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAA9tYAAQAAAADTLQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAlkZXNjAAAA8AAAAHRyWFlaAAPsxpwBwiKNIuGxlOz0WhhpoWCMcALb/iyiLCTt8l10f//Z",
      //         }]
      // };

      // return;

      fetch('/api/task_result')
          .then(response => response.json())
          .then(data => {
            console.log('getLatestResultData');
            console.log(data);

            if (!data || Object.keys(data).length === 0) {
              return;
            }

            for (const key in data) {
              if (data[key].length === 0) {
                continue;
              }
              for (const item of data[key]) {
                if (!this.bufferedTaskCache[key] || this.bufferedTaskCache[key].length === 0) {
                  this.bufferedTaskCache[key] = [];
                }
                this.bufferedTaskCache[key].push(item);
                if (this.bufferedTaskCache[key].length > this.maxBufferedTaskCacheSize) {
                  this.bufferedTaskCache[key].shift();
                }
              }
            }

          });
    },


    // 3. 获取结果展示纵轴描述（开始时获取）
    // /result_prompt
    // 后端返回json格式
    // return {
    //     ‘prompt’: ‘执行结果’
    // }


    getResultPrompt() {
      // for test
      // this.prompt = '执行结果';
      // return;


      fetch('/api/result_prompt')
          .then(response => response.json())
          .then(data => {
            if (!data) {
              return;
            }
            console.log(data);
            this.visualizing_prompt = data.visualizing_prompt;
            this.result_title_prompt = data.result_title_prompt;
            this.result_text_prompt = data.result_text_prompt;
            this.delay_text_prompt = data.delay_text_prompt;
          });
    },


    // 下载文件
    exportTaskLog() {
      console.log('exportTaskLog');
      fetch('/api/download_log')
          .then(response => {
            if (!response.ok) {
              throw new Error('Network response was not ok');
            }
            // 从Content-Disposition头获取文件名
            const contentDisposition = response.headers.get('Content-Disposition');
            let filename = 'LOG.json';  // 如果没有获取到文件名，使用默认值
            if (contentDisposition) {
              const filenameMatch = /filename="([^"]+)"/.exec(contentDisposition);
              if (filenameMatch && filenameMatch[1]) {
                filename = filenameMatch[1];
              }
            }

            return response.blob().then(blob => {
              const url = window.URL.createObjectURL(new Blob([blob]));
              const link = document.createElement('a');
              link.href = url;
              link.setAttribute('download', filename);
              document.body.appendChild(link);
              link.click();

              document.body.removeChild(link);
              window.URL.revokeObjectURL(url);
            });
          })
          .catch(error => {
            console.error('Download log failed:', error);
          });
    },


  },
  mounted() {
    // this.generateRandomJPG();
    // this.generateRandomGraphData();
    // setInterval(() => {
    //     this.generateRandomJPG();
    //     this.generateRandomGraphData();
    // }, 5000);

    this.getDataSourceList();
    // this.getResultPrompt();

    // this.generateFakeDataForAllSources();
    // setInterval(() => {
    //     this.generateFakeDataForAllSources();
    // }, 5000);

    this.getLatestResultData();
    console.log(this.bufferedTaskCache);
    setInterval(() => {
      this.getLatestResultData();
      console.log(this.bufferedTaskCache);
    }, 2000);


  },
  watch: {
    selectedDataSource: function (newVal, oldVal) {
      if (newVal !== oldVal) {
        // console.log(newVal);
        // console.log(oldVal);

        // this.resultData = this.bufferedTaskCache[newVal];
        this.resultData = this.bufferedTaskCache[newVal];
        this.delayData = this.bufferedTaskCache[newVal];
        this.visualizeData = this.bufferedTaskCache[newVal];


      }
    },

  },
  computed: {
    filteredVisualizeData() {
      if (!this.visualizeData) {
        return null;
      }
      return this.visualizeData.map((item) => {
        return {
          taskId: item.taskId,
          visualize: item.visualize
        }
      });
    },
    filteredResultData() {
      if (!this.resultData) {
        return null;
      }
      return this.resultData.map((item) => {
        return {
          taskId: item.taskId,
          result: item.result
        }
      });

    },
    filteredDelayData() {
      if (!this.delayData) {
        return null;
      }
      return this.delayData.map((item) => {
        return {
          taskId: item.taskId,
          delay: item.delay
        }
      });
    },

  }
};


</script>

<style scoped lang="scss">
$homeNavLengh: 8;


.home-container {
  overflow: hidden;

  .home-card-one,
  .home-card-two,
  .home-card-three {
    .home-card-item {
      width: 100%;
      height: 130px;
      border-radius: 4px;
      transition: all ease 0.3s;
      padding: 20px;
      overflow: hidden;
      background: var(--el-color-white);
      color: var(--el-text-color-primary);
      border: 1px solid var(--next-border-color-light);

      &:hover {
        box-shadow: 0 2px 12px var(--next-color-dark-hover);
        transition: all ease 0.3s;
      }

      &-icon {
        width: 70px;
        height: 70px;
        border-radius: 100%;
        flex-shrink: 1;

        i {
          color: var(--el-text-color-placeholder);
        }
      }

      &-title {
        font-size: 15px;
        font-weight: bold;
        height: 30px;
      }
    }
  }

  .home-card-one {
    @for $i from 0 through 3 {
      .home-one-animation#{$i} {
        opacity: 0;
        animation-name: error-num;
        animation-duration: 0.5s;
        animation-fill-mode: forwards;
        animation-delay: calc($i/4) + s;
      }
    }
  }

  .home-card-two,
  .home-card-three {
    .home-card-item {
      height: 50vh;
      width: 100%;
      overflow: scroll;

      .home-monitor {
        height: 100%;

        .flex-warp-item {
          width: 25%;
          height: 111px;
          display: flex;

          .flex-warp-item-box {
            margin: auto;
            text-align: center;
            color: var(--el-text-color-primary);
            display: flex;
            border-radius: 5px;
            background: var(--next-bg-color);
            cursor: pointer;
            transition: all 0.3s ease;

            &:hover {
              background: var(--el-color-primary-light-9);
              transition: all 0.3s ease;
            }
          }

          @for $i from 0 through $homeNavLengh {
            .home-animation#{$i} {
              opacity: 0;
              animation-name: error-num;
              animation-duration: 0.5s;
              animation-fill-mode: forwards;
              animation-delay: calc($i/10) + s;
            }
          }
        }
      }
    }
  }

  .free-manage {
    .home-card-item {
      height: 50vh;
      width: 100%;
      overflow: scroll;
    }
  }

  .toggleSource {
    .home-card-item {
      height: 10vh;
      width: 100%;
      overflow: scroll;
    }
  }

}
</style>
