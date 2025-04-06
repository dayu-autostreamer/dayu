<template>
  <div class="home-container layout-pd">
    <el-row :gutter="15" class="home-card-two mb15">
      <el-col :xs="24" :sm="24" :md="20" :lg="20" :xl="20">
        <div class="home-card-item data-source-container">
          <div class="flex-margin flex w100">
            <div class="flex-auto" style="font-weight: bold">

              Choose Datasource: &nbsp; &nbsp;

              <el-select v-model="selectedDataSource" placeholder="Please choose datasource"
                         style="width: 70%; font-weight: normal">
                <el-option v-for="item in dataSourceList" :key="item.id" :label="item.label"
                           :value="item.id">
                </el-option>
              </el-select>

            </div>
          </div>
        </div>
      </el-col>
      <el-col :xs="24" :sm="24" :md="4" :lg="4" :xl="4">
        <div class="home-card-item export-container">
          <div class="flex-margin flex w100">
            <div class="flex-auto">
              <div style="display: flex; justify-content: center; align-items: center;">
                <el-button type="primary" class="export-button" @click="exportTaskLog"
                           style="font-weight: bold; width:100%">
                  Export Log
                </el-button>
              </div>

            </div>
          </div>
        </div>
      </el-col>
    </el-row>

    <!-- Visualization Controls Row -->
    <el-row class="viz-controls-row mb15">
      <el-col :span="24">
        <div class="viz-controls-panel">
          <div class="control-group">
            <h4>Visualization Modules:</h4>
            <el-checkbox-group v-model="activeVisualizationsArray">
              <el-checkbox
                  v-for="viz in visualizationConfig"
                  :key="viz.id"
                  :label="viz.id"
                  class="module-checkbox"
              >
                {{ viz.name }}
              </el-checkbox>
            </el-checkbox-group>
          </div>
        </div>
      </el-col>
    </el-row>

    <!-- Visualization Modules Row -->
    <el-row :gutter="15" class="home-card-two mb15">
      <el-col
          v-for="viz in visualizationConfig"
          :key="viz.id"
          :xs="24" :sm="24" :md="8" :lg="8" :xl="8"
          v-show="activeVisualizations.has(viz.id)"
      >
        <div class="home-card-item viz-module">
          <div class="viz-module-header">
            <component
                :is="vizControls[viz.type]"
                v-if="vizControls[viz.type]"
                :config="viz"
                :variable-states="variableStates[viz.id]"
                @update:variable-states="updateVariableStates(viz.id, $event)"
            />
          </div>

          <component
              :is="visualizationComponents[viz.type]"
              v-if="visualizationComponents[viz.type]"
              :config="viz"
              :data="processedData[viz.id]"
              :variable-states="variableStates[viz.id]"
          />
        </div>
      </el-col>
    </el-row>

  </div>
</template>

<script>


import {defineAsyncComponent, defineComponent, ref, reactive} from 'vue'

export default {
  data() {
    return {
      visualizationConfig: [], // Store visualization templates
      activeVisualizations: new Set(), // Track enabled visualizations
      variableStates: reactive({}), // Track variable visibility for curves
      visualizationComponents: reactive({}),
      vizControls: reactive({}),

      dataSourceList: [],
      selectedDataSource: null,

      bufferedTaskCache: {},
      maxBufferedTaskCacheSize: 20,
    }
  },

  computed: {
    processedData() {
      const result = {}
      this.visualizationConfig.forEach(viz => {
        result[viz.id] = this.processVizData(viz)
      })
      return result
    }
  },

  async created() {
    await this.autoRegisterComponents()
    await this.fetchDataSourceList()
    await this.fetchVisualizationConfig()
    this.setupDataPolling()
  },


  methods: {
    async autoRegisterComponents() {
      try {
        const modules = import.meta.glob('./visualization/*Template.vue')
        const controls = import.meta.glob('./visualization/*Controls.vue')

        for (const path in modules) {
          const type = path.split('/').pop().replace('Template.vue', '').toLowerCase()
          this.visualizationComponents[type] = defineAsyncComponent(() => modules[path]())
        }

        for (const path in controls) {
          const type = path.split('/').pop().replace('Controls.vue', '').toLowerCase()
          this.vizControls[type] = defineAsyncComponent(() => controls[path]())
        }
      } catch (error) {
        console.error('Component auto-registration failed:', error)
      }
    },

    processVizData(vizConfig) {
      const sourceData = this.bufferedTaskCache[this.selectedDataSource] || []
      return sourceData.map(task => ({
        taskId: task.task_id,
        ...this.extractVizVariables(task.data, vizConfig)
      }))
    },

    extractVizVariables(taskData, vizConfig) {
      const vizData = taskData[vizConfig.id] || {}
      return Object.fromEntries(
          Object.entries(vizData)
              .filter(([key]) => vizConfig.variables.includes(key))
      )
    },

    updateVariableStates(vizId, newStates) {
      this.variableStates[vizId] = {
        ...this.variableStates[vizId],
        ...newStates
      }
    },

    async fetchDataSourceList() {
      try {
        const response = await fetch('/api/source_list')
        this.dataSourceList = await response.json()
        this.dataSourceList.forEach(source => {
          this.bufferedTaskCache[source.id] = this.bufferedTaskCache[source.id] || []
        })
      } catch (error) {
        console.error('Failed to fetch data sources:', error)
      }
    },

    async fetchVisualizationConfig() {
      try {
        const response = await fetch('/api/visualization_config')
        this.visualizationConfig = await response.json()

        this.visualizationConfig.forEach(viz => {
          this.activeVisualizations.add(viz.id)
          if (!this.variableStates[viz.id]) {
            this.variableStates[viz.id] = reactive({})
          }
          viz.variables?.forEach(varName => {
            if (this.variableStates[viz.id][varName] === undefined) {
              this.variableStates[viz.id][varName] = true
            }
          })
        })
      } catch (error) {
        console.error('Failed to fetch visualization config:', error)
      }
    },

    async getLatestResultData() {
      try {
        const response = await fetch('/api/task_result')
        const data = await response.json()

        for (const sourceId in data) {
          if (data[sourceId].length === 0) continue

          if (!this.bufferedTaskCache[sourceId]) {
            this.bufferedTaskCache[sourceId] = []
          }

          data[sourceId].forEach(task => {
            this.bufferedTaskCache[sourceId].push(task)
            if (this.bufferedTaskCache[sourceId].length > this.maxBufferedTaskCacheSize) {
              this.bufferedTaskCache[sourceId].shift()
            }
          })
        }
      } catch (error) {
        console.error('Failed to fetch task results:', error)
      }
    },

    setupDataPolling() {
      this.getLatestResultData()
      this.pollingInterval = setInterval(() => {
        this.getLatestResultData()
      }, 2000)
    },

    exportTaskLog() {
      console.log('exportTaskLog');
      fetch('/api/download_log')
          .then(response => {
            if (!response.ok) {
              throw new Error('Network response was not ok');
            }
            const contentDisposition = response.headers.get('Content-Disposition');
            let filename = 'task_log.json';
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
  beforeUnmount() {
    if (this.pollingInterval) {
      clearInterval(this.pollingInterval)
    }
  },
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

.viz-controls {
  padding: 10px;
  border-bottom: 1px solid #eee;

  .el-checkbox {
    margin-right: 15px;
  }

  .variable-selector {
    margin-top: 8px;
    padding-left: 20px;
  }
}

.chart-container {
  width: 24vw;
  height: 32vh;
}

.responsive-image {
  max-width: 100%;
  height: auto;
  display: block;
  margin: 0 auto;
}

.home-container {
  overflow: hidden;
}

.viz-header {
  padding: 10px;
  border-bottom: 1px solid #eee;
  display: flex;
  flex-direction: column;
}

.viz-checkbox {
  margin-bottom: 8px;
}

.home-card-item {
  height: 500px;
  overflow: hidden;
  display: flex;
  flex-direction: column;
}

.home-card-item > div:last-child {
  flex-grow: 1;
  overflow: auto;
}

</style>
