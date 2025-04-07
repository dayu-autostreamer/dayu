<template>
  <div class="home-container layout-pd">
    <!-- Data Source Selection Row -->
    <el-row :gutter="15" class="home-card-two mb15">
      <el-col :xs="24" :sm="24" :md="20" :lg="20" :xl="20">
        <div class="home-card-item data-source-container">
          <div class="flex-margin flex w100">
            <div class="flex-auto" style="font-weight: bold">
              Choose Datasource: &nbsp; &nbsp;
              <el-select
                  v-model="selectedDataSource"
                  placeholder="Please choose datasource"
                  class="compact-select"
              >
                <el-option
                    v-for="item in dataSourceList"
                    :key="item.id"
                    :label="item.label"
                    :value="item.id"
                />
              </el-select>
            </div>
          </div>
        </div>
      </el-col>
      <el-col :xs="24" :sm="24" :md="4" :lg="4" :xl="4">
        <div class="home-card-item export-container">
          <el-button
              type="primary"
              class="export-button"
              @click="exportTaskLog"
              style="font-weight: bold"
          >
            Export Log
          </el-button>
        </div>
      </el-col>
    </el-row>

    <!-- Visualization Controls Row -->
    <el-row class="viz-controls-row mb15">
      <el-col :span="24">
        <div class="viz-controls-panel">
          <div class="control-group">
            <h4>Active Visualizations:</h4>
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
            <h3 class="viz-title">{{ viz.name }}</h3>
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
import {defineAsyncComponent, reactive, markRaw, shallowRef} from 'vue'

export default {
  data() {
    return {
      selectedDataSource: null,
      dataSourceList: [],
      bufferedTaskCache: {},
      maxBufferedTaskCacheSize: 20,

      // Visualization system
      visualizationConfig: [],
      activeVisualizations: new Set(),
      variableStates: reactive({}),
      visualizationComponents: {},
      vizControls: {},
      pollingInterval: null
    }
  },
  computed: {
    processedData() {
      const result = {}
      this.visualizationConfig.forEach(viz => {
        result[viz.id] = this.processVizData(viz)
      })
      return result
    },
    activeVisualizationsArray: {
      get() {
        return Array.from(this.activeVisualizations)
      },
      set(newVal) {
        this.activeVisualizations = new Set(newVal)
      }
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
          this.visualizationComponents[type] = markRaw(defineAsyncComponent(() => modules[path]()))
        }

        for (const path in controls) {
          const type = path.split('/').pop().replace('Controls.vue', '').toLowerCase()
          this.vizControls[type] = markRaw(defineAsyncComponent(() => controls[path]()))
        }
      } catch (error) {
        console.error('Component auto-registration failed:', error)
      }
    },

    processVizData(vizConfig) {
      const sourceData = this.bufferedTaskCache[this.selectedDataSource] || [];
      return sourceData.map(task => {
        const vizData = task.data[vizConfig.id] || {};

        return {
          taskId: task.task_id,
          ...Object.fromEntries(
              Object.entries(vizData)
                  .filter(([key]) => vizConfig.variables.includes(key))
          )
        };
      });
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

        // Initialize module states
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

        console.log('get result data: ', data)

        Object.keys(data).forEach(sourceId => {
          if (!this.bufferedTaskCache[sourceId]) {
            this.$set(this.bufferedTaskCache, sourceId, [])
          }

          data[sourceId].forEach(task => {
            const newTask = {
              task_id: task.task_id,
              data: task.data.map(item => ({...item}))
            }

            this.bufferedTaskCache[sourceId].splice(
                this.bufferedTaskCache[sourceId].length,
                0,
                newTask
            )

            if (this.bufferedTaskCache[sourceId].length > this.maxBufferedTaskCacheSize) {
              this.bufferedTaskCache[sourceId].splice(0, 1)
            }
          })
        })

        // 强制触发视图更新
        this.bufferedTaskCache = {...this.bufferedTaskCache}
      } catch (error) {
        console.error('Failed to fetch task results:', error)
      }
    },

    setupDataPolling() {
      this.getLatestResultData()
      this.pollingInterval = setInterval(() => {
        this.getLatestResultData()
      }, 1000)
    },

    exportTaskLog() {
      fetch('/api/download_log')
          .then(response => response.blob())
          .then(blob => {
            const url = window.URL.createObjectURL(blob)
            const link = document.createElement('a')
            link.href = url
            link.setAttribute('download', 'task_log.json')
            document.body.appendChild(link)
            link.click()
            link.remove()
          })
    }
  },
  beforeUnmount() {
    if (this.pollingInterval) {
      clearInterval(this.pollingInterval)
    }
  }
}
</script>

<style scoped>
.home-container {
  overflow: hidden;
  padding: 16px;
}

.data-source-container {
  height: auto;
  padding: 8px 12px;
}

.export-container {
  height: auto;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 8px;
}

.compact-select {
  width: 70%;
}

.compact-select ::v-deep .el-input__inner {
  height: 32px;
  line-height: 32px;
}

.export-button {
  padding: 8px 12px;
}

.viz-controls-row {
  margin-top: 20px;
}

.viz-controls-panel {
  background: var(--el-bg-color);
  border-radius: 4px;
  padding: 15px;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, .1);
  display: flex;
  flex-direction: column;
}

.control-group {
  margin-bottom: 8px;
}

.control-group h4 {
  margin-bottom: 10px;
  color: var(--el-text-color-primary);
}

.el-checkbox-group {
  display: flex;
  flex-wrap: wrap;
  gap: 8px 12px;
  align-items: center;
}

.module-checkbox {
  margin-right: 12px;
  margin-bottom: 4px;
  white-space: nowrap;
}

.viz-module {
  height: 500px;
  overflow: hidden;
  display: flex;
  flex-direction: column;
  margin-top: 15px;
}

.viz-module-header {
  padding: 12px;
  border-bottom: 1px solid var(--el-border-color-light);
}

.viz-title {
  margin: 0 0 8px 0;
  font-size: 1.1em;
  color: var(--el-text-color-primary);
  text-align: center;
}

.home-card-item {
  background: var(--el-bg-color);
  border-radius: 4px;
  border: 1px solid var(--el-border-color-light);
}

@media (max-width: 768px) {
  .viz-controls-panel {
    padding: 10px;
  }

  .adaptive-checkbox-group {
    gap: 6px 8px;
  }

  .module-checkbox {
    margin-right: 8px;
    font-size: 0.9em;
  }
}

</style>