<template>
  <div class="home-container layout-pd">
    <!-- Data Source Selection Row -->
    <el-row :gutter="15" class="home-card-two mb15">
      <el-col :xs="24" :sm="24" :md="20" :lg="20" :xl="20">
        <div class="home-card-item data-source-container">
          <div class="flex-margin flex w100">
            <div class="flex-auto" style="font-weight: bold">
              Choose Datasource: &nbsp;
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
          v-show="componentsLoaded && activeVisualizations.has(viz.id)"
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
              v-if="componentsLoaded && visualizationComponents[viz.type]"
              :key="`${viz.type}-${selectedDataSource}`"
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
import {defineAsyncComponent, reactive, ref, markRaw, toRaw, watch} from 'vue'
import mitt from 'mitt'

const emitter = mitt()

export default {
  data() {
    return {
      selectedDataSource: null,
      dataSourceList: [],
      bufferedTaskCache: reactive({}),
      maxBufferedTaskCacheSize: 20,
      componentsLoaded: false,
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
    this.dataSourceList.forEach(source => {
      this.bufferedTaskCache[source.id] = reactive([])
    })

    watch(
        () => this.bufferedTaskCache,
        (newVal) => {
          console.log('Cache updated:', newVal)
        },
        {deep: true}
    )

    await this.autoRegisterComponents()
    this.componentsLoaded = true
    await this.fetchDataSourceList()
    await this.fetchVisualizationConfig()
    this.setupDataPolling()

    emitter.on('force-update-charts', () => {
      this.$nextTick(() => {
        this.visualizationConfig.forEach(viz => {
          this.variableStates[viz.id] = {...this.variableStates[viz.id]}
        })
      })
    })
  },
  methods: {
    async autoRegisterComponents() {
      try {
        const modules = import.meta.glob('./visualization/*Template.vue')
        const controls = import.meta.glob('./visualization/*Controls.vue')

        await Promise.all([
          ...Object.entries(modules).map(async ([path, loader]) => {
            const type = path.split('/').pop().replace('Template.vue', '').toLowerCase()
            try {
              const comp = await loader()
              this.visualizationComponents[type] = markRaw(comp.default)
              console.log('Successfully registered:', type)
            } catch (e) {
              console.error(`Failed to load ${type} template:`, e)
            }
          }),
          ...Object.entries(controls).map(async ([path, loader]) => {
            const type = path.split('/').pop().replace('Controls.vue', '').toLowerCase()
            try {
              const comp = await loader()
              this.vizControls[type] = markRaw(comp.default)
              console.log('Successfully registered control:', type)
            } catch (e) {
              console.error(`Failed to load ${type} control:`, e)
            }
          })
        ])
      } catch (error) {
        console.error('Component auto-registration failed:', error)
      }
    },

    processVizData(vizConfig) {

      return (this.bufferedTaskCache[this.selectedDataSource] || [])
          .filter(task => {
            // 更宽松的过滤条件
            return task?.task_id &&
                task.data?.some?.(item =>
                    item?.id === vizConfig.id &&
                    Object.keys(item?.data || {}).length > 0
                )
          })
          .map(task => {
            const vizData = task.data.find(item => item.id === vizConfig.id)?.data || {}
            return {
              taskId: String(task.task_id),
              // 增强数据类型转换
              ...Object.fromEntries(
                  Object.entries(vizData).map(([k, v]) => [
                    k,
                    typeof v === 'number' ? v :
                        !isNaN(v) ? parseFloat(v) : 0
                  ])
              )
            }
          })
    },

    updateVariableStates(vizId, newStates) {
      this.variableStates[vizId] = {
        ...this.variableStates[vizId],
        ...newStates
      }
      emitter.emit('force-update-charts')
    },

    async fetchDataSourceList() {
      try {
        const response = await fetch('/api/source_list')
        this.dataSourceList = await response.json()
        this.dataSourceList.forEach(source => {
          this.bufferedTaskCache[source.id] = reactive([])
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

        this.visualizationConfig = [...this.visualizationConfig]
        console.log('Visualization config initialized:', {
          config: toRaw(this.visualizationConfig),
          variables: toRaw(this.variableStates)
        })
      } catch (error) {
        console.error('Failed to fetch visualization config:', error)
      }
    },

    async getLatestResultData() {
      try {
        const response = await fetch('/api/task_result')
        const rawData = await response.json()
        console.log('[DEBUG] API Response:', JSON.parse(JSON.stringify(rawData)))

        Object.entries(rawData).forEach(([sourceId, tasks]) => {
          if (!Array.isArray(tasks)) return

          const validTasks = tasks
              .filter(task => task?.task_id && Array.isArray(task.data))
              .map(task => ({
                task_id: task.task_id,
                data: task.data.map(item => ({
                  id: item.id || 'unknown',
                  data: item.data || {}
                }))
              }))

          const currentCache = this.bufferedTaskCache[sourceId] || []
          const newCache = [...currentCache, ...validTasks]
              .slice(-this.maxBufferedTaskCacheSize)

          // 关键响应式更新
          this.bufferedTaskCache = {
            ...this.bufferedTaskCache,
            [sourceId]: reactive(newCache)
          }
        })

        // 双保险更新机制
        this.$nextTick(() => {
          emitter.emit('force-update-charts')
          this.visualizationConfig = [...this.visualizationConfig]
        })
      } catch (error) {
        console.error('[ERROR] Fetch failed:', error)
      }
    },

    setupDataPolling() {
      this.getLatestResultData()
      this.pollingInterval = setInterval(() => {
        this.getLatestResultData()
      }, 2000)
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
    },

    forceChartUpdate() {
      emitter.emit('force-update-charts')
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
  width: 100%;
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
}

.control-group {
  margin-bottom: 8px;
}

.control-group h4 {
  margin-bottom: 10px;
  color: var(--el-text-color-primary);
}

.module-checkbox {
  margin-right: 20px;
  margin-bottom: 8px;
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

/* 确保容器尺寸正确 */
.viz-module {
  height: 500px !important;
  min-height: 500px;
}

/* 修复ECharts容器尺寸 */
.chart-wrapper {
  width: 100% !important;
  height: 100% !important;
  min-height: 400px !important;
}
</style>