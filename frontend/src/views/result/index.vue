<template>
  <div class="home-container layout-pd">
    <!-- Compact Data Source Selection Row -->
    <el-row :gutter="15" class="mb15 compact-control-row">
      <el-col :xs="24" :sm="24" :md="20" :lg="20" :xl="20">
        <div class="compact-control-item">
          <div class="data-source-select">
            <span class="control-label">Datasource:</span>
            <el-select
              v-model="selectedDataSource"
              placeholder="Select datasource"
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
      </el-col>
      <el-col :xs="24" :sm="24" :md="4" :lg="4" :xl="4">
        <div class="compact-control-item export-wrapper">
          <el-button
            type="primary"
            class="compact-export"
            @click="exportTaskLog"
          >
            Export Log
          </el-button>
        </div>
      </el-col>
    </el-row>

    <!-- Visualization Controls -->
    <div class="viz-controls-panel">
      <div class="control-group">
        <h4>Active Modules:</h4>
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

    <!-- Visualization Modules -->
    <el-row :gutter="15" class="viz-modules-row">
      <el-col
        v-for="viz in visualizationConfig"
        :key="viz.id"
        :xs="24" :sm="24" :md="8" :lg="8" :xl="8"
        v-show="activeVisualizations.has(viz.id)"
      >
        <div class="viz-module">
          <!-- Module Title -->
          <div class="module-header">
            <h3 class="module-title">{{ viz.name }}</h3>
          </div>

          <!-- Visualization Component -->
          <component
            :is="visualizationComponents[viz.type]"
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
// [Keep script section same as previous implementation]
</script>

<style scoped>
/* Compact Control Row */
.compact-control-row {
  margin-bottom: 15px;
}

.compact-control-item {
  display: flex;
  align-items: center;
  height: 50px; /* Fixed height for single line */
  padding: 5px 15px;
  background: var(--el-bg-color);
  border-radius: 4px;
  border: 1px solid var(--el-border-color-light);
}

.data-source-select {
  display: flex;
  align-items: center;
  gap: 10px;
}

.control-label {
  font-weight: bold;
  color: var(--el-text-color-primary);
}

.compact-select {
  width: 70%;
}

.export-wrapper {
  display: flex;
  justify-content: flex-end;
}

.compact-export {
  width: 100%;
  max-width: 150px;
}

/* Visualization Modules */
.module-header {
  padding: 12px;
  border-bottom: 1px solid var(--el-border-color-light);
}

.module-title {
  margin: 0;
  font-size: 1.1em;
  color: var(--el-text-color-primary);
  text-align: center;
}

.viz-module {
  background: var(--el-bg-color);
  border-radius: 4px;
  border: 1px solid var(--el-border-color-light);
  margin-bottom: 15px;
}

/* Keep other styles same as previous implementation */
</style>