<template>
  <div class="curve-controls">
    <div class="variable-group">
      <span class="variable-title">Display Variables:</span>
      <el-checkbox-group v-model="localVariableStates">
        <el-checkbox
          v-for="varName in config.variables"
          :key="varName"
          :label="varName"
          class="variable-checkbox"
        />
      </el-checkbox-group>
    </div>
  </div>
</template>

<style scoped>
.curve-controls {
  padding: 8px 0;
}

.variable-group {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  align-items: center;
}

.variable-title {
  font-size: 0.85em;
  color: var(--el-text-color-secondary);
  margin-right: 8px;
}

.variable-checkbox {
  margin-right: 8px;
  ::v-deep .el-checkbox__label {
    font-size: 0.85em;
  }
}
</style>