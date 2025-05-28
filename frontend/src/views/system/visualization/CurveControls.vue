<template>
  <div class="curve-controls">
    <div class="variable-group">
      <span class="variable-title">Display Variables:</span>
      <el-checkbox-group
          v-model="selectedVariables"
          @change="handleVariableChange"
      >
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

<script>
import {ref, watch, toRefs} from 'vue'

export default {
  props: {
    config: {
      type: Object,
      required: true
    },
    variableStates: {
      type: Object,
      default: () => ({})
    }
  },
  emits: ['update:variable-states'],

  setup(props, {emit}) {
    const {variableStates} = toRefs(props)
    const selectedVariables = ref([])

    // 初始化时同步状态
    selectedVariables.value = Object.keys(variableStates.value)
        .filter(k => variableStates.value[k])

    const handleVariableChange = () => {
      const newStates = {}
      props.config.variables.forEach(varName => {
        newStates[varName] = selectedVariables.value.includes(varName)
      })
      emit('update:variable-states', newStates)
    }

    watch(() => props.variableStates, (newVal) => {
      selectedVariables.value = Object.keys(newVal).filter(k => newVal[k])
    }, {deep: true})

    return {selectedVariables, handleVariableChange}
  }
}
</script>

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
}

.variable-checkbox ::v-deep .el-checkbox__label {
  font-size: 0.85em;
}
</style>