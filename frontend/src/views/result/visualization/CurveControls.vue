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

<script>
import { reactive, watch } from 'vue'

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

  setup(props, { emit }) {
    const localVariableStates = reactive(
      Object.keys(props.variableStates).filter(k => props.variableStates[k])
    )

    const handleChange = () => {
      const newStates = {}
      props.config.variables.forEach(varName => {
        newStates[varName] = localVariableStates.includes(varName)
      })
      emit('update:variable-states', newStates)
    }

    watch(() => props.variableStates, (newVal) => {
      localVariableStates.splice(0)
      Object.keys(newVal).filter(k => newVal[k]).forEach(k => {
        localVariableStates.push(k)
      })
    }, { deep: true })

    watch(localVariableStates, handleChange)

    return { localVariableStates }
  }
}
</script>

<style scoped>
.curve-controls {
  padding: 8px 0;
}

.variable-group {
  display: flex;
  align-items: center;
  gap: 10px;
}

.variable-title {
  font-size: 0.9em;
  color: var(--el-text-color-secondary);
  white-space: nowrap;
}

.variable-checkbox {
  margin-right: 12px;
}

.el-checkbox-group {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}
</style>