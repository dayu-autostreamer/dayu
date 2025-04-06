<template>
  <div class="variable-controls">
    <el-checkbox
      v-for="varName in config.variables"
      :key="varName"
      v-model="localVariableStates[varName]"
      @change="handleChange"
    >
      {{ varName }}
    </el-checkbox>
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
    const localVariableStates = reactive({...props.variableStates})

    const handleChange = () => {
      emit('update:variable-states', {...localVariableStates})
    }

    watch(() => props.variableStates, (newVal) => {
      Object.assign(localVariableStates, newVal)
    }, { deep: true })

    return { localVariableStates, handleChange }
  }
}
</script>

<style scoped>
.variable-controls {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 8px;
}

.variable-controls .el-checkbox {
  margin-right: 0;
}
</style>