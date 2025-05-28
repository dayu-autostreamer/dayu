<template>
  <div class="topology-controls">
    <el-tag type="info" size="small" effect="plain">
      {{tagContent}}
    </el-tag>
  </div>
</template>

<script>
export default {
  props: {
    config: {
      type: Object,
      required: true,
      validator: (val) => {
        return Array.isArray(val.variables) && val.variables.length > 0
      }
    }
  },
  computed: {
    tagContent() {
      return this.config?.variables?.[0] || 'Auto-Updated Topology'
    }
  }
}
</script>

<style scoped>
.topology-controls {
  padding-top: 4px;
  text-align: center;
}

.el-tag {
  white-space: nowrap;
  max-width: 100%;
  overflow: hidden;
  text-overflow: ellipsis;
}
</style>