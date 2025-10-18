<template>
  <div class="image-controls">
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
    }
  },
  computed: {
    tagContent() {
      return this.config?.variables?.[0] || 'image'
    }
  }
}
</script>

<style scoped>
.image-controls {
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