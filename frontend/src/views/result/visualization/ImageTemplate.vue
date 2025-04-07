<template>
  <div class="image-container">
    <img
        v-if="currentImage"
        :src="currentImage"
        :alt="config.name"
        class="responsive-image"
    />
    <div v-else class="image-placeholder">
      <el-icon :size="40">
        <Picture/>
      </el-icon>
      <p>No data available</p>
    </div>
  </div>
</template>

<script>
import {ref, watch} from 'vue'
import {Picture} from '@element-plus/icons-vue'

export default {
  components: {Picture},
  props: ['config', 'data'],
  setup(props) {
    const currentImage = ref(null)

    watch(() => props.data, () => {
      if (Array.isArray(props.data) && props.data.length > 0) {
        const validItems = props.data.filter(item =>
            item.image && typeof item.image === 'string'
        )
        if (validItems.length > 0) {
          currentImage.value = validItems[validItems.length - 1].image
          return
        }
      }
      currentImage.value = null
    }, {immediate: true, deep: true})

    return {currentImage}
  }
}
</script>

<style scoped>
.image-container {
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #f5f7fa;
  border-radius: 4px;
  overflow: hidden;
}

.responsive-image {
  max-width: 100%;
  max-height: 100%;
  object-fit: contain;
}

.image-placeholder {
  color: #999;
  font-size: 0.9em;
  text-align: center;
  padding: 20px;
}
</style>