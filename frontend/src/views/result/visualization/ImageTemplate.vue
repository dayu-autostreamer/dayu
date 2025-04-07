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

    watch(() => props.data, (newData) => {
      console.log('[DEBUG] Image data:', newData)
      currentImage.value = newData?.filter?.(item =>
          item?.image?.startsWith?.('http')
      )?.pop()?.image || null
    }, {deep: true})

    return {currentImage}
  }
}

function isValidImageUrl(url) {
  try {
    new URL(url)
    return url.match(/\.(jpeg|jpg|gif|png|svg)$/) != null
  } catch {
    return false
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