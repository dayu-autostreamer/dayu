<template>
  <div class="image-container">
    <!-- 加载状态 -->
    <div v-if="isLoading" class="loading-overlay">
      <el-icon class="loading-icon" :size="30">
        <Loading/>
      </el-icon>
    </div>

    <!-- 错误状态 -->
    <div v-if="loadError" class="error-state">
      <el-icon :size="40" color="#F56C6C">
        <Warning/>
      </el-icon>
      <p>Image load failed</p>
    </div>

    <!-- 正常显示 -->
    <img
        v-else-if="currentImage"
        :src="currentImage"
        :alt="config.name"
        class="responsive-image"
        @error="loadError = true"
    />

    <!-- 空状态 -->
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
import {Picture, Warning, Loading} from '@element-plus/icons-vue'

// Base64 前缀检测正则
const BASE64_REGEX = /^data:image\/(\w+);base64,/

export default {
  components: {Picture, Warning, Loading},
  props: ['config', 'data'],
  setup(props) {
    const currentImage = ref(null)
    const isLoading = ref(false)
    const loadError = ref(false)

    const processBase64 = (input) => {
      // 已有完整前缀直接返回
      if (BASE64_REGEX.test(input)) return input

      // 默认添加 png 类型前缀
      return `data:image/png;base64,${input}`
    }

    watch(() => props.data, (newData) => {
      loadError.value = false
      const validItems = (newData || []).filter(item =>
          item?.image !== undefined && item.image !== null
      )

      if (validItems.length === 0) {
        currentImage.value = null
        return
      }

      // 取最新数据项
      const latestItem = validItems[validItems.length - 1]

      try {
        isLoading.value = true
        currentImage.value = processBase64(latestItem.image)
      } catch (e) {
        console.error('Image process error:', e)
        loadError.value = true
        currentImage.value = null
      } finally {
        isLoading.value = false
      }
    }, {deep: true, immediate: true})

    return {
      currentImage,
      isLoading,
      loadError
    }
  }
}
</script>

<style scoped>
.image-container {
  width: 100%;
  height: 100%;
  min-height: 300px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 4px;
  overflow: hidden;
  position: relative;
}

.loading-overlay {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(255, 255, 255, 0.8);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 2;
}

.loading-icon {
  animation: rotate 2s linear infinite;
}

@keyframes rotate {
  from {
    transform: rotate(0deg);
  }
  to {
    transform: rotate(360deg);
  }
}

.error-state {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  text-align: center;
  color: #F56C6C;
  z-index: 1;
}

.responsive-image {
  max-width: 100%;
  max-height: 80vh;
  object-fit: contain;
}

.image-placeholder {
  color: #999;
  font-size: 0.9em;
  text-align: center;
  padding: 20px;
}
</style>