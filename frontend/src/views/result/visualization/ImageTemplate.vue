<template>
  <div class="image-container">
    <div class="image-wrapper">
      <img
        v-if="currentImage"
        :src="currentImage"
        :alt="config.name"
        class="responsive-image"
      />
      <div v-else class="image-placeholder">
        No image available
      </div>
    </div>
  </div>
</template>

<script>
import { ref, watch } from 'vue'

export default {
  props: ['config', 'data'],

  setup(props) {
    const currentImage = ref(null)

    watch(() => props.data, () => {
      if (props.data?.length) {
        currentImage.value = props.data[props.data.length - 1]?.image
      } else {
        currentImage.value = null
      }
    }, { immediate: true })

    return { currentImage }
  }
}
</script>

<style scoped>
.image-container {
  width: 100%;
  height: calc(100% - 40px);
  padding: 12px;
}

.image-wrapper {
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
}

.responsive-image {
  max-width: 100%;
  max-height: 100%;
  object-fit: contain;
}

.image-placeholder {
  color: #999;
  font-size: 0.9em;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
}
</style>