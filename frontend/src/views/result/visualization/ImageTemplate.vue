<template>
  <div class="image-module">
    <div class="image-container">
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

<style scoped>
.image-module {
  padding: 15px;
  height: calc(100% - 50px); /* Account for header height */
}

.image-container {
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
  color: var(--el-text-color-secondary);
  font-style: italic;
}
</style>