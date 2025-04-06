<template>
  <div class="curve-module">
    <div class="chart-container" :id="config.name"></div>
  </div>
</template>

<script>
// [Keep script section same as previous implementation]
</script>

<style scoped>
.curve-module {
  padding: 15px;
  height: calc(100% - 50px); /* Account for header height */
}

.chart-container {
  width: 100%;
  height: 100%;
  min-height: 300px;
}
</style>