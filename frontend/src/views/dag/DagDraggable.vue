<template>
  <!-- 可拖拽的 div -->
  <div ref="draggableDiv" class="draggable" @mousedown="startDrag">
    <div class="container">
      <!-- 2333 -->
      <div class="namebox">
        {{ daginfo.service_id }}
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, defineProps, defineEmits } from "vue";

const props = defineProps({
  daginfo: {
    type: Object,
  },
});

const emit = defineEmits(["update-position"]);

const draggableDiv = ref(null);
let boxPos = ref({
  x: 0,
  y: 0,
});
let startPos = ref({
  x: 0,
  y: 0,
});
let isMove = ref(0);

const startDrag = (event) => {
  const { clientX, clientY } = event;

  boxPos.value = {
    x: draggableDiv.value.offsetLeft,
    y: draggableDiv.value.offsetTop,
  };

  startPos.value = {
    x: clientX,
    y: clientY,
  };

  isMove.value = 1;
  draggableDiv.value.style.cursor = "grab";
  const moveHandler = (event) => {
    if (!isMove.value) return;
    const { clientX, clientY } = event;
    const newClientX = clientX;
    const newClientY = clientY;
    const x = newClientX - startPos.value.x;
    const y = newClientY - startPos.value.y;

    draggableDiv.value.style.left = `${x + boxPos.value.x}px`;
    draggableDiv.value.style.top = `${y + boxPos.value.y}px`;

    const rect = draggableDiv.value.getBoundingClientRect();

    const data = {
      id: props.daginfo.id,
      leftSide: parseFloat(draggableDiv.value.style.left),
      rightSide: parseFloat(draggableDiv.value.style.left) + rect.width,
      upSide: parseFloat(draggableDiv.value.style.top),
      downSide: parseFloat(draggableDiv.value.style.top) + rect.height,
    };
    console.log(data);
    emit("update-position", data);
  };
  const upHandler = (event) => {
    isMove.value = 0;
    document.body.style.cursor = "default";
    document.removeEventListener("mousemove", moveHandler);
    document.removeEventListener("mouseup", upHandler);
  };
  document.addEventListener("mousemove", moveHandler);
  document.addEventListener("mouseup", upHandler);
};
</script>

<style>
.draggable {
  position: absolute;
}
.container {
  display: flex;
  justify-content: center;
  align-items: center;

  border: 1px solid #ccc;
  overflow: hidden;
  position: relative;
  border-radius: 5px;

  background-color: aquamarine;
}
.namebox {
  margin: 15px;
  font-size: 20px;
}
</style>
