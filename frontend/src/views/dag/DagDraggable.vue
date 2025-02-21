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

// 传入的数据
const props = defineProps({
  daginfo: {
    type: Object,
  },
});

// 传入的事件
const emit = defineEmits(["update-position"]);

// 初始化可拖拽的 div 的引用
const draggableDiv = ref(null);
// 组件初始位置
let boxPos = ref({
  x: 0,
  y: 0,
});
// 鼠标初始位置
let startPos = ref({
  x: 0,
  y: 0,
});
let isMove = ref(0);

// 处理开始拖拽的函数
const startDrag = (event) => {
  const { clientX, clientY } = event;
  // console.log(draggableDiv.value.offsetLeft);
  // console.log(draggableDiv.value.offsetTop);
  boxPos.value = {
    x: draggableDiv.value.offsetLeft,
    y: draggableDiv.value.offsetTop,
  };
  // console.log(boxPos);

  // console.log(clientX);
  // console.log(clientY);
  startPos.value = {
    x: clientX,
    y: clientY,
  };
  // console.log(startPos);

  isMove.value = 1;
  draggableDiv.value.style.cursor = "grab";
  const moveHandler = (event) => {
    // console.log("移动回调");
    if (!isMove.value) return;
    // 计算新的位置
    const { clientX, clientY } = event;
    const newClientX = clientX;
    const newClientY = clientY;
    const x = newClientX - startPos.value.x;
    const y = newClientY - startPos.value.y;

    draggableDiv.value.style.left = `${x + boxPos.value.x}px`;
    draggableDiv.value.style.top = `${y + boxPos.value.y}px`;

    const rect = draggableDiv.value.getBoundingClientRect();

    // console.log(parseFloat(draggableDiv.value.style.left));
    // console.log(parseFloat(draggableDiv.value.style.top));
    // console.log(rect.height);
    // console.log(rect.width);
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
    // console.log("结束移动回调");
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
