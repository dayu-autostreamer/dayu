<template>
  <div class="outline">
    <div>
      <h3>Add Application Dags</h3>
    </div>

    <div>
      <div class="new-dag-font-style">Dag Name</div>
      <el-input v-model="newInputName" placeholder="Please fill the dag name" />
      <br />
      <br />
      <div style="display: inline">
        <div class="new-dag-font-style">
          Service Containers
          <el-tooltip placement="right">
            <template #content>
              From docker Registry: https://hub.docker.com/repositories/dayuhub
            </template>
            <el-button size="small" circle>i</el-button>
          </el-tooltip>
        </div>
        <!-- #TODO: 重新组织dag的用户编排方式（拖拽？） 已完成    -->
        <!--        # 定义一个DAG                               -->
        <!--        # dag = {                                  -->
        <!--        # 'A': ['B', 'C'],  # 节点A有两条边指向B和C   -->
        <!--        # 'B': ['D'],       # 节点B有一条边指向D      -->
        <!--        # 'C': ['D'],       # 节点C有一条边指向D      -->
        <!--        # 'D': []           # 节点D没有出边          -->
        <!--        #  }                                       -->
      </div>
      <div class="svc-container-description">
        <div
          style="
            display: flex;
            justify-content: center;
            font-size: large;
            font-weight: 180;
            font-family: 'Times New Roman', Times, serif;
          "
        >
          YOU CAN DRAG THESE SERVICES TO THE PANE.
        </div>
        <ul style="list-style-type: none" class="svc-container">
          <li
            v-for="(service, index) in services"
            :key="index"
            class="svc-item"
          >
            <el-tooltip placement="top">
              <template #content>
                <div class="description">
                  {{ service.description }}
                </div>
              </template>
              <div
                class="vue-flow__node-input"
                :draggable="true"
                @dragstart="onDragStart($event, '', service)"
              >
                {{ service.name }}
              </div>
            </el-tooltip>
            <!-- <el-divider /> -->
          </li>
        </ul>
      </div>
      <!-- <el-input v-model="newInputDag" placeholder="[]" disabled="disabled"/> -->
    </div>
    <br />

    <div>
      <ElRow>
        <ElCol :span="2">
          <el-button type="warning" @click="draw">Draw</el-button>
        </ElCol>
        <ElCol :span="18"> </ElCol>
        <ElCol :span="2">
          <el-button
            type="primary"
            round
            @click="handleNewSubmit"
            v-if="drawing"
            >Add</el-button
          >
        </ElCol>
        <ElCol :span="2">
          <el-button type="primary" round @click="clearInput" v-if="drawing"
            >Clear</el-button
          >
        </ElCol>
      </ElRow>
    </div>

    <!-- 作图区域 -->
    <div
      class="draw-container"
      v-if="drawing"
      @drop="onDrop($event, nodeList, nodeMap)"
    >
      <VueFlow
        :nodes="nodeList"
        :edges="lineList"
        :default-viewport="{ zoom: 1.5 }"
        :min-zoom="0.2"
        :max-zoom="4"
        @dragover="onDragOver"
        @dragleave="onDragLeave"
      >
        <Background pattern-color="#aaa" :gap="16" />

        <MiniMap />
        <Panel class="process-panel" position="top-right">
          <div class="layout-panel">
            <button title="set horizontal layout" @click="layoutGraph('LR')">
              <Icon name="horizontal" />
            </button>

            <button title="set vertical layout" @click="layoutGraph('TB')">
              <Icon name="vertical" />
            </button>
          </div>
        </Panel>
      </VueFlow>

      <!-- <DagDraggable
        v-for="(daginfo, index) in nodeList"
        :daginfo="daginfo"
        @dblclick="startDrawing(daginfo, index)"
        @contextmenu="drawLine(index)"
        @update-position="handleUpdatePosition"
      ></DagDraggable>
      <svg ref="svg" xmlns="http://www.w3.org/2000/svg">
        <line
          v-for="(line, index) in lineList"
          :key="index"
          :x1="line.x1"
          :y1="line.y1"
          :x2="line.x2"
          :y2="line.y2"
          stroke="black"
          stroke-width="2"
        >
          {{ line }}
        </line>
      </svg> -->
    </div>
    <br /><br />
    <div>
      <h3>Current Application Dags</h3>
    </div>
    <!-- #TODO: 修改原来展示pipeline的位置，改成某种简单形式展示dag（如果实在复杂也可以考虑不展示？） 暂时跳过  -->
    <el-table :data="dagList" style="width: 100%">
      <el-table-column label="Dag Name" width="180">
        <template #default="scope">
          <div style="display: flex; align-items: center">
            <!-- <el-icon><timer /></el-icon> -->
            <span style="margin-left: 10px">{{ scope.row.dag_name }}</span>
          </div>
        </template>
      </el-table-column>
      <el-table-column label="Dag" width="320">
        <!-- <template #default="scope">
          <div>{{ scope.row.dag }}</div>
        </template> -->
        <!-- <template #default="scope">
          <div class="compact-container">
            <VueFlow
              :nodes="scope.row.nodeList"
              :edges="scope.row.lineList"
              draggable="false"
              :default-viewport="{ zoom: 4 }"
              :min-zoom="0.6"
              :max-zoom="8"
            >
            </VueFlow>
          </div>
        </template> -->
      </el-table-column>
      <el-table-column label="Action">
        <template #default="scope">
          <el-button
            size="small"
            type="danger"
            @click="deleteWorkflow(scope.$index, scope.row.dag_id)"
            >Delete
          </el-button>
        </template>
      </el-table-column>
    </el-table>

    <br />

    <br />
  </div>
</template>

<script>
import {
  ElTable,
  ElTableColumn,
  ElTooltip,
  ElTag,
  ElInput,
  ElButton,
  ElMessage,
  ElCol,
  ElRow,
} from "element-plus";
import DagDraggable from "./DagDraggable.vue";
import { ref, nextTick } from "vue";
import { Panel, VueFlow, useVueFlow, MarkerType } from "@vue-flow/core";
import { ControlButton, Controls } from "@vue-flow/controls";
import { Background } from "@vue-flow/background";
import { MiniMap } from "@vue-flow/minimap";
import useDragAndDrop from "./useDnD";
import Icon from "./Icon.vue";
import { useLayout } from "./useLayout";
export default {
  name: "DagManage",
  components: {
    nextTick,
    ElTable,
    ElTableColumn,
    ElTooltip,
    ElTag,
    ElInput,
    ElButton,
    ElCol,
    ElRow,
    ElMessage,
    DagDraggable,
    VueFlow,
    Background,
    MarkerType,
    MiniMap,
    ControlButton,
    Controls,
    Icon,
    Panel,
  },
  setup() {
    const { onInit, onNodeDragStop, onConnect, fitView } = useVueFlow();

    const { onDragOver, onDrop, onDragLeave, isDragOver, onDragStart } =
      useDragAndDrop();

    const { layout } = useLayout();

    // 线段数组
    const lineList = ref([]);
    // 节点数组
    const nodeList = ref([]);
    // 节点id到index的映射
    const nodeMap = ref({});

    async function layoutGraph(direction) {
      console.log(nodeList.value);
      console.log(lineList.value);

      nodeList.value = layout(nodeList.value, lineList.value, direction);
      nextTick(() => {
        fitView();
      });
    }

    // 生命周期初始化回调
    onInit((vueFlowInstance) => {
      vueFlowInstance.fitView();
    });
    // 节点拖拽停止
    onNodeDragStop(({ event, nodes, node }) => {
      console.log("Node Drag Stop", { event, nodes, node });
    });

    // 当边被建立时
    onConnect((connection) => {
      const line = {
        id: connection.source + "-" + connection.target,
        source: connection.source,
        target: connection.target,
        label: "",
        markerEnd: MarkerType.ArrowClosed,
      };
      lineList.value.push(line);
      console.log(nodeMap.value[connection.source]);
      nodeMap.value[connection.source].data.succ.push(connection.target);
      nodeMap.value[connection.target].data.prev.push(connection.source);
    });

    return {
      onDragOver,
      onDrop,
      onDragLeave,
      isDragOver,
      onDragStart,
      layoutGraph,
      lineList,
      nodeList,
      nodeMap,
    };
  },

  data() {
    return {
      services: [
        {
          id: 1,
          name: "car_detection",
          description: "我是car_detection",
        },
        {
          id: 2,
          name: "plate_detection",
          description: "我是plate_detection",
        },
      ],
      editInput: "",
      newInputName: "",
      newInputDag: "",
      newInputDagId: "",
      editDisabled: true,
      editingIndex: -1,
      editingRow: null,
      dagList: [
        {
          dag_id: "1",
          dag_name: "233",
          nodeList: [
            {
              id: "car_detection",
              type: "",
              position: {
                x: 127.77777099609375,
                y: 72.88887532552083,
              },
              style: {
                backgroundColor: "#578FCA",
              },
              data: {
                label: "car_detection",
                prev: [],
                succ: [],
                service_id: "car_detection",
              },
            },
            {
              id: "plate_detection",
              type: "",
              position: {
                x: 288.44443766276044,
                y: 178.88887532552084,
              },
              style: {
                backgroundColor: "#578FCA",
              },
              data: {
                label: "plate_detection",
                prev: [],
                succ: [],
                service_id: "plate_detection",
              },
            },
          ],
          lineList: [
            {
              id: "car_detection-plate_detection",
              source: "car_detection",
              target: "plate_detection",
              label: "",
              markerEnd: "arrowclosed",
            },
          ],
        },
      ],

      // 标识是否在绘制阶段
      drawing: false,
      // 数据
      configKonva: {
        width: 1920,
        height: 300,
      },

      // 是否在绘图
      isDrawing: true,
      // 绘图相关信息
      drawInfo: {},

      // 线段池,避免同一个节点出现多个边
      linePool: {},
      // 维护从节点id到边id list的映射
      sourcePool: {},
      targetPool: {},
    };
  },
  mounted() {},
  watch: {
    // 监听数据的变化
    nodeList: {
      // 更新所有lineList的值
      handler() {
        // console.log("nodeList changed");
        // 遍历nodeList 更新lineList
        for (let i = 0; i < this.nodeList.length; i++) {
          // 获取当前节点
          const node = this.nodeList[i];
          const nodeId = node.id;
          // 更新此源节点发出的边
          if (this.sourcePool[nodeId] !== undefined) {
            for (let j = 0; j < this.sourcePool[nodeId].length; j++) {
              const lineId = this.sourcePool[nodeId][j];
              if (this.linePool[lineId]) {
                this.linePool[lineId].x1 = this.nodeList[nodeId].leftEdge;
                this.linePool[lineId].y1 =
                  (this.nodeList[nodeId].topEdge +
                    this.nodeList[nodeId].bottomEdge) /
                    2 -
                  330;
              }
            }
          }
          if (this.targetPool[nodeId] !== undefined) {
            // 更新此目标节点收到的边
            for (let j = 0; j < this.targetPool[nodeId].length; j++) {
              const lineId = this.targetPool[nodeId][j];
              if (this.linePool[lineId]) {
                this.linePool[lineId].x2 = this.nodeList[nodeId].leftEdge;
                this.linePool[lineId].y2 =
                  (this.nodeList[nodeId].topEdge +
                    this.nodeList[nodeId].bottomEdge) /
                    2 -
                  330;
              }
            }
          }
        }
      },
      deep: true,
    },
    message(newValue, oldValue) {
      console.log("新值:", newValue);
      console.log("旧值:", oldValue);
    },
  },
  methods: {
    draw() {
      // 展开绘图区
      if (!this.drawing) {
        // console.log("展开绘图区");
        this.drawing = !this.drawing;
      } else {
        // 清空状态
        this.drawing = !this.drawing;

        this.lineList = [];
        this.drawInfo = {};
        this.nodeList = [];
        this.nodeMap = {};
      }
    },
    // 双击开始绘图
    startDrawing(daginfo, index) {
      console.log(daginfo, index);
      this.drawInfo.id = index;
    },
    // 结束绘图
    drawLine(index) {
      event.preventDefault();
      // 如果存在S到T的边或者存在T到S的边，则不绘制
      const Sindex = this.drawInfo.id;
      const lineId = Sindex + "-" + index;
      if (this.linePool[lineId] !== undefined) {
        ElMessage("边已存在,不需要再绘制");
        return;
      }

      // 绘制直线 x1 y1为源节点的右边缘 x2 y2为目标节点的左边缘
      const line = {};
      line.id = lineId;
      line.x1 = this.nodeList[Sindex].leftEdge;
      line.y1 =
        (this.nodeList[Sindex].topEdge + this.nodeList[Sindex].bottomEdge) / 2 -
        330;
      line.x2 = this.nodeList[index].leftEdge;
      line.y2 =
        (this.nodeList[index].topEdge + this.nodeList[index].bottomEdge) / 2 -
        330;
      this.lineList.push(line);
      this.linePool[lineId] = line;
      // 更新源点集合和目标点集合
      if (this.sourcePool[Sindex] === undefined) {
        this.sourcePool[Sindex] = [];
      }
      this.sourcePool[Sindex].push(lineId);
      if (this.targetPool[index] === undefined) {
        this.targetPool[index] = [];
      }
      this.targetPool[index].push(lineId);
      // Pool内添加相关信息
      this.nodeList[index].prev.push(Sindex);
      this.nodeList[Sindex].succ.push(index);
      // 清空相关信息
      this.drawInfo = {};
    },
    handleUpdatePosition(component) {
      // console.log("当前组件为,", component);
      const id = component.id;
      const left = component.leftSide;
      const right = component.rightSide;
      const top = component.upSide;
      const bottom = component.downSide;
      this.nodeList[id].leftEdge = left;
      this.nodeList[id].rightEdge = right;
      this.nodeList[id].topEdge = top;
      this.nodeList[id].bottomEdge = bottom;
    },
    clearInput() {
      this.newInputName = "";
      this.newInputDag = "";
      this.newInputDagId = "";
      this.nodeList = [];
      this.lineList = [];
    },
    // 删除对应的dag
    deleteWorkflow(index, dag_id) {
      this.dagList.splice(index, 1);
      console.log(dag_id);
      const content = {
        dag_id: dag_id,
      };
      fetch("/api/dag_workflow", {
        method: "DELETE",
        body: JSON.stringify(content),
      })
        .then((response) => response.json())
        .then((data) => {
          const state = data["state"];
          let msg = data["msg"];
          msg += ". Refreshing..";
          this.showMsg(state, msg);
          setTimeout(() => {
            location.reload();
          }, 500);
        })
        .catch((error) => {
          ElMessage.error("Network error");
          console.log(error);
        });
    },
    // 获得Dag图
    getDagGraph() {
      const graph = {};
      // 将所有的nodeList构建出一个图
      for (let i = 0; i < this.nodeList.length; i++) {
        const node = {
          service_id: this.nodeList[i].id,
          prev: this.nodeList[i].data.prev,
          succ: this.nodeList[i].data.succ,
        };
        console.log(node);
        graph[node.service_id] = node;

        if (this.nodeList[i].data.prev.length === 0) {
          if (graph.begin === undefined) {
            graph.begin = [];
          }
          graph.begin.push(node.id);
        }
      }
      return graph;
    },
    // 点击add按钮后触发
    handleNewSubmit() {
      if (this.newInputName === "" || this.newInputName === null) {
        ElMessage.error("Please fill the dag name");
        return;
      }
      console.log(this.nodeList);
      if (this.nodeList === undefined || this.nodeList.length === 0) {
        ElMessage.error("Please choose services");
        return;
      }
      const graph = this.getDagGraph();
      const newData = {
        dag_name: this.newInputName,
        dag: graph,
      };
      // 更新Daglist
      this.updateDagList(newData);
    },
    getDagList() {
      fetch("/api/dag_workflow")
        .then((response) => response.json())
        .then((data) => {
          // 成功获取数据后，更新dagList字段
          this.dagList = data;
          for (let i = 0; i < this.dagList.length; i++) {
            this.dagList[i].nodeList = this.layout(
              this.dagList.nodeList.value,
              this.dagList.lineList.value,
              "LR"
            );
          }
        })
        .catch((error) => {
          // console.error('Error fetching data:', error);
          console.error("Error fetching data");
        });
    },
    fetchData() {
      this.getDagList();
    },
    showMsg(state, msg) {
      if (state === "success") {
        ElMessage({
          message: msg,
          showClose: true,
          type: "success",
          duration: 3000,
        });
      } else {
        ElMessage({
          message: msg,
          showClose: true,
          type: "error",
          duration: 3000,
        });
      }
    },
    // 添加新的dag
    updateDagList(data) {
      console.log(data);
      fetch("/api/dag_workflow", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(data),
      })
        .then((response) => response.json())
        .then((data) => {
          const state = data["state"];
          const msg = data["msg"];
          console.log(state);
          this.showMsg(state, msg);
          if (state === "success") {
            this.getDagList();
            this.newInputName = "";
            this.newInputDag = "";
            this.newInputDagId = "";
            location.reload();
          }
        })
        .catch((error) => {
          console.error("Error sending data:", error);
        });
    },
    async getServiceList() {
      const response = await fetch("/api/service");
      const data = await response.json();
      this.services = data;
      console.log(this.services);
    },

    putSvcIntoList(service_id, service) {
      console.log(service_id);
      service = '"' + service + '"';
      service_id = '"' + service_id + '"';
      if (this.newInputDag === "") {
        this.newInputDag = "[" + service + "]";
        this.newInputDagId = "[" + service_id + "]";
      } else {
        service = "," + service;
        service_id = "," + service_id;
        const lastBracketIndex = this.newInputDag.lastIndexOf("]");
        const lastBracketIndex_id = this.newInputDagId.lastIndexOf("]");
        if (lastBracketIndex !== -1) {
          this.newInputDag =
            this.newInputDag.slice(0, lastBracketIndex) +
            service +
            this.newInputDag.slice(lastBracketIndex);
          this.newInputDagId =
            this.newInputDagId.slice(0, lastBracketIndex_id) +
            service_id +
            this.newInputDagId.slice(lastBracketIndex_id);
        } else {
          this.newInputDag += service;
          this.newInputDagId += service_id;
        }
      }
    },
  },
  mounted() {
    // 初次加载数据
    this.fetchData();

    // 每隔一段时间获取一次数据
    // setInterval(() => {
    //   this.fetchData();
    // }, 5000);

    this.getServiceList();
    // setInterval(this.getServiceList, 5000);
  },
};
</script>

<style scoped>
body {
  font-family: Arial, sans-serif;
  background-color: #f9f9f9;
  margin: 0;
  display: flex;
  justify-content: center;
  align-items: center;
  height: 100vh;
}

form {
  max-width: 600px;
  padding: 20px;
  background-color: #fff;
  border-radius: 8px;
  /* box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1); */
}

h3 {
  font-size: 24px;
  color: #333;
  margin-bottom: 20px;
}

input[type="text"],
input[type="file"] {
  width: calc(100% - 20px);
  padding: 10px;
  margin-bottom: 10px;
  border: 1px solid #ccc;
  border-radius: 4px;
  font-size: 16px;
}

input[type="file"] {
  cursor: pointer;
}

.draw-container {
  width: 100%;
  height: 500px;

  display: flex;
  align-items: flex-start;
  justify-content: center;
  margin-top: 15px;
  border: 1px solid #ccc;
}

.draw-container svg {
  width: 100%;
  height: 100%;
  pointer-events: none; /* 防止 SVG 阻挡鼠标事件 */
}

.compact-container {
  width: 100%;
  height: 100px;
}

.el-button {
  font-size: 16px;
  margin-right: 10px;
}

.el-button:first-child {
  margin-left: 0;
}

.outline {
  /* max-width: 600px; */
  padding: 20px;
  background-color: #fff;
  border-radius: 8px;
  /* box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1); */
}

.new-dag-font-style {
  font-size: 16px;
  margin-bottom: 20px;
  font-weight: bold;
}

.svc-container {
  display: flex;
  flex-wrap: wrap;
  padding: 3px; /* 可根据需要调整 */
  list-style-type: none;
}

.svc-container-description {
  background-color: antiquewhite;
}

.svc-item {
  margin: 2px; /* 可根据需要调整 */
  padding: 2px; /* 可根据需要调整 */
  border-radius: 10px; /* 圆角矩形 */
}

.description {
  white-space: pre-wrap; /* 保留换行符并自动换行 */
  word-wrap: break-word; /* 长单词自动换行 */
}

.el-button {
  font-size: 16px;
  margin-right: 10px;
}

.el-button:first-child {
  margin-left: 0;
}

.process-panel,
.layout-panel {
  display: flex;
  gap: 10px;
}

.process-panel {
  background-color: #2d3748;
  padding: 10px;
  border-radius: 8px;
  box-shadow: 0 0 10px rgba(0, 0, 0, 0.5);
  display: flex;
  flex-direction: column;
}

.process-panel button {
  border: none;
  cursor: pointer;
  background-color: #4a5568;
  border-radius: 8px;
  color: white;
  box-shadow: 0 0 10px rgba(0, 0, 0, 0.5);
}

.process-panel button {
  font-size: 16px;
  width: 40px;
  height: 40px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.checkbox-panel {
  display: flex;
  align-items: center;
  gap: 10px;
}

.process-panel button:hover,
.layout-panel button:hover {
  background-color: #2563eb;
  transition: background-color 0.2s;
}

.process-panel label {
  color: white;
  font-size: 12px;
}
</style>
