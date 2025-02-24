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
          </li>
        </ul>
      </div>
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
    </div>
    <br /><br />
    <div>
      <h3>Current Application Dags</h3>
    </div>
    <!-- #TODO: Display Dag thumbnails (left 4 future)  -->
    <el-table :data="dagList" style="width: 100%">
      <el-table-column label="Dag Name" width="180">
        <template #default="scope">
          <div style="display: flex; align-items: center">
            <!-- <el-icon><timer /></el-icon> -->
            <span style="margin-left: 10px">{{ scope.row.dag_name }}</span>
          </div>
        </template>
      </el-table-column>
      <el-table-column label="Dag" width="320"> </el-table-column>
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

    // Edge Array
    const lineList = ref([]);
    // Node Array
    const nodeList = ref([]);
    // node id -> array index
    const nodeMap = ref({});

    // Using Dagre algorithm 2 beautify dag
    async function layoutGraph(direction) {
      nodeList.value = layout(nodeList.value, lineList.value, direction);
      nextTick(() => {
        fitView();
      });
    }

    // life cycle callback
    onInit((vueFlowInstance) => {
      vueFlowInstance.fitView();
    });
    onNodeDragStop(({ event, nodes, node }) => {
      console.log("Node Drag Stop", { event, nodes, node });
    });
    onConnect((connection) => {
      const line = {
        id: connection.source + "-" + connection.target,
        source: connection.source,
        target: connection.target,
        label: "",
        markerEnd: MarkerType.ArrowClosed,
      };
      lineList.value.push(line);
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
          description: "I am car_detection",
        },
        {
          id: 2,
          name: "plate_detection",
          description: "I amd plate_detection",
        },
      ],
      editInput: "",
      newInputName: "",
      newInputDag: "",
      newInputDagId: "",
      editDisabled: true,
      editingIndex: -1,
      editingRow: null,

      // drawing flag
      drawing: false,
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
    };
  },

  methods: {
    flushDrawData() {
      this.lineList = [];
      this.nodeList = [];
      this.nodeMap = {};
    },
    draw() {
      // show draw area
      if (!this.drawing) {
        this.drawing = !this.drawing;
      } else {
        // flush draw state
        this.drawing = !this.drawing;
        t;
      }
    },
    clearInput() {
      this.newInputName = "";
      this.newInputDag = "";
      this.newInputDagId = "";
      this.flushDrawData();
    },
    // delete dag
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

    //
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

      // get graph
      const constructDagGraph = () => {
        const graph = {};
        for (let i = 0; i < this.nodeList.length; i++) {
          const node = {
            service_id: this.nodeList[i].id,
            prev: this.nodeList[i].data.prev,
            succ: this.nodeList[i].data.succ,
          };
          graph[node.service_id] = node;

          if (this.nodeList[i].data.prev.length === 0) {
            if (graph.begin === undefined) {
              graph.begin = [];
            }
            graph.begin.push(node.id);
          }
        }
        return graph;
      };

      const graph = constructDagGraph();
      const newData = {
        dag_name: this.newInputName,
        dag: graph,
      };
      // update all Daglist
      this.updateDagList(newData);
    },
    // get dag from backen
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
    // update dag to backen
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
  },
  mounted() {
    // init dag data list
    this.fetchData();

    const getServiceInterval = () => {
      let timer;
      if (timer !== undefined) {
        clearInterval(timer);
      }
      timer = setInterval(() => {
        this.fetchData();
      }, 5000);
    };
    getServiceInterval();

    this.getServiceList();
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
