<template>
  <div class="outline">
    <div>
      <h3>Add Application Dag</h3>
    </div>

    <div>
      <div class="new-dag-font-style">Dag Name</div>
      <el-input v-model="newInputName" placeholder="Please fill the dag name"/>
      <br/>
      <br/>
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
      <div>
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
    <br/>

    <div>
      <ElRow>
        <ElCol :span="2">
          <el-button type="warning" @click="draw">Draw</el-button>
        </ElCol>
        <ElCol :span="18"></ElCol>
        <ElCol :span="2">
          <el-button
              type="primary"
              round
              @click="handleNewSubmit"
              v-if="drawing"
          >Add
          </el-button
          >
        </ElCol>
        <ElCol :span="2">
          <el-button type="primary" round @click="clearInput" v-if="drawing"
          >Clear
          </el-button
          >
        </ElCol>
      </ElRow>
    </div>

    <!-- drawing area -->
    <div
        class="draw-container"
        v-if="drawing"
        @drop="onDrop($event, nodeList, nodeMap)"
    >
      <VueFlow
          :nodes="nodeList"
          :edges="lineList"
          :default-viewport="{ zoom: 1.5 }"
          :min-zoom="1.3"
          :max-zoom="2"
          @dragover="onDragOver"
          @dragleave="onDragLeave"
      >
        <div v-if="drawing" class="drag-tip">
          <el-icon class="tip-icon">
            <MagicStick/>
          </el-icon>
          <span>Drag service nodes to build your workflow</span>
        </div>

        <Background pattern-color="#aaa" :gap="16"/>

        <MiniMap/>
        <Panel class="process-panel" position="top-right">
          <div class="layout-panel">
            <button title="set horizontal layout" @click="layoutGraph('LR')">
              <Icon name="horizontal"/>
            </button>

            <button title="set vertical layout" @click="layoutGraph('TB')">
              <Icon name="vertical"/>
            </button>
          </div>
        </Panel>
      </VueFlow>
    </div>
    <br/><br/>
    <div>
      <h3>Current Application Dags</h3>
    </div>

    <el-table :data="dagList" style="width: 100%">
      <el-table-column label="Dag Name" width="180">
        <template #default="scope">
          <div style="display: flex; align-items: center">
            <span style="margin-left: 10px">{{ scope.row.dag_name }}</span>
          </div>
        </template>
      </el-table-column>
      <el-table-column label="Dag" width="320">
        <template #default="scope">
          <div class="dag-preview" @mouseenter="showDagDetail(scope.row)" @mouseleave="hideDagDetail">
            <!-- thumbnail view -->
            <div class="mini-dag">
              <div class="nodes">
          <span
              v-for="node in scope.row.dag.begin"
              :key="node"
              class="node"
          >{{ node }}</span>
              </div>
              <div class="edges">
                <el-icon v-for="i in 3" :key="i">
                  <Right/>
                </el-icon>
              </div>
            </div>
            <div class="stats">
              <el-tag type="info" size="small">
                <el-icon>
                  <Connection/>
                </el-icon>
                {{ Object.keys(scope.row.dag).length - 1 }} nodes
              </el-tag>
              <el-tag type="success" size="small">
                <el-icon>
                  <Link/>
                </el-icon>
                {{ countEdges(scope.row.dag) }} links
              </el-tag>
            </div>
          </div>

          <!-- floating details card -->
          <div v-if="activeDag === scope.row.dag_id" class="dag-detail-card">
            <div class="dag-title">{{ scope.row.dag_name }}</div>
            <VueFlow
                :nodes="scope.row.nodeList"
                :edges="scope.row.lineList"
                :view-fit="false"
                class="detail-flow"
            />
            <div class="hover-tip">Click to expand</div>
          </div>
        </template>
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

    <br/>

    <br/>
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
import {ref, nextTick} from "vue";
import {Panel, VueFlow, useVueFlow, MarkerType} from "@vue-flow/core";
import {ControlButton, Controls} from "@vue-flow/controls";
import {Background} from "@vue-flow/background";
import {MiniMap} from "@vue-flow/minimap";
import useDragAndDrop from "./useDnD";
import Icon from "./Icon.vue";
import {useLayout} from "./useLayout";

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
    const {onInit, onNodeDragStop, onConnect, fitView} = useVueFlow();

    const {onDragOver, onDrop, onDragLeave, isDragOver, onDragStart} =
        useDragAndDrop();

    const {layout} = useLayout();

    // Edge Array
    const lineList = ref([]);
    // Node Array
    const nodeList = ref([]);
    // node id -> array index
    const nodeMap = ref({});

    // Using Dagre algorithm to beautify dag
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
    onNodeDragStop(({event, nodes, node}) => {
      // console.log("Node Drag Stop", { event, nodes, node });
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
      services: [],
      editInput: "",
      newInputName: "",
      newInputDag: "",
      newInputDagId: "",
      editDisabled: true,
      editingIndex: -1,
      editingRow: null,

      // drawing flag
      drawing: false,
      dagList: [],
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
            graph.begin.push(node.service_id);
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
            // update dagList
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

    /*methods for dag view*/
    showDagDetail(row) {
      this.activeDag = row.dag_id;
      // Dynamically load Dag details (if not loaded)
      if (!row.nodeList) {
        this.$set(row, 'nodeList', this.parseDag(row.dag));
        this.$set(row, 'lineList', this.generateEdges(row.dag));
      }
    },

    hideDagDetail() {
      this.activeDag = null;
    },
    countEdges(dag) {
      let count = 0
      for (const node of Object.values(dag)) {
        if (node.succ && Array.isArray(node.succ)) {
          count += node.succ.length
        }
      }
      return count
    },

    parseDag(dag) {
      // Convert Dag data to VueFlow node format
      return Object.keys(dag)
          .filter(k => k !== 'begin')
          .map(k => ({
            id: k,
            position: {x: 0, y: 0},
            data: {label: k}
          }));
    },

    generateEdges(dag) {
      const edges = [];
      Object.entries(dag).forEach(([id, node]) => {
        node.succ?.forEach(target => {
          edges.push({
            source: id,
            target,
            id: `${id}-${target}`
          });
        });
      });
      return edges;
    }
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
    this.layoutGraph('LR')
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
  margin-bottom: 15px;
  font-weight: bold;
}

.svc-container {
  margin-top: -20px;
  display: flex;
  flex-wrap: wrap;
  padding: 10px;
  list-style-type: none;
}


.svc-item {
  display: inline-block;
  margin: 2px;
  padding: 2px;
  border-radius: 12px;
}

.vue-flow__node-input {
  border: 1px solid #e2e8f0; /* 柔和的蓝灰色 */
  border-radius: 8px;
  background: #ffffff;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
  transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
  width: auto;
  padding: 10px;
  font-size: 18px;
}

.vue-flow__node-input:hover {
  border-color: #94a3b8; /* 悬停时稍深的颜色 */
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}

.vue-flow__node-input2 {
  width: auto;
  padding: 5px;
  font-size: 20px;
}

.description {
  white-space: pre-wrap;
  word-wrap: break-word;
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

.drag-tip {
  position: absolute;
  top: 20px;
  left: 50%;
  transform: translateX(-50%);
  z-index: 10;
  background: rgba(255, 255, 255, 0.9);
  padding: 8px 16px;
  border-radius: 24px;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.1);
  display: flex;
  align-items: center;
  gap: 8px;
  backdrop-filter: blur(4px);
  border: 1px solid #ebeef5;
  animation: float 3s ease-in-out infinite;
}

.tip-icon {
  color: #409eff;
  font-size: 18px;
}

.drag-tip span {
  font-family: 'Segoe UI', system-ui, sans-serif;
  font-size: 14px;
  color: #606266;
  font-weight: 500;
  letter-spacing: 0.3px;
}

@keyframes float {
  0%, 100% {
    transform: translateX(-50%) translateY(0);
  }
  50% {
    transform: translateX(-50%) translateY(-4px);
  }
}


.dag-preview {
  position: relative;
  padding: 8px;
  border-radius: 8px;
  background: #f8fafc;
  transition: all 0.3s;
}

.mini-dag {
  display: flex;
  align-items: center;
  gap: 4px;
  margin-bottom: 8px;

  .node {
    padding: 2px 8px;
    background: #e2e8f0;
    border-radius: 4px;
    font-size: 12px;
  }

  .edges {
    color: #94a3b8;
    display: flex;
    gap: 2px;
  }
}

.stats {
  display: flex;
  gap: 8px;

  .el-tag {
    display: flex;
    align-items: center;
    gap: 4px;
  }
}


.dag-detail-card {
  position: fixed;
  z-index: 2000;
  width: 400px;
  height: 300px;
  background: white;
  border-radius: 12px;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.1);
  padding: 16px;
  animation: slide-in 0.3s ease;

  .dag-title {
    font-weight: 600;
    margin-bottom: 12px;
    color: #2d3748;
  }

  .detail-flow {
    height: 240px;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
  }

  .hover-tip {
    position: absolute;
    bottom: 8px;
    right: 8px;
    font-size: 12px;
    color: #94a3b8;
  }
}

@keyframes slide-in {
  from {
    opacity: 0;
    transform: translateY(8px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

</style>
