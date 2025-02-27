<template xmlns="http://www.w3.org/1999/html">
  <div>
    <div>
      <h3>Scheduler Policy</h3>
    </div>
    <div>
      <el-select
        style="width: 100%"
        v-model="selectedPolicyIndex"
        placeholder="Please choose scheduler policy"
      >
        <el-option
          v-for="(option, index) in policyOptions"
          :key="index"
          :label="option.policy_name"
          :value="index"
        ></el-option>
      </el-select>
    </div>
  </div>

  <br />

  <div>
    <div>
      <h3>DataSource Configuration</h3>
    </div>
    <div>
      <div>
        <el-select
          style="width: 100%"
          v-model="selectedDatasourceIndex"
          @change="handleDatasourceChange"
          placeholder="Please choose datasource config"
        >
          <el-option
            v-for="(option, index) in datasourceOptions"
            :key="index"
            :label="option.source_name"
            :value="index"
          ></el-option>
        </el-select>
      </div>
    </div>
  </div>

  <br />

  <div>
    <div
      v-for="(source, index) in selectedSources"
      :key="index"
      style="margin-top: 10px"
    >
      <div>
        <h2>source {{ source.id }}: {{ source.name }}</h2>
      </div>
      <el-select
        style="width: 38%; margin-top: 20px"
        v-model="source.dag_selected"
        @change="updateDagSelection(index, source, source.dag_selected)"
        placeholder="Assign dag"
      >
        <el-option
          v-for="(option, index) in dagOptions"
          :key="index"
          :label="option.dag_name"
          :value="option.dag_id"
        ></el-option>
      </el-select>

      <el-select
        style="width: 58%; margin-top: 20px; margin-left: 4%"
        v-model="source.node_selected"
        @change="updateNodeSelection(index, source, source.node_selected)"
        placeholder="Bind edge nodes(Default bind all nodes)"
        multiple
      >
        <el-option
          v-for="(option, index) in nodeOptions"
          :key="index"
          :label="option.name"
          :value="option.name"
        ></el-option>
      </el-select>
    </div>
  </div>

  <div style="text-align: center">
    <el-button
      type="primary"
      round
      native-type="submit"
      :loading="loading"
      :disabled="installed === 'install'"
      style="margin-top: 25px"
      @click="submitService"
      >Install Services
    </el-button>
  </div>
</template>

<script>
import { ElButton } from "element-plus";
import { ElMessage } from "element-plus";

import axios from "axios";
import { useInstallStateStore } from "/@/stores/installState";
import { ref, watch, onMounted } from "vue";
export default {
  components: {
    ElButton,
  },
  data() {
    return {
      selectedSources: [
        // { id: 0, name: "s1", dag_selected: "", node_selected: [] },
        // { id: 1, name: "s2", dag_selected: "", node_selected: [] },
      ],
      // imageList: [],
      selectedDatasourceIndex: null,
      selectedPolicyIndex: null,

      selectedUrls: {},
      successMessage: "",
      // installed: install_state.status, // install:已安装, uninstall:未安装
      stageMessage: null,
      loading: false,
    };
  },
  setup() {
    const install_state = useInstallStateStore();
    const installed = ref(null);
    const policyOptions = ref(null);
    const datasourceOptions = ref(null);
    // 临时数据
    const dagOptions = ref([
      // { dag_name: "233dag", dag_id: "233dag" },
      // { dag_name: "244dag", dag_id: "244dag" },
    ]);
    const nodeOptions = ref([]);
    const getTask = async () => {
      try {
        const response = await axios.get("/api/policy");
        if (response.data !== null) {
          policyOptions.value = response.data;
        }
      } catch (error) {
        console.error("Failed to fetch policy options", error);
        ElMessage.error("System Error");
      }

      try {
        const response = await axios.get("/api/datasource");
        if (response.data !== null) {
          datasourceOptions.value = response.data;
          console.log(datasourceOptions.value);
        }
      } catch (error) {
        console.error("Failed to fetch datasource options", error);
        ElMessage.error("System Error");
      }

      try {
        const response = await axios.get("/api/dag_workflow");
        if (response.data !== null) {
          dagOptions.value = response.data;
        }
      } catch (error) {
        console.error("Failed to fetch dag options", error);
        ElMessage.error("System Error");
      }

      try {
        const response = await axios.get("/api/edge_node");
        if (response.data !== null) {
          nodeOptions.value = response.data;
        }
      } catch (error) {
        console.error("Failed to fetch node options", error);
        ElMessage.error("System Error");
      }

      try {
        const response = await axios.get("/api/install_state");
        installed.value = response.data["state"];
        if (installed.value === "install") {
          install_state.install();
        } else {
          install_state.uninstall();
        }
      } catch (error) {
        console.error("query state error");
      }
    };

    watch(
      () => install_state.status,
      (newValue, oldValue) => {
        installed.value = newValue;
      }
    );

    onMounted(async () => {
      getTask();
    });

    return {
      installed,
      install_state,
      policyOptions,
      datasourceOptions,
      dagOptions,
      nodeOptions,
      getTask,
    };
  },
  methods: {
    async updateDagSelection(index, source, selected) {
      this.selectedSources[index].dag_selected = selected;
    },
    async updateNodeSelection(index, source, selected) {
      this.selectedSources[index].node_selected = selected;
    },
    async handleDatasourceChange() {
      this.successMessage = "";
      this.selectedSources = [];

      try {
        const index = this.selectedDatasourceIndex;
        if (
          index !== null &&
          index >= 0 &&
          index < this.datasourceOptions.length
        ) {
          console.log(this.datasourceOptions);
          const datasource = this.datasourceOptions[index];
          for (var i = 0; i < datasource.source_list.length; i++) {
            this.selectedSources.push(datasource.source_list[i]);
            this.selectedSources[i].node_selected = [];
            this.selectedSources[i].dag_selected = "";
          }
        } else {
          console.error("Invalid selected index.");
        }
      } catch (error) {
        console.error("Submission failed", error);
      }
    },

    submitService() {
      const policy_index = this.selectedPolicyIndex;
      if (
        policy_index === null ||
        policy_index < 0 ||
        policy_index >= this.policyOptions.length
      ) {
        ElMessage.error("Please choose scheduler policy");
        return;
      }

      const source_index = this.selectedDatasourceIndex;
      if (
        source_index === null ||
        source_index < 0 ||
        source_index >= this.datasourceOptions.length
      ) {
        ElMessage.error("Please choose datasource configuration");
        return;
      }

      const source_config_label =
        this.datasourceOptions[source_index].source_label;
      const policy_id = this.policyOptions[policy_index].policy_id;

      console.log(this.selectedSources);
      // 如果没有指定就全选
      for (let i = 0; i < this.selectedSources.length; i++) {
        if (this.selectedSources[i].node_selected.length === 0) {
          this.selectedSources.node_selected = nodeOptions;
        }
      }

      // selectedSources contains all map info
      const content = {
        source_config_label: source_config_label,
        policy_id: policy_id,
        source: this.selectedSources,
      };
      let task_info = JSON.stringify(content);

      // console.log(JSON.stringify(content));
      this.loading = true;
      fetch("/api/install", {
        method: "POST",
        body: task_info,
      })
        .then((response) => response.json())
        .then((data) => {
          const state = data.state;
          let msg = data.msg;
          this.loading = false;
          if (state === "success") {
            this.install_state.install();

            msg += ". Refreshing..";
            ElMessage({
              message: msg,
              showClose: true,
              type: "success",
              duration: 3000,
            });
            setTimeout(() => {
              location.reload();
            }, 3000);
          } else {
            ElMessage({
              message: msg,
              showClose: true,
              type: "error",
              duration: 3000,
            });
          }
        })
        .catch((error) => {
          this.loading = false;
          // console.error(error);
          ElMessage.error("Network Error", 3000);
        });
    },
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

.el-button {
  font-size: 16px;
  margin-right: 10px;
}

.el-button:first-child {
  margin-left: 0;
}
</style>
