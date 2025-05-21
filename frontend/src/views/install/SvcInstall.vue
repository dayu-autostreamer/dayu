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

  <br/>

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

  <br/>

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
          placeholder="Bind edge nodes (default all nodes)"
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
import {ElButton} from "element-plus";
import {ElMessage} from "element-plus";

import axios from "axios";
import {useInstallStateStore} from "/@/stores/installState";
import {ref, watch, onMounted} from "vue";

export default {
  components: {
    ElButton,
  },
  data() {
    return {
      successMessage: "",
      // installed: install_state.status
      stageMessage: null,
      loading: false,
    };
  },
  setup() {

    const INSTALL_STATE_KEY = 'savedInstallConfig';
    const DRAFT_STATE_KEY = 'savedDraftConfig';

    const selectedPolicyIndex = ref(null);
    const selectedDatasourceIndex = ref(null);
    const selectedSources = ref([]);

    const install_state = useInstallStateStore();
    const installed = ref(null);
    const policyOptions = ref(null);
    const datasourceOptions = ref(null);
    const dagOptions = ref([
      // { dag_name: "233dag", dag_id: "233dag" },
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

          const savedInstall = localStorage.getItem(INSTALL_STATE_KEY);
          if (savedInstall) {
            const parsed = JSON.parse(savedInstall);
            console.log('install configuration:' , parsed);
            selectedPolicyIndex.value = parsed.selectedPolicyIndex;
            selectedDatasourceIndex.value = parsed.selectedDatasourceIndex;

            if (datasourceOptions.value && selectedDatasourceIndex.value !== null) {
              const datasource = datasourceOptions.value[selectedDatasourceIndex.value];
              selectedSources.value = datasource.source_list.map(source => ({
                ...source,
                dag_selected: parsed.selectedSources.find(s => s.id === source.id)?.dag_selected || '',
                node_selected: parsed.selectedSources.find(s => s.id === source.id)?.node_selected || []
              }));

              console.log('selectedDatasourceIndex: ',selectedDatasourceIndex.value);
              console.log('datasource: ',datasource.value);
              console.log('selectedSources: ', selectedSources.value);
            }

          }

        } else {
          install_state.uninstall();
          const savedDraft = localStorage.getItem(DRAFT_STATE_KEY);

          if (savedDraft) {
            const parsed = JSON.parse(savedDraft);
            selectedPolicyIndex.value = parsed.selectedPolicyIndex;
            selectedDatasourceIndex.value = parsed.selectedDatasourceIndex;
            selectedSources.value = parsed.selectedSources || [];
          } else {
            selectedPolicyIndex.value = null;
            selectedDatasourceIndex.value = null;
            selectedSources.value = [];
          }
        }
      } catch
          (error) {
        console.error("query state error");
      }
    };

    watch(
        () => install_state.status,
        (newValue, oldValue) => {
          installed.value = newValue;
          if (oldValue === 'install' && newValue === 'uninstall') {
            localStorage.removeItem(INSTALL_STATE_KEY);
          }
          if (oldValue === 'uninstall' && newValue === 'install') {
            const currentConfig = {
              selectedPolicyIndex: selectedPolicyIndex.value,
              selectedDatasourceIndex: selectedDatasourceIndex.value,
              selectedSources: JSON.parse(JSON.stringify(selectedSources.value)) // 深拷贝
            };
            localStorage.setItem(INSTALL_STATE_KEY, JSON.stringify(currentConfig));
            localStorage.removeItem(DRAFT_STATE_KEY);
          }
        }
    );
    watch(
        [selectedPolicyIndex, selectedDatasourceIndex, selectedSources],
        ([policyIdx, dsIdx, sources]) => {
          if (installed.value === 'uninstall') {
            const draftData = {
              selectedPolicyIndex: policyIdx,
              selectedDatasourceIndex: dsIdx,
              selectedSources: JSON.parse(JSON.stringify(sources))
            };
            localStorage.setItem(DRAFT_STATE_KEY, JSON.stringify(draftData));
          }
        },
        {deep: true}
    );

    onMounted(async () => {
      getTask();
    });

    return {
      INSTALL_STATE_KEY,
      DRAFT_STATE_KEY,

      selectedPolicyIndex,
      selectedDatasourceIndex,
      selectedSources,

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
    }
    ,
    async updateNodeSelection(index, source, selected) {
      this.selectedSources[index].node_selected = selected;
    }
    ,
    async handleDatasourceChange() {
      this.successMessage = "";

      try {
        const index = this.selectedDatasourceIndex;
        if (
            index !== null &&
            index >= 0 &&
            index < this.datasourceOptions.length
        ) {
          const datasource = this.datasourceOptions[index];
          const newSources = datasource.source_list.map(source => ({
            ...source,
            dag_selected: '',
            node_selected: []
          }));

          if (this.installed === 'uninstall') {
            const savedDraft = localStorage.getItem(this.DRAFT_STATE_KEY);
            if (savedDraft) {
              const parsed = JSON.parse(savedDraft);
              parsed.selectedSources.forEach(savedSource => {
                const target = newSources.find(s => s.id === savedSource.id);
                if (target) {
                  target.dag_selected = savedSource.dag_selected;
                  target.node_selected = savedSource.node_selected;
                }
              });
            }
          }
          this.selectedSources = newSources;

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
      // if user assigned none then add all.
      for (let i = 0; i < this.selectedSources.length; i++) {
        if (this.selectedSources[i].node_selected.length === 0) {
          // this.selectedSources.node_selected = nodeOptions;
          this.selectedSources[i].node_selected = this.nodeOptions.map(n => n.name);
        }
      }

      // selectedSources contains all map info
      const content = {
        source_config_label: source_config_label,
        policy_id: policy_id,
        source: this.selectedSources,
      };
      let task_info = JSON.stringify(content);

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

              const installConfig = {
                selectedPolicyIndex: this.selectedPolicyIndex,
                selectedDatasourceIndex: this.selectedDatasourceIndex,
                selectedSources: JSON.parse(JSON.stringify(this.selectedSources))
              };
              localStorage.setItem(this.INSTALL_STATE_KEY, JSON.stringify(installConfig));
              localStorage.removeItem(this.DRAFT_STATE_KEY);

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
