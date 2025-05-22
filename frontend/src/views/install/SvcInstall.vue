<template>
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
        <template v-for="(option, index) in policyOptions" :key="index">
          <el-option
              v-if="isValidIndex(index, policyOptions)"
              :value="index"
              :label="option.policy_name"
          />
        </template>
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
          <template v-for="(option, index) in datasourceOptions" :key="index">
            <el-option
                v-if="isValidIndex(index, datasourceOptions)"
                :value="index"
                :label="option.source_name"
            />
          </template>
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
        <template v-for="(option, index) in dagOptions" :key="index">
          <el-option
              v-if="isValidIndex(index, dagOptions)"
              :label="option.dag_name"
              :value="option.dag_id"
          ></el-option>
        </template>

      </el-select>

      <el-select
          style="width: 58%; margin-top: 20px; margin-left: 4%"
          v-model="source.node_selected"
          @change="updateNodeSelection(index, source, source.node_selected)"
          placeholder="Bind edge nodes (default all nodes)"
          multiple
      >
        <template v-for="(option, index) in nodeOptions" :key="index">
          <el-option
              v-if="isValidIndex(index, nodeOptions)"
              :label="option.name"
              :value="option.name"
          ></el-option>
        </template>

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
    >Install
    </el-button>

    <el-button
        type="danger"
        round
        style="margin-top: 25px; margin-left: 10px"
        @click="handleClear"
    >Clear
    </el-button>
  </div>
</template>

<script>
import {ElButton, ElMessage} from "element-plus";

import axios from "axios";
import {useInstallStateStore} from "/@/stores/installState";
import {onMounted, ref, watch} from "vue";

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
    const LENGTH_KEYS = {
      policy: 'prev_len:policy',
      datasource: 'prev_len:datasource',
      dag: 'prev_len:dag',
      node: 'prev_len:node'
    };

    const selectedPolicyIndex = ref(null);
    const selectedDatasourceIndex = ref(null);
    const selectedSources = ref([]);

    const install_state = useInstallStateStore();
    const installed = ref(null);
    const policyOptions = ref([]);
    const datasourceOptions = ref([]);
    const dagOptions = ref([]);
    const nodeOptions = ref([]);

    const isValidIndex = (index, array) => {
      return (
          Number.isSafeInteger(index) &&
          index >= 0 &&
          Array.isArray(array) &&
          index < array.length &&
          array.hasOwnProperty(index)
      );
    };

    const loadStorage = () => {
      if (installed.value === 'install') {
        const savedInstall = localStorage.getItem(INSTALL_STATE_KEY);
        return JSON.parse(savedInstall);

      } else {
        const savedDraft = localStorage.getItem(DRAFT_STATE_KEY);
        return JSON.parse(savedDraft);
      }
    };

    const updateStorage = (configuration) => {
      const currentConfig = {
        selectedPolicyIndex: configuration.selectedPolicyIndex,
        selectedDatasourceIndex: configuration.selectedDatasourceIndex,
        selectedSources: JSON.parse(JSON.stringify(configuration.selectedSources))
      };

      if (installed.value === 'install') {
        localStorage.setItem(INSTALL_STATE_KEY, JSON.stringify(currentConfig));
      } else {
        localStorage.setItem(DRAFT_STATE_KEY, JSON.stringify(currentConfig));
      }
    }

    const getTask = async () => {

      try {
        const response = await axios.get("/api/install_state");
        installed.value = response.data["state"];
        if (installed.value === "install") {
          install_state.install();
        } else {
          install_state.uninstall();
        }

      } catch (error) {
        console.error("Fail to fetch install state", error);
        ElMessage.error("Fail to fetch install state");
      }
      console.log('config storage 1: ', loadStorage())


      try {
        const response = await axios.get("/api/policy");
        if (response.data !== null) {
          const received_policy = response.data;
          const prevPL = localStorage.getItem(LENGTH_KEYS.policy);
          console.log('policy length:', prevPL, ' -> ', received_policy.length);
          if (prevPL && received_policy.length < prevPL) {
            const config = loadStorage();
            config.selectedPolicyIndex = null
            updateStorage(config)
          }
          policyOptions.value = response.data;
          localStorage.setItem(LENGTH_KEYS.policy, received_policy.length);
        }
      } catch (error) {
        console.error("Fail to fetch policy options", error);
        ElMessage.error("Fail to fetch policy options");
      }

      console.log('config storage 2: ', loadStorage())

      try {
        const response = await axios.get("/api/datasource");
        if (response.data !== null) {
          const received_datasource = response.data;
          const prevDL = localStorage.getItem(LENGTH_KEYS.datasource);
          console.log('datasource length:', prevDL, ' -> ', received_datasource.length);
          if (prevDL && received_datasource.length < prevDL) {
            const config = loadStorage();
            config.selectedDatasourceIndex = null;
            config.selectedSources = [];
            updateStorage(config);
          }
          datasourceOptions.value = response.data;
          localStorage.setItem(LENGTH_KEYS.datasource, received_datasource.length);
        }
      } catch (error) {
        console.error("Fail to fetch datasource options", error);
        ElMessage.error("Fail to fetch datasource options");
      }

      console.log('config storage 3: ', loadStorage())

      try {
        const response = await axios.get("/api/dag_workflow");
        if (response.data !== null) {
          const received_dag = response.data;
          const prevDagL = localStorage.getItem(LENGTH_KEYS.dag);
          console.log('dag length:', prevDagL, ' -> ', received_dag.length);
          if (prevDagL && received_dag.length < prevDagL) {
            const config = loadStorage();
            config.selectedSources = config.selectedSources.map(source => ({
              ...source,
              dag_selected: dagOptions.value.some(dag => dag.dag_id === source.dag_selected)
                  ? source.dag_selected
                  : ''
            }));
            updateStorage(config);
          }
          dagOptions.value = response.data;
          localStorage.setItem(LENGTH_KEYS.dag, received_dag.length);
        }
      } catch (error) {
        console.error("Fail to fetch dag options", error);
        ElMessage.error("Fail to fetch dag options");
      }

      console.log('config storage 4: ', loadStorage())

      try {
        const response = await axios.get("/api/edge_node");
        if (response.data !== null) {
          const received_node = response.data;
          const prevNodeL = localStorage.getItem(LENGTH_KEYS.node);
          console.log('node length:', prevNodeL, ' -> ', received_node.length);
          if (prevNodeL && received_node.length < prevNodeL) {
            const config = loadStorage();
            config.selectedSources = selectedSources.map(source => ({
              ...source,
              node_selected: source.node_selected.filter(nodeName =>
                  nodeOptions.value.some(node => node.name === nodeName)
              )
            }));
            updateStorage(config);
          }
          nodeOptions.value = response.data;
          localStorage.setItem(LENGTH_KEYS.node, received_node.length);
        }
      } catch (error) {
        console.error("Fail to fetch node options", error);
        ElMessage.error("Fail to fetch node options");
      }

      console.log('config storage 5: ', loadStorage())


      if (installed.value === "install") {
        const savedInstall = localStorage.getItem(INSTALL_STATE_KEY);
        if (savedInstall) {
          const parsed = JSON.parse(savedInstall);
          console.log('config from save_install: ',parsed);

          if (isValidIndex(parsed.selectedPolicyIndex, policyOptions.value)) {
            selectedPolicyIndex.value = parsed.selectedPolicyIndex;
          } else {
            selectedPolicyIndex.value = null;
          }

          if (isValidIndex(parsed.selectedDatasourceIndex, datasourceOptions.value)) {
            selectedDatasourceIndex.value = parsed.selectedDatasourceIndex;

            const datasource = datasourceOptions.value[parsed.selectedDatasourceIndex];
            selectedSources.value = datasource.source_list.map(source => ({
              ...source,
              dag_selected: parsed.selectedSources.find(s => s.id === source.id)?.dag_selected ?? '',
              node_selected: parsed.selectedSources.find(s => s.id === source.id)?.node_selected ?? []
            }));
          } else {
            selectedDatasourceIndex.value = null;
            selectedSources.value = [];
          }

        }

      } else {
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
              selectedSources: JSON.parse(JSON.stringify(selectedSources.value))
            };
            localStorage.setItem(INSTALL_STATE_KEY, JSON.stringify(currentConfig));
            localStorage.removeItem(DRAFT_STATE_KEY);
          }
        }
    );
    watch(
        [selectedPolicyIndex, selectedDatasourceIndex, selectedSources],
        ([policyIdx, dsIdx, sources]) => {

          console.log('policyIdx: ', policyIdx)
          console.log('dsIdx: ', dsIdx)
          console.log('sources: ', sources)

          if (!isValidIndex(policyIdx, policyOptions.value)) {
            console.log('policyIdx: ', policyIdx)
            console.log('policyOptions.value: ', policyOptions.value)
            policyIdx = null;
          }
          if (!isValidIndex(dsIdx, datasourceOptions.value)) {
            console.log('dsIdx: ', dsIdx)
            console.log('datasourceOptions.value: ', datasourceOptions.value)
            dsIdx = null;
            sources = [];
          }
          console.log('install: ', installed.value)

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
      isValidIndex,
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

        if (!this.isValidIndex(index, this.datasourceOptions)) {
          this.selectedDatasourceIndex = null;
          this.selectedSources = [];
          return;
        }

        if (
            index !== null &&
            index >= 0 &&
            index < this.datasourceOptions.length
        ) {
          const datasource = this.datasourceOptions[index];

          this.selectedSources = datasource.source_list.map(source => ({
            ...source,
            dag_selected: '',
            node_selected: []
          }));

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

      if (!this.isValidIndex(this.selectedPolicyIndex, this.policyOptions)) {
        ElMessage.error("Invalid policy selection");
        return;
      }

      if (!this.isValidIndex(this.selectedDatasourceIndex, this.datasourceOptions)) {
        ElMessage.error("Invalid datasource selection");
        return;
      }


      const source_config_label =
          this.datasourceOptions[source_index].source_label;
      const policy_id = this.policyOptions[policy_index].policy_id;

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
              console.log('installConfig1', installConfig);
              localStorage.setItem(this.INSTALL_STATE_KEY, JSON.stringify(installConfig));
              localStorage.removeItem(this.DRAFT_STATE_KEY);
              console.log('installConfig2', installConfig);

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

    handleClear() {
      this.selectedPolicyIndex = null;
      this.selectedDatasourceIndex = null;
      this.selectedSources = [];

      localStorage.removeItem(this.INSTALL_STATE_KEY);
      localStorage.removeItem(this.DRAFT_STATE_KEY);

      this.getTask();
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
