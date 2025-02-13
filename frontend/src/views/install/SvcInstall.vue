<template xmlns="http://www.w3.org/1999/html">
  <!-- <el-card class="box-card"> -->
  <form @submit.prevent="submitService">


    <div>
      <div>
        <h3>Scheduler Policy</h3>
      </div>
      <div>

        <el-select style="width: 100%;" v-model="selectedPolicyIndex"
                   placeholder="Please choose scheduler policy">
          <el-option v-for="(option, index) in policyOptions" :key="index" :label="option.policy_name"
                     :value="index"></el-option>
        </el-select>

      </div>
    </div>

    <br>


    <div>
      <div>
        <h3>DataSource Configuration</h3>
      </div>
      <div>
        <div>
          <el-select style="width: 100%;" v-model="selectedDatasourceIndex"
                     @change="handleDatasourceChange" placeholder="Please choose datasource config">
            <el-option v-for="(option, index) in datasourceOptions" :key="index" :label="option.source_name"
                       :value="index"></el-option>
          </el-select>
        </div>
      </div>
    </div>

    <br>


    <div>

      <div v-for="(source, index) in selectedSources" :key="index" style="margin-top: 10px;">
        <div>
          <h2> source {{ source.id }}: {{source.name}}</h2>
        </div>
        <el-select style="width: 48%;margin-top: 20px" v-model="source.dag_selected"
                   @change="updateDagSelection(index, source, source.dag_selected)"
                   placeholder="Please assign dag">
          <el-option
              v-for="(option,index) in dagOptions"
              :key="index"
              :label="option.dag_name"
              :value="option.dag_id"
          ></el-option>
        </el-select>

        <!-- #TODO: 将绑定节点的单选框改为多选框，返回一个节点列表（支持不选，不选为默认使用所有节点，需要有文字提示）   -->
        <el-select style="width: 48%;margin-top: 20px;margin-left: 4%" v-model="source.node_selected"
                   @change="updateNodeSelection(index, source, source.node_selected)"
                   placeholder="Please bind edge node">
          <el-option
              v-for="(option,index) in nodeOptions"
              :key="index"
              :label="option.name"
              :value="option.name"
          ></el-option>
        </el-select>


      </div>
    </div>


    <!-- :loading="loading" :disabled="installed === 'install'" -->
    <div style="text-align: center">
      <el-button type="primary" round native-type="submit"
                 :loading="loading" :disabled="installed === 'install'"
                 style="margin-top: 25px;"
      >Install Services
      </el-button>
    </div>
  </form>
</template>

<script>
import {ElButton} from "element-plus";
import {ElMessage} from "element-plus";

import axios from 'axios';
import {useInstallStateStore} from '/@/stores/installState';
import {ref, watch, onMounted} from 'vue';
// const install_state = useInstallStateStore()
export default {
  components: {
    ElButton,
  },
  data() {
    return {
      selectedSources: [],
      selectedDags: [],
      selectedNodes: [],
      // imageList: [],
      selectedDatasourceIndex: null,
      selectedPolicyIndex: null,


      selectedUrls: {},
      successMessage: '',
      // installed: install_state.status, // install:已安装, uninstall:未安装
      stageMessage: null,
      loading: false
    };
  },
  setup() {
    const install_state = useInstallStateStore();
    const installed = ref(null);
    const policyOptions = ref(null);
    const datasourceOptions = ref(null);
    const dagOptions = ref(null);
    const nodeOptions = ref(null);
    const getTask = async () => {


      try {
        const response = await axios.get('/api/policy');
        if (response.data !== null) {
          policyOptions.value = response.data;
        }
      } catch (error) {
        console.error('Failed to fetch policy options', error);
        ElMessage.error("System Error")
      }

      try {
        const response = await axios.get('/api/datasource');
        if (response.data !== null) {
          datasourceOptions.value = response.data;
        }
      } catch (error) {
        console.error('Failed to fetch datasource options', error);
        ElMessage.error("System Error")
      }

      try {
        const response = await axios.get('/api/dag_workflow');
        if (response.data !== null) {
          dagOptions.value = response.data;
        }
      } catch (error) {
        console.error('Failed to fetch dag options', error);
        ElMessage.error("System Error")
      }

      try {
        const response = await axios.get('/api/edge_node');
        if (response.data !== null) {
          nodeOptions.value = response.data;
        }
      } catch (error) {
        console.error('Failed to fetch node options', error);
        ElMessage.error("System Error")
      }

      try {
        const response = await axios.get('/api/install_state');
        installed.value = response.data['state'];
        if (installed.value === 'install') {
          install_state.install();
        } else {
          install_state.uninstall();
        }
      } catch (error) {
        console.error("query state error");
      }
    }

    watch(() => install_state.status, (newValue, oldValue) => {
      installed.value = newValue;
    });

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
      getTask
    };
  },
  methods: {
    async updateDagSelection(index, source, selected) {
      this.selectedDags[index] = selected;
      console.log(selected)
    },
    async updateNodeSelection(index, source, selected) {
      this.selectedNodes[index] = selected;
      console.log(selected)
    },
    async handleDatasourceChange() {
      this.successMessage = '';
      this.selectedSources = [];
      this.selectedDags = [];
      this.selectedNodes = [];

      try {
        const index = this.selectedDatasourceIndex;
        if (index !== null && index >= 0 && index < this.datasourceOptions.length) {
          const datasource = this.datasourceOptions[index]
          // console.log(englishLabel);
          // console.log(data.length)
          for (var i = 0; i < datasource.source_list.length; i++) {
            this.selectedSources.push(datasource.source_list[i]);
            this.selectedDags.push('')
            this.selectedNodes.push('')
          }
        } else {
          console.error('Invalid selected index.');
        }
      } catch (error) {
        console.error('Submission failed', error);
      }
    },

    // TODO: 提交install的js脚本检查一下需不需要改，原来是对多数据源每个数据源选择一个node，最后是一个node_list，现在应该需要node_list里面每个元素是个列表（每个数据源选择一个列表）
    submitService() {

      const policy_index = this.selectedPolicyIndex;
      if (policy_index === null || policy_index < 0 || policy_index >= this.policyOptions.length) {
        ElMessage.error('Please choose scheduler policy');
        return;
      }

      const source_index = this.selectedDatasourceIndex;
      if (source_index === null || source_index < 0 || source_index >= this.datasourceOptions.length) {
        ElMessage.error('Please choose datasource configuration');
        return;
      }


      const source_config_label = this.datasourceOptions[source_index].source_label;
      const policy_id = this.policyOptions[policy_index].policy_id

      const dag_list = Object.values(this.selectedDags);
      if (dag_list.includes('')) {
        ElMessage.error('Please assign dags for all sources');
        return;
      }

      const node_list = Object.values(this.selectedNodes);
      if (node_list.includes('')) {
        ElMessage.error('Please bind edge node for all sources');
        return;
      }


      // console.log(image_list);
      const content = {
        'source_config_label': source_config_label,
        'policy_id': policy_id,
        'dag_list':dag_list,
        'node_list':node_list
      }
      let task_info = JSON.stringify(content);

      // console.log(JSON.stringify(content));
      this.loading = true;
      fetch('/api/install', {
        method: "POST",
        body: task_info
      }).then((response) => response.json())
          .then((data) => {
            const state = data.state;
            let msg = data.msg;
            this.loading = false;
            if (state === 'success') {
              this.install_state.install();
              // this.installed = 'install';
              // console.log(this.install_state.status);
              msg += ". Refreshing.."
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

          }).catch((error) => {
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

  
  
  