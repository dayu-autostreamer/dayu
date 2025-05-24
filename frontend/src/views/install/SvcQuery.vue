<template>
  <div class="outline">
    <div>
      <h3>Installed Services</h3>
    </div>
    <ul style="list-style-type: none" class="svc-container">
      <li v-for="(service, index) in services" :key="index" class="svc-item">
        <el-radio v-model="selected" :label="service" @change="sendRequest(service)">
          {{ service }}
        </el-radio>
      </li>
    </ul>
    <br>
    <div>
      <h3>Current Service Details<span style="visibility: hidden;">LL</span> <el-button @click="refresh()">Refresh</el-button></h3>
    </div>

    <div class="table-container">
      <table>
        <thead>
          <tr>
            <th>IP Address</th>
            <th>Hostname</th>
            <th>CPU Usage</th>
            <th>Memory Usage</th>
            <th>Bandwidth</th>
            <th>Creation Time</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="item in urlData" class="outer-li">
            <td>{{ item.ip }}</td>
            <td>{{ item.hostname }}</td>
            <td>{{ item.cpu }}</td>
            <td>{{ item.memory }}</td>
            <td>{{ item.bandwidth }}</td>
            <td>{{ item.age }}</td>
          </tr>
        </tbody>
      </table>
    </div>
    <div style="text-align: right; margin-top: 20px;">
      <el-button type="danger" @click="stopService" :loading="loading" :disabled="installed !== 'install'" >Stop Services</el-button>
    </div>
  </div>
</template>

<script>
import {useInstallStateStore} from '/@/stores/installState';
import {ElMessage} from "element-plus";
import {ref, watch} from 'vue';

export default {
  data() {
    return {
      services: [],
      urlData: null,
      selected: null,
      selected_service:null
    };
  },
  setup() {
    const install_state = useInstallStateStore();
    const installed = ref(null);
    // installed.value = true;
    watch(() => install_state.status, (newValue, oldValue) => {
      installed.value = newValue;
    });
    const loading = ref(null)
    setInterval(() => {
      fetch('/api/install_state').then(response=> response.json())
      .then(data=>{
        installed.value = data['state'];
        const val = data['state']
        if(val === 'install'){
          install_state.install();
        }else{
          install_state.uninstall();
        }
      }).catch(error=>{
        ElMessage.error("System Error",3000);
      })
    }, 2000);
    return {
      installed,
      loading,
      stopService: () => {
        loading.value = true
        fetch('/api/stop_service',{
          method:'POST'
        }).then(response => response.json())
        .then(data => {
          const state = data.state;
          let msg = data.msg;

          loading.value = false;
          // this.getServiceList();
          if(state === 'success'){
            install_state.uninstall();
            msg += ". Refreshing"
            ElMessage({
              message: msg,
              showClose: true,
              type: "success",
              duration: 3000,
            });
            setTimeout(() => {
              location.reload();
            }, 3000);  
          }else{
            ElMessage({
              message: msg,
              showClose: true,
              type: "error",
              duration: 3000,
            });
          }
        }).catch((error) => {
          loading.value = false;
          console.error(error);
          ElMessage.error("Network Error",3000);
        });
      }
    };
  },
  methods: {
    async getServiceList() {
      try {
        const response = await fetch("/api/installed_service");
        const data = await response.json();
        this.services = data;
      } catch (error) {
        ElMessage.error("System Error");
      }
      
    },
    refresh(){
      this.sendRequest(this.selected_service)
    },
    async sendRequest(service) {
      try {
        this.selected_service = service
        const response = await fetch(`/api/service_info/${service}`);
        this.urlData = await response.json();
      } catch (error) {
        ElMessage.error("System Error");
      }
    },
    
  },
  mounted() {
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
  h3 {
    font-size: 24px;
    color: #333;
    margin-bottom: 20px;
  }
  .outline{
    /* max-width: 600px; */
    padding: 20px;
    background-color: #fff;
    border-radius: 8px;
    /* box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1); */
  }

  .svc-container {
    display: flex;
    flex-wrap: wrap;
    padding: 5px; /* 可根据需要调整 */
    list-style-type: none;
  }

  .svc-item {
    list-style-type: none;
    background-color: #d6d6d6; /* 底色 */
    margin: 5px; /* 可根据需要调整 */
    padding: 5px; /* 可根据需要调整 */
    border-radius: 10px; /* 圆角矩形 */
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  }

  .svc-item:hover {
    background-color: #e0e0e0;
  }

  .el-radio {
    margin-right: 10px;
  }

  table {
    border-collapse: collapse;
    width: 100%;
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    font-family: Arial, sans-serif;
  }

  th, td {
    border: 1px solid #f2f2f2;
    padding: 10px;
    text-align: center;
  }

  th {
    background-color: #f8f9fa;
  }

  td {
    background-color: #ffffff;
  }

  tr:nth-child(even) {
    background-color: #f2f2f2;
  }

  .table-container {
    overflow-x: auto;
  }
  
  </style>
  