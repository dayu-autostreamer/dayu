
import abc

from core.lib.common import ClassFactory, ClassType, KubeConfig, Context, ConfigLoader, TaskConstant, LOGGER
from core.lib.estimation import OverheadEstimator
from core.lib.content import Task

from .steady import ContextRecord, AccuracyCalculation, OverallScheduler

import copy
from collections import deque




from .base_agent import BaseAgent

__all__ = ('SteadyAgent',)


#SteadyAgent中没有也不必实现的接口：
'''
    get_source_selection_plan
    get_initial_deployment_plan
    get_redeployment_plan
    should_generate
    get_schedule_overhead
    __call__
'''
@ClassFactory.register(ClassType.SCH_AGENT, alias='steady')
class SteadyAgent(BaseAgent, abc.ABC):

    def __init__(self, system, agent_id: int,  #system和agent_id必要的参数，其他自由添加
                 sch_param: dict, # 调度相关参数
                 steady_param: dict # 稳定调度器相关参数
                 ):
        
        # (1) 运行时情境感知相关
        self.cur_resource_table={} 
        self.cur_scenario = {}
        self.cur_policy = {}
        self.cur_task = None

        # (2) query and arch info
        self.agent_id = agent_id
        self.cloud_device = system.cloud_device
        self.edge_device = None #这个需要等用户第一次发来请求的时候才能初始化
        self.service_names = None #这个也一样，需要从dag中进行初始化


        # (3) 配置信息
        self.fps_list = system.fps_list
        self.resolution_list = system.resolution_list
        self.buffer_size_list =[ x for x in system.buffer_size_list if x > 2]
        # fps [1, 2, 3, 4, 5, 10, 15, 20, 25, 30]
        # resolution ['240p', '360p', '480p', '540p', '720p', '900p', '1080p']
        # buffer_size [10, 9, 8, 7, 6, 5, 4, 3, 2]  #不应该有2
        self.edge_serv_num_list = None #这个需要等用户第一次发来请求的时候才能初始化

        # (4) 文件路径
        self.record_path = Context.get_file_path(sch_param['record_path']) + '-'+ str(agent_id) + '-online'
        from datetime import datetime
        current_time = datetime.now()
        time_string = current_time.strftime("%Y-%m-%d-%H-%M-%S")
        self.record_path = self.record_path + '-' + time_string + '.json'

        # (5)用于设置调度器终止的必要条件
        # 注：具体来说，意思是：如果逻辑上处理的帧数量（包含因fps小于30而跳过的帧）达到上限，就停止记录更多context_record。
        # 这是为了方便及时结束记录，从而快速查看结果
        self.if_stop_record_in_single_cycle = sch_param['if_stop_record_in_single_cycle']
        self.stop_max_frame_num = sch_param['stop_max_frame_num']
        self.processed_frame_num = 0 #逻辑上已经处理的帧数量，包含那些跳过的帧
        self.if_keep_record = True

        # (6)第几次制定调度策略
        self.schedule_plan_num = 0

        # (7) 重要配置参数初始化
        self.init_param = steady_param #steady_param里传入了所有可以通过配置文件传入的初始化参数
        self.init_param['context_names'] = ['band_Mbps', 'obj_num', 'obj_size_norm', 'obj_speed']
        self.init_param['kb_path'] = Context.get_file_path(steady_param['kb_path'])
        self.init_param['steady_record_path'] = Context.get_file_path(steady_param['steady_record_path']) + '-'+ str(agent_id) + '-online' + '-' + time_string + '.json'
        self.init_param['correct_record_path'] = Context.get_file_path(steady_param['correct_record_path']) + '-'+ str(agent_id) + '-online' + '-' + time_string + '.json'
            
         
        # (8)全局调度器
        self.overall_scheduler = None

        # (9)task历史记录
        self.task_history_deque = deque(maxlen=10)

        # (10)精度采样间隔
        self.acc_sample_interval = steady_param['acc_sample_interval']



    def run(self):
        pass

    def update_scenario(self, scenario):
        self.cur_scenario = scenario

    def update_resource(self, device, resource):
        self.cur_resource_table[device] = resource

    def update_policy(self, policy):
        self.cur_policy = policy

    def update_task(self, task:Task):
        if task == None:
            print('当前新的task是None')
            return
        else:
            print('当前新的task的编号是',task.get_task_id())

        # print('进行task更新')
        cur_task = copy.deepcopy(task)

        self.cur_task = cur_task
        self.update_record(cur_task = cur_task)
        print('完成对当前task的update_record操作')
        self.update_aware(cur_task = cur_task)
        print('完成对当前task的update_aware操作')
    

    # 更新最新的执行记录
    def update_record(self, cur_task:Task):
        task = copy.deepcopy(cur_task)
        if self.if_keep_record:
            print('当前if_keep_record为真')
            context_record = ContextRecord(
                    task=task,
                    resource_table=self.cur_resource_table
                )
            ContextRecord.write_record(context_record=context_record,
                                    file_path=self.record_path)
            print('完成task的write_record写入')
        else:
            print('当前if_keep_record为假, 不可记录')
        
        # 终止是有条件的。
        if self.if_stop_record_in_single_cycle == 1:
            print('if_stop_record_in_single_cycle为真, 满足条件后结束记录')
            self.processed_frame_num +=self.get_logic_frame_num_from_task(cur_task=task)
            print('当前已处理逻辑帧数:', self.processed_frame_num, ' 逻辑帧数上限:',self.stop_max_frame_num)

            if self.processed_frame_num > self.stop_max_frame_num:
                print('逻辑帧数已达上限, 需要终止')
                self.if_keep_record = False
        else:
            print('if_stop_record_in_single_cycle为假, 一直记录')



    # 更新调度器内部感知的运行时情境
    def update_aware(self, cur_task:Task):

        # TODO：未来在此更新调度器的时候，要先调用get_conf_info_and_task_info_from_task，这里面包含对精度采样点的检查
        # 此处会执行update_scheduler函数。其中conf_info和task_info仅用来更新矫正器，因此需要基于当前采样点情况专门处理
        task = copy.deepcopy(cur_task)
        if self.overall_scheduler is not None:
            context_info = self.get_context_info_from_task(cur_task=task)
            conf_info, task_info = self.get_conf_info_task_info_from_task(cur_task=task)
            self.overall_scheduler.update_scheduler(context_info = context_info,
                                                    conf_info = conf_info,
                                                    task_info = task_info)
        

    def get_schedule_plan(self, info):
        
        # 增加启动调度的次数
        self.schedule_plan_num += 1
        if self.schedule_plan_num > 1000:
            self.schedule_plan_num = self.schedule_plan_num % 1000

        new_schedule_plan = None
        
        if self.edge_device is None:
            self.edge_device = info['source_device']
        
        if self.edge_serv_num_list is None or self.service_names is None:
            pipeline_dict = Task.extract_pipeline_deployment_from_dag_deployment(info['dag'])
            # 此处pipeline_dict不含有start和end。所有元素均为'service_name'和'execute_device'构成的键值对
            self.service_names = []
            for service_info in pipeline_dict:
                if service_info['service_name'] not in (TaskConstant.START.value, TaskConstant.END.value):
                    self.service_names.append(service_info['service_name'])
            self.edge_serv_num_list = [ i for i in range(0, len(self.service_names) + 1)]
        

        # 初始化调度器
        if self.overall_scheduler is None:

            raw_meta_data = info['meta_data']

            adjusted_delay_cons = self.init_param['delay_cons']*self.init_param['delay_cons_adjust']
            adjusted_acc_cons = self.init_param['acc_cons']*self.init_param['acc_cons_adjust']

            self.overall_scheduler = OverallScheduler(
                                kb_path = self.init_param['kb_path'],
                                service_name_pipeline = self.service_names,
                                corrector_param = self.init_param['corrector_param'],
                                queue_param = self.init_param['queue_param'],
                                knob_value_range_dict = {
                                            'fps': self.fps_list,
                                            'resolution':self.resolution_list,
                                            'buffer_size':self.buffer_size_list,
                                            'edge_serv_num':self.edge_serv_num_list
                                        },
                                delay_cons = adjusted_delay_cons,
                                acc_cons = adjusted_acc_cons,
                                delay_weight = self.init_param['delay_weight'],
                                acc_weight = self.init_param['acc_weight'],
                                default_policy = self.init_param['default_policy'],
                                raw_meta_data = raw_meta_data,
                                context_names = self.init_param['context_names'], 
                                history_lenghth = self.init_param['history_lenghth'],
                                stop_threshold = self.init_param['stop_threshold'],
                                macro_update_interval = self.init_param['macro_update_interval'],
                                context_anylze_type = self.init_param['context_anylze_type'],
                                coeff_info = self.init_param['coeff_info'],
                                steady_record_path = self.init_param['steady_record_path'],
                                correct_record_path = self.init_param['correct_record_path'],
                                cluster_threshold = self.init_param['cluster_threshold'],
                                )

        
        # 非测试模式，但是需要进行精度采样
        if self.schedule_plan_num % (self.acc_sample_interval) == 0  : # 否则，使用真实调度器
            new_schedule_plan = self.get_schedule_plan_for_acc_sample(info)
        
        # 非测试模式，正常执行调度。此时全局调度器已经完成初始化。
        else:
            task = copy.deepcopy(self.cur_task)
            task_id = None
            cur_policy = None
            context_info = None
            real_time_delay = None
            real_time_acc = None
            if task is not None:
                task_id = task.get_task_id()
                cur_policy = self.get_conf_info_from_task(cur_task = task)
                context_info = self.get_context_info_from_task(cur_task = task)
                real_time_delay = self.get_delay_from_task(cur_task = task)
                real_time_acc = self.overall_scheduler.macro_search.knowledge_base.performance_predictor.acc_pre(conf_info = context_info,
                                                                                             conf_info = cur_policy,
                                                                                             if_correct = True)

                
            
            # cur_task_id, cur_policy, context_info, real_time_delay, real_time_acc
            new_policy = self.overall_scheduler.get_schedule_plan(cur_task_id = task_id,
                                                                cur_policy = cur_policy,
                                                                context_info = context_info,
                                                                real_time_delay = real_time_delay)
            
            old_pipeline_dict = Task.extract_pipeline_deployment_from_dag_deployment(info['dag'])
            new_pipeline_dict = self.trans_edge_serv_num_to_pipeline_dict(edge_serv_num = new_policy['edge_serv_num'],
                                                                    pipeline_dict = old_pipeline_dict,
                                                                    edge_device = self.edge_device,
                                                                    cloud_device = self.cloud_device)
            new_dag = Task.extract_dag_deployment_from_pipeline_deployment(new_pipeline_dict)
        
            new_schedule_plan = {
                'fps':new_policy['fps'],
                'resolution':new_policy['resolution'],
                'buffer_size':new_policy['buffer_size'],
                'dag':new_dag,
                'encoding':'mp4v'  #编码默认不变
            }
            
        return new_schedule_plan
    
    # 用于进行精度采样的配置
    def get_schedule_plan_for_acc_sample(self,info):

        # 设置为30帧，意在逼近真实情况
        new_schedule_plan = {}
        new_schedule_plan['resolution'] = '1080p'
        new_schedule_plan['fps'] = 30
        new_schedule_plan['buffer_size'] = 2
        new_schedule_plan['encoding'] = 'mp4v'


        tmp_edge_serv_num = self.edge_serv_num_list[0]
        old_pipeline_dict = Task.extract_pipeline_deployment_from_dag_deployment(info['dag'])
        new_pipeline_dict = self.trans_edge_serv_num_to_pipeline_dict(edge_serv_num=tmp_edge_serv_num,
                                                                      pipeline_dict=old_pipeline_dict,
                                                                      edge_device=self.edge_device,
                                                                      cloud_device=self.cloud_device)
        
        new_dag = Task.extract_dag_deployment_from_pipeline_deployment(new_pipeline_dict)
        new_schedule_plan['dag'] = new_dag
        
        return new_schedule_plan
    
    # 获取一个task的平均端到端时延,按照fps进行了矫正
    def get_delay_from_task(self, cur_task:Task):

        task = copy.deepcopy(cur_task)
        exe_delay = 0
        edge_cloud_trans_delay = 0
        if_found_patition = False #判断是否找到了切分点

        pipeline_dict = task.get_pipeline_deployment_info()
        metadata = task.get_metadata()
        raw_metadata = task.get_raw_metadata()
        dag = task.get_dag()
 
        for service_info in pipeline_dict:
            # 计算执行时延
            if service_info['service_name'] not in (TaskConstant.START.value, TaskConstant.END.value):
                service = dag.get_node(service_info['service_name']).service
                exe_delay += service.get_execute_time()
            # 计算云边传输时延
            if service_info['service_name'] not in (TaskConstant.START.value, TaskConstant.END.value):
                service = dag.get_node(service_info['service_name']).service
                if service.get_execute_device() == self.cloud_device:
                    if not if_found_patition:
                        edge_cloud_trans_delay = service.get_transmit_time()
                        if_found_patition = True
        
        avg_delay = ( exe_delay + edge_cloud_trans_delay ) / metadata['buffer_size']
        raw_fps = raw_metadata['fps']
        conf_fps = metadata['fps']
        avg_delay *= (conf_fps/raw_fps)

        return avg_delay
    
    # 获取逻辑上的处理帧数量
    def get_logic_frame_num_from_task(self, cur_task:Task):
        task = copy.deepcopy(cur_task)
        metadata = task.get_metadata()
        raw_metadata = task.get_raw_metadata()
        buffer_size = metadata['buffer_size']
        fps_ratio = raw_metadata['fps'] / metadata['fps'] 
        logic_frame_num = buffer_size * fps_ratio 
        return logic_frame_num

    # 获取最新的带宽、目标大小数量和速度,可能返回为None
    def get_context_info_from_task(self, cur_task:Task):
        task = copy.deepcopy(cur_task)
        context_num = 0
        context_info = {}
        # 尝试感知带宽
        if self.edge_device != None:
            if self.edge_device in self.cur_resource_table:
                if 'available_bandwidth' in self.cur_resource_table[self.edge_device]: #确实存在可用带宽
                    context_info['band_Mbps'] = self.cur_resource_table[self.edge_device]['available_bandwidth']
                    context_num +=1
        # 尝试感知其目标大小、数量和速度
        if task != None:
            scenario_data = task.get_first_scenario_data()
            tmp_data = task.get_tmp_data()
            scenario_data['file_size'] = tmp_data['file_size']

            if 'obj_size' in scenario_data and 'obj_num' in scenario_data and 'obj_velocity' in scenario_data:
                if(len(scenario_data['obj_num']) > 0):
                    context_info['obj_size_norm'] = sum(scenario_data['obj_size']) / len(scenario_data['obj_size'])
                    context_info['obj_num'] = sum(scenario_data['obj_num']) / len(scenario_data['obj_num'])
                else:
                    context_info['obj_size_norm'] = 0
                    context_info['obj_num'] = 0
                context_info['obj_speed'] = scenario_data['obj_velocity']
                context_num +=3
        
        # 搜集到足够数量的运行时情境(4种都得到)，才能返回
        if context_num == 4:
            return context_info
        else:
            return None

    # 除了运行时情境信息，conf_info和task_info要基于当前task是否是一个精度采样点进行调整
    def get_conf_info_task_info_from_task(self, cur_task:Task):

        # 首先判断当前cur_task是不是一个精度采样点。
        # 如果是，只在task_info中放入真实精度,conf_info则从上一个task中提取, 方便作为真实精度进行矫正器内部更新
        task_info = {}
        task_info['task_id'] = cur_task.get_task_id()
        conf_info = {}
        cur_reso = cur_task.get_metadata()['resolution']
        cur_fps = cur_task.get_metadata()['fps']
        if_is_sample = 0
        if self.task_history_deque and cur_reso == '1080p' and cur_fps == 30: #如果不为空, 且当前任务符合需要
            det_task = self.task_history_deque[-1]
            det_reso = det_task.get_metadata()['resolution']
            if det_reso != '1080p': #上一个分辨率不是1080p，此时可以求精度
                real_acc_reso = AccuracyCalculation.get_real_acc(det_task=copy.deepcopy(det_task),
                                                            gt_task=copy.deepcopy(cur_task))
                
                # 此时，算出精度，但是配置信息和运行时情境信息都从上一个task里提取
                task_info['real_acc_reso'] = real_acc_reso
                conf_info = self.get_conf_info_from_task(cur_task=det_task)
                if_is_sample = 1
  
        self.task_history_deque.append(copy.deepcopy(cur_task))
        # 作为采样点加入历史序列以后，才返回task_info
        # 采样点的其他性能表现不值得
        if if_is_sample:
            return conf_info, task_info
        
        else: #非采样点时正常感知
            conf_info = self.get_conf_info_from_task(cur_task=cur_task)
            task_info = self.get_task_info_from_task(cur_task=cur_task)
            return  conf_info, task_info

    # 获取最新task对应的配置信息
    def get_conf_info_from_task(self, cur_task:Task):

        task = copy.deepcopy(cur_task)

        if task == None:
            return None
        
        metadata = task.get_metadata()
        pipeline_dict = task.get_pipeline_deployment_info()

        conf_info = {}
        conf_info['resolution'] = metadata['resolution']
        conf_info['fps'] = metadata['fps']
        conf_info['buffer_size'] = metadata['buffer_size']
        conf_info['edge_serv_num'] = self.trans_pipeline_dict_to_edge_serv_num(pipeline_dict=pipeline_dict,
                                                                               cloud_device=self.cloud_device)
        
        return conf_info

    # 获取最新task内部的其他重要信息,主要是task_id以及各个阶段的时延
    def get_task_info_from_task(self, cur_task:Task):

        
        task = copy.deepcopy(cur_task)

        if task == None:
            return None
        
        task_info = {}
        pipeline_dict = task.get_pipeline_deployment_info()
        metadata = task.get_metadata()
        dag = task.get_dag()
        edge_cloud_trans_delay = 0
        if_found_patition = False #判断是否找到了切分点

        service_num = 0
        for service_info in pipeline_dict:

            if service_info['service_name'] not in (TaskConstant.START.value, TaskConstant.END.value):
                service_num += 1
                service = dag.get_node(service_info['service_name']).service
                exe_delay =  (service.get_real_execute_time()) / metadata['buffer_size']
                wait_delay =   - exe_delay + ((service.get_execute_time()) / metadata['buffer_size'])
                
                if service_num == 1:
                    task_info['real_exe_detect'] = exe_delay
                    task_info['detect_wait_delay'] = wait_delay
                elif service_num == 2:
                    task_info['real_exe_classify'] = exe_delay
                    task_info['classify_wait_delay'] = wait_delay


            # 计算云边传输时延
            if service_info['service_name'] not in (TaskConstant.START.value, TaskConstant.END.value):
                service = dag.get_node(service_info['service_name']).service
                if service.get_execute_device() == self.cloud_device:
                    if not if_found_patition:
                        edge_cloud_trans_delay = service.get_transmit_time() / metadata['buffer_size']
                        if_found_patition = True
    
        
        # 最后标注task_id和task_trans信息
        task_info['task_id'] = task.get_task_id()
        task_info['real_trans'] = edge_cloud_trans_delay

        print('展示task_info如下:')
        print(task_info)
        
        return task_info

    # 将edge_serv_num转化为pipeline_dict
    def trans_edge_serv_num_to_pipeline_dict(self, edge_serv_num, pipeline_dict, edge_device, cloud_device):
        # 如果真的有一个start开头，那么edge_serv_num需要对应加上1
        if pipeline_dict[0]['service_name'] == TaskConstant.START.value:
            edge_serv_num+=1
        pipeline_dict = [{**p, 'execute_device': edge_device} for p in pipeline_dict[:edge_serv_num]] + \
                   [{**p, 'execute_device': cloud_device} for p in pipeline_dict[edge_serv_num:]]
        return pipeline_dict

    # 将pipeline_dict转化为edge_serv_num
    def trans_pipeline_dict_to_edge_serv_num(self, pipeline_dict, cloud_device):
        edge_serv_num = 0
        for service_info in pipeline_dict:
            if service_info['service_name'] not in (TaskConstant.START.value, TaskConstant.END.value) and \
               service_info['execute_device'] != cloud_device:
                edge_serv_num += 1
            else:
                break
        return edge_serv_num
    










            

            
    