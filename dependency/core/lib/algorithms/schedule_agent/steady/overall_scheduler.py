

from .knowledge_base import KnowledgeBase
from .steady_record import SteadyRecord
from .knowledge_base import KnowledgeBase
import asyncio
import threading
import copy
import time
import math

# 这是一个宏观微观调度器，使用线程安全的性能预估器，
# 它这里不提供针对encoding的配置维度更新


# 宏观阶段模拟：
class MacroSearch:
    def __init__(self, 
                 kb_path, 
                 service_name_pipeline, 
                 corrector_param,
                 queue_param,
                 knob_value_range_dict, 
                 delay_cons,
                 acc_cons,
                 delay_weight,
                 acc_weight,
                 default_policy,
                 raw_meta_data, 
                 context_names, 
                 history_lenghth, 
                 stop_threshold,
                 context_anylze_type,
                 cluster_threshold,
                 ):
        
        # 初始化知识库
        self.knowledge_base = KnowledgeBase(
                 kb_path = kb_path, 
                 service_name_pipeline = service_name_pipeline, 
                 corrector_param = corrector_param,
                 queue_param = queue_param,
                 knob_value_range_dict = knob_value_range_dict, 
                 delay_cons = delay_cons,
                 acc_cons = acc_cons,
                 delay_weight = delay_weight,
                 acc_weight = acc_weight,
                 raw_meta_data = raw_meta_data, 
                 stop_threshold = stop_threshold,
                 cluster_threshold = cluster_threshold
                 )
        
        # 初始化配置取值范围
        self.knob_value_range_dict = copy.deepcopy(knob_value_range_dict)

        # 初始化收敛点
        self.conv_policy = copy.deepcopy(default_policy)

        # 初始化运行时情境分布相关成员
        # context_names = ['band_Mbps', 'obj_size_norm', 'obj_num', 'obj_speed']
        self.context_history = {}
        self.history_lenghth = history_lenghth
        for context in context_names:
            self.context_history[context] = []
        # 分析运行时情境的方法
        self.context_anylze_type = context_anylze_type 
    
    # 对外接口:更新情境历史窗口，在长度超限的时候删除早期情境,或者直接删除到历史窗口为空
    # 注：需要被外部调用
    def update_context(self,context_info):


        self.knowledge_base.update_context_for_classifier_train(context_info=context_info)
        
        if context_info is not  None:
            for context in self.context_history.keys():

                self.context_history[context].append(context_info[context])

                if len(self.context_history[context]) > self.history_lenghth:
                    self.context_history[context].pop(0)
        else:

            for context in self.context_history.keys():
                while  len(self.context_history[context]) > self.history_lenghth:
                    self.context_history[context].pop(0)

    # 获取最严峻的运行时情境等运行时情境分析结果
    # 注：需要被外部调用
    def get_anylzed_context(self):
        #这里理论上需要算出运行时情境的概率分析结果
        #基于历史窗口，可以是均值，也可以是最差值
        anylzed_context = {}

        if self.context_anylze_type == 1:
            # 此时求均值
             for context in self.context_history.keys():
                
                if len(self.context_history[context]) == 0:
                    anylzed_context = {}
                    break

                else:
                    anylzed_context[context] = sum(self.context_history[context])/len(self.context_history[context])

        else: #求最差情境
            for context in self.context_history.keys():

                if len(self.context_history[context]) == 0:
                    anylzed_context = {}
                    break

                else:

                    if context in [ 'band_Mbps' , 'obj_size_norm']:
                        anylzed_context[context] = min(self.context_history[context])

                    elif context in ['obj_num', 'obj_speed']:
                        anylzed_context[context] = max(self.context_history[context])
            
        return anylzed_context

    # 基于当前配置，从某个点出发开始更新收敛点。返回一个搜索过程中产生的搜索列表
    # 注：需要被外部调用
    '''
    
    '''
    def update_conv_policy(self, start_policy, real_time_delay, real_time_acc):
        # 计算最差情境和收敛点在最差情境下的分数
        cur_context = self.get_anylzed_context()

        path_record = []

        if cur_context is not None:
            if len(cur_context)>0:

                delay,acc = self.pre_delay_and_acc(context_info = cur_context,
                                                   conf_info = start_policy)
                

                start_loss = self.cal_loss(delay=delay,acc=acc)

                # 首先，尝试使用分类器得到排序后的待修改配置参数
                sorted_knob_list = self.knowledge_base.get_sorted_knob_list(cur_policy=start_policy,
                                                                            cur_context = cur_context,
                                                                            real_time_delay = real_time_delay,
                                                                            real_time_acc = real_time_acc)
                if_greedy_for_path_record = 1

                if sorted_knob_list is not None:
                    if len(sorted_knob_list) > 0:
                        path_record = self.knowledge_base.sorted_search(policy = start_policy,
                                                                        cur_context = cur_context,
                                                                        loss = start_loss,
                                                                        sorted_knob_list = sorted_knob_list)
                        if_greedy_for_path_record = 0 #不需要传统方法

                # 尝试基于分类器得到排序后的配置旋钮失败后，用传统方法来得到path_record
                if if_greedy_for_path_record == 1:
                    all_knob_list = list(self.knob_value_range_dict.keys())
                    path_record = self.knowledge_base.greedy_search(policy = start_policy,
                                                    cur_context = cur_context,
                                                    loss = start_loss,
                                                    all_knob_list = all_knob_list)
                
                # 基于搜索结果更新搜索的收敛点
                if len(path_record) > 0:
                    self.conv_policy = copy.deepcopy(path_record[-1]['policy'])
                    conv_loss = copy.deepcopy(path_record[-1]['loss'])

        
        return path_record #返回结果

    # 结合收敛点选择待修改配置旋钮
    # 注：需要被外部调用
    def choose_knobs_by_conv_policy(self,cur_policy):
        # 基于搜索的收敛点来选择待修改的knobs
        chosen_knob_list = []
        for knob in list(cur_policy.keys()):
            if self.conv_policy[knob] is not cur_policy[knob]:
                chosen_knob_list.append(knob)

        return chosen_knob_list

    # 外部调度器通过这个接口来更新性能预估器本身
    # 注：需要被外部调用, 间接使用知识库
    def update_corrector(self, context_info, conf_info, task_info):

        self.knowledge_base.update_corrector(context_info = context_info,
                                             conf_info = conf_info,
                                             task_info = task_info)


    # 基于选定的chosen_knob_list, 来获取最小化时延和最大化精度的方向，以供负反馈使用
    # 注：需要被外部调用,间接使用知识库
    def get_delay_decrease_and_acc_increase_weight(self, cur_policy, cur_context, chosen_knob_list):
        norm_delay_decrease_weight, norm_acc_increase_weight = self.knowledge_base.get_delay_decrease_and_acc_increase_weight(cur_policy=cur_policy,
                                                                                                                              cur_context=cur_context,
                                                                                                                              chosen_knob_list=chosen_knob_list)
        return norm_delay_decrease_weight, norm_acc_increase_weight

    # 评估配置和运行时情境下的时延和精度
    # 注：需要被外部调用,间接使用知识库
    def pre_delay_and_acc(self, context_info, conf_info, record_path = None):
        delay,acc = self.knowledge_base.pre_delay_and_acc(context_info = context_info,
                                                          conf_info = conf_info,
                                                          record_path = record_path)
        return delay, acc
            
    # 基于时延和精度计算损失
    # 注：需要被外部调用,间接使用知识库
    def cal_loss(self, delay ,acc):
        loss = self.knowledge_base.cal_loss(delay=delay,
                                            acc=acc)
        return loss
           
# 微观阶段模拟
class MicroFeedback:
    
    def __init__(self,coeff_info, knob_value_range_dict):

        # 初始化超参数
        self.coeff_info = copy.deepcopy(coeff_info)
        self.coeff_info['step_coeff']['cur_value'] = self.coeff_info['step_coeff']['start_value']
        # 
        self.delay_loss = 0
        self.acc_loss = 0
        '''形如
        {
            'step_coeff':{
                    'start_value':10,
                    'add_interval':5,
                    'min_value':10,
                    'max_value':1000,
                    'cur_value':10
                },
        }
        '''
        # 初始化调整权重
        self.all_knob_weight = {}
        '''形如
        {
            'fps':fps_weight,
            'resolution':resolution_weight,
            'buffer_size':buffer_size_weight,
            'edge_serv_num':edge_serv_num_weight
        }
        '''
        # 初始化旋钮调整范围
        self.knob_value_range_dict = copy.deepcopy(knob_value_range_dict)
        '''形如
        {
            'fps':[1,2,3,4,5,10,15,25,30],
            'resolution':['240p','350p','480p','540p','720p','900p','1080p'],
            'buffer_size':[10,9,8,7,6,5,4,3,2,1],
            'edge_serv_num':[0,1,2]

        }
        '''

    # 更新当前损失情况
    def update_delay_loss_and_acc_loss(self,delay_loss, acc_loss):
        self.delay_loss = delay_loss
        self.acc_loss = acc_loss

    # 更新超参数
    def update_step_coeff(self, if_meet_cons_before, if_meet_cons_now):

        # 之前不满足约束，现在不满足约束，增加步长系数，调整幅度更大，加速逼近最优解
        if (not if_meet_cons_before) and (not if_meet_cons_now):
            self.update_coeff(coeff_name='step_coeff',if_add = True)
        # 之前不满足约束，现在满足约束，减少步长系数，调整幅度更小，防止超出约束
        elif (not if_meet_cons_before) and if_meet_cons_now:
            self.update_coeff(coeff_name='step_coeff',if_add = False)
        # 之前满足约束，现在不满足约束，增加步长系数，调整幅度更大，加速逼近最优解
        elif if_meet_cons_before and (not if_meet_cons_now):
            self.update_coeff(coeff_name='step_coeff',if_add = True)
        # 之前满足约束，现在满足约束，说明一切很好，保持不变。
        elif if_meet_cons_before and if_meet_cons_now:
           pass
        

    # 更新权重
    def update_all_knob_weight(self, delay_decrease_weight, acc_increase_weight):
        '''
        all_knob_weight:
        {
            'fps':fps_weight,
            'resolution':resolution_weight,
            'buffer_size':buffer_size_weight,
            'edge_serv_num':edge_serv_num_weight
        }
    
        '''
        # delay_decrease_weight和acc_increase_weight分别是最小化时延的方向，最大化精度的方向
        # 二者需要合成为新的调整方向。按照归一化的损失进行调整，谁损失大偏向谁

        if self.delay_loss + self.acc_loss == 0:
            delay_perf = 0
            acc_perf = 0
        else:
            delay_perf = self.delay_loss / (self.delay_loss + self.acc_loss)
            acc_perf = self.acc_loss / (self.delay_loss + self.acc_loss)

        for knob in delay_decrease_weight.keys():
            self.all_knob_weight[knob] = acc_perf * acc_increase_weight[knob]  + delay_perf * delay_decrease_weight[knob]

    # 权重更新函数
    def update_coeff(self, coeff_name, if_add):

        if if_add:
            if self.coeff_info[coeff_name]['cur_value'] + self.coeff_info[coeff_name]['add_interval'] <= self.coeff_info[coeff_name]['max_value']:
                #print('有效增加参数')
                self.coeff_info[coeff_name]['cur_value'] += self.coeff_info[coeff_name]['add_interval']
            else:
                self.coeff_info[coeff_name]['cur_value'] = self.coeff_info[coeff_name]['max_value']
        else:
            if (self.coeff_info[coeff_name]['cur_value'] / 2.0 ) >= self.coeff_info[coeff_name]['min_value']:
                #print('有效减小参数')
                self.coeff_info[coeff_name]['cur_value'] /= 2.0 
            else:
                self.coeff_info[coeff_name]['cur_value'] = self.coeff_info[coeff_name]['min_value']
    
    # 基于超参数、权重
    def get_new_policy_by_feedback(self, old_policy):

        '''policy形如
        {
            'fps':15,
            'resolution':'1080p',
            'buffer_size':5,
            'edge_serv_num':1
        
        }
        '''
        # 参数为字典的时候一定要用拷贝
        new_policy = copy.deepcopy(old_policy)
        
        for knob in list(self.knob_value_range_dict.keys()):

            if knob == 'encoding':
                continue

            knob_value_range = list(self.knob_value_range_dict[knob])

            old_knob_value = old_policy[knob]
            old_knob_idx = knob_value_range.index(old_knob_value)

            step_coeff = self.coeff_info['step_coeff']['cur_value'] 

            knob_weight = self.all_knob_weight[knob]

            new_knob_idx = min(  max(   int( old_knob_idx +  (self.delay_loss + self.acc_loss) * step_coeff * knob_weight),
                                        0), 
                                 len(knob_value_range)-1)
            
            new_knob_value =  knob_value_range[new_knob_idx]

            new_policy[knob] = new_knob_value
        
        return new_policy
            
class OverallScheduler:
    # 调用宏观和微观，来进行调度

    # 初始化过程
    def __init__(self, 
                 kb_path, 
                 service_name_pipeline, 
                 corrector_param,
                 queue_param,
                 knob_value_range_dict, 
                 delay_cons,
                 acc_cons,
                 delay_weight,
                 acc_weight,
                 default_policy,
                 raw_meta_data,
                 context_names, 
                 history_lenghth, 
                 stop_threshold,
                 macro_update_interval,
                 context_anylze_type,
                 coeff_info,
                 steady_record_path,
                 correct_record_path,
                 cluster_threshold
                 ):
        
        # 必要参数
        self.knob_value_range_dict = copy.copy(knob_value_range_dict)
        self.default_policy = default_policy
        self.delay_cons = delay_cons
        self.acc_cons = acc_cons
        self.delay_weight = delay_weight
        self.acc_weight = acc_weight
        self.stop_threshold = stop_threshold

        self.latest_real_time_loss = None

        # steady_record_path记录宏观微观调度中产生的信息
        self.steady_record_path = steady_record_path
        # correct_record_path记录性能预估器更新过程中产生的信息
        self.correct_record_path = correct_record_path


        #宏观调度阶段异步进行，宏观调度需要的输入参数如下（读取最新输入）：
        self.cur_policy = None
        self.real_time_context_info = None
        self.real_time_delay = None
        self.real_time_acc = None

        # 需要的输出的参数如下（保存执行结果）:
        self.if_need_macro_search = None
        self.path_record = None
        self.context_history = None
        self.macro_search_delay = None
        self.chosen_knob_list = None
        self.delay_decrease_weight = None 
        self.acc_increase_weight = None
        
        # 初始化宏观和微观调度模块
        self.macro_search = MacroSearch(kb_path = kb_path,
                                        service_name_pipeline = service_name_pipeline,
                                        corrector_param = corrector_param,
                                        queue_param = queue_param,
                                        knob_value_range_dict = knob_value_range_dict,
                                        delay_cons = delay_cons,
                                        acc_cons = acc_cons,
                                        delay_weight = delay_weight,
                                        acc_weight = acc_weight,
                                        default_policy = default_policy,
                                        raw_meta_data = raw_meta_data,
                                        context_names = context_names,
                                        history_lenghth = history_lenghth,
                                        stop_threshold = stop_threshold,
                                        context_anylze_type = context_anylze_type,
                                        cluster_threshold = cluster_threshold)

        self.micro_feedback = MicroFeedback(coeff_info = coeff_info,
                                            knob_value_range_dict = knob_value_range_dict)
        
        # 宏观阶段循环的周期时间，单位为s
        self.macro_update_interval = macro_update_interval
        
        # 让宏观调度模块异步运行
        self._loop = asyncio.new_event_loop() #负责异步任务处理
        self._stop_event = threading.Event() #负责线程的停止
        self._thread = threading.Thread(target=self._run_loop, daemon=True)  #负责循环的线程

        
        self._thread.start() # 启动后台线程运行异步循环

    # _run_loop在初始化后成为线程循环。使用set_event_loop，将self._loop设置为循环事件
    # 使用try finally确保事件循环最终可以被关闭。在try中调用run_until_complete，用于启动异步计算循环
    def _run_loop(self):
        # 设置事件循环在后台线程中运行
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._compute_loop())
        finally:
            self._loop.close()
    
    # 传入更新是需要锁的
    _lock = threading.Lock()

    # 用于更新线程的输入参数
    def update_parameter(self, cur_policy, real_time_context_info, real_time_delay, real_time_acc):
        with self._lock:
            self.cur_policy = copy.deepcopy(cur_policy)
            self.real_time_context_info = copy.deepcopy(real_time_context_info)
            self.real_time_delay = real_time_delay
            self.real_time_acc = real_time_acc
    
    # 用于获取一次宏观调度后的输出结果
    def get_macro_output(self):
        with self._lock:
            macro_output = {
                    'if_need_macro_search':self.if_need_macro_search,
                    'path_record':self.path_record,
                    'context_history':self.context_history,
                    'macro_search_delay':self.macro_search_delay,
                    'chosen_knob_list':self.chosen_knob_list,
                    'delay_decrease_weight':self.delay_decrease_weight,
                    'acc_increase_weight':self.acc_increase_weight
                    }
            
            for key in macro_output.keys():
                tmp_value = macro_output[key]
                macro_output[key] = copy.deepcopy(tmp_value)
            
            return macro_output
            
            
    # 用于停止后台线程
    def stop(self):
        # 停止后台线程和异步循环
        self._stop_event.set()
        self._thread.join()
    
    # 核心运行线程，要做好对None的处理
    async def _compute_loop(self):

        # 循环本体是一个普通函数
        async def compute_f(cur_policy, real_time_context_info):
            # 以下是宏观调度阶段的处理过程
            # 第一，判断是否需要更新收敛点
            #print('一次宏观调度')
            anylzed_context = self.macro_search.get_anylzed_context()
            if_need_macro_search = False
            if self.chosen_knob_list is None or len(anylzed_context)==0 :
                if_need_macro_search = True
            else:
                delay, acc = self.macro_search.pre_delay_and_acc(context_info = anylzed_context,
                                                                 conf_info = self.macro_search.conv_policy)
                conv_loss = self.macro_search.cal_loss(delay = delay,
                                                            acc = acc)
                delay, acc = self.macro_search.pre_delay_and_acc(context_info = anylzed_context,
                                                                conf_info = cur_policy)
                cur_loss = self.macro_search.cal_loss(delay = delay,
                                                            acc = acc)
                if conv_loss > cur_loss * (1-self.stop_threshold):
                    # print('conv_loss不能继续指导配置旋钮','cur_loss',cur_loss, ' conv_loss',conv_loss)
                    if_need_macro_search = True
                else:
                    # print('conv_loss还能继续指导配置旋钮','cur_loss',cur_loss, ' conv_loss',conv_loss)
                    pass

            # 第二，如果需要更新收敛点，就更新收敛点
            path_record = []
            if if_need_macro_search:
                # print('更新收敛点')
                path_record = copy.deepcopy(self.macro_search.update_conv_policy(start_policy = self.macro_search.conv_policy,
                                                                                 real_time_delay = self.real_time_delay,
                                                                                 real_time_acc = self.real_time_acc
                                                                                 ))
            
            # 第三，根据最新收敛点计算待修改配置旋钮和负反馈所需的权重
            chosen_knob_list = copy.deepcopy(self.macro_search.choose_knobs_by_conv_policy(cur_policy = cur_policy))
            # print('得到待修改配置旋钮', chosen_knob_list)
            
            # 计算负反馈所需的权重
            delay_decrease_weight, acc_increase_weight = self.macro_search.get_delay_decrease_and_acc_increase_weight(cur_policy = cur_policy,
                                                                                                                        cur_context = real_time_context_info,
                                                                                                                        chosen_knob_list = chosen_knob_list,
                                                                                                                        )
            context_history = copy.deepcopy(self.macro_search.context_history)

            
            
            # 返回处理结果
            return if_need_macro_search, path_record, context_history, chosen_knob_list, delay_decrease_weight, acc_increase_weight
        

        while not self._stop_event.is_set():

            start_time = time.time()

            # 轮旋间隔为10ms
            await asyncio.sleep(self.macro_update_interval)

            # 陷入循环，现在需要考虑为None的时候如何处理。很显然，如果为None，就不进行相关计算
            # 注意copy.deepcopy
            if self.cur_policy is not None and self.real_time_context_info is not None:

                if_need_macro_search, path_record, context_history, chosen_knob_list,delay_decrease_weight, acc_increase_weight = await compute_f(cur_policy=copy.deepcopy(self.cur_policy),
                                                                                              real_time_context_info=copy.deepcopy(self.real_time_context_info))
                
                end_time = time.time()
                macro_search_delay = end_time - start_time
                
                with self._lock:
                    self.if_need_macro_search = if_need_macro_search
                    self.path_record = path_record
                    self.context_history = context_history
                    self.macro_search_delay = macro_search_delay
                    self.chosen_knob_list = chosen_knob_list
                    self.delay_decrease_weight = delay_decrease_weight
                    self.acc_increase_weight = acc_increase_weight

    # 对宏观调度器状态的更新：用于更新调度器中的运行时情境窗口，以及矫正器本身
    def update_scheduler(self,context_info, conf_info, task_info):

        #注意，只有在不为None的时候才操作
        if context_info is not None and conf_info is not None and task_info is not None:

            self.macro_search.update_context(context_info=context_info)
            self.macro_search.update_corrector(context_info=context_info,
                                               conf_info=conf_info,
                                                task_info=task_info)

    # 微观调度，使用宏观调度的功能
    def get_schedule_plan(self, cur_task_id, cur_policy, context_info, real_time_delay, real_time_acc):

        #print('cur_policy:',cur_policy)

        # print('开始申请新调度策略')

        # 更新宏观调度阶段所需参数
        self.update_parameter(cur_policy = cur_policy,
                              real_time_context_info = context_info,
                              real_time_delay = real_time_delay,
                              real_time_acc = real_time_acc
                              )

        # 冷启动判断，如果cur_policy是None说明没有一个已经采用的策略，此时返回默认策略
        if cur_policy is None or context_info is None or real_time_delay is None or real_time_acc is None:

            print(cur_policy, context_info)
            print('冷启动')

            new_policy = copy.deepcopy(self.default_policy)

            return new_policy
        
        # 不是冷启动，cur_policy不为None，但宏观阶段没有给出结果，那就沿用cur_policy
        elif self.chosen_knob_list is None or self.delay_decrease_weight is None or self.acc_increase_weight is None:

            print('沿用旧策略')

            new_policy = copy.deepcopy(self.cur_policy)
            return new_policy
        
        # 不是冷启动，cur_policy不为None，且宏观调度给出了结果，那就使用宏观调度阶段算出的结果来进行负反馈调度
        else: 

            #print('准备负反馈')
            
            # 判断上一次是否满足时延约束，这一次是否满足时延约束
            # 用if_meet_cons_before保存上一次的结果，并在判断结束后更新为这一次的结果
            if_meet_cons_before = False
            if_meet_cons_now = False
            if self.latest_real_time_loss is not None: #上一次损失为0，说明上一次已经满足
                if self.latest_real_time_loss == 0:
                    if_meet_cons_before = True
            
            real_time_loss = self.macro_search.knowledge_base.cal_loss(delay = real_time_delay,
                                                                       acc = real_time_acc)
            
            if real_time_loss == 0: #本轮损失为0，说明本轮已经满足
                if_meet_cons_now = True
            # 判断完上一轮就要更新本轮了
            self.latest_real_time_loss = real_time_loss

            # 想要进行微观调度，需要宏观调度阶段给出的self.delay_decrease_weight和self.acc_increase_weight

            macro_output = self.get_macro_output()
            '''
            my_output = {
                'if_need_macro_search':self.if_need_macro_search,
                'path_record':self.path_record,
                'context_history':self.context_history,
                'macro_search_delay':self.macro_search_delay,
                'chosen_knob_list':self.chosen_knob_list,
                'delay_decrease_weight':self.delay_decrease_weight,
                'acc_increase_weight':self.acc_increase_weight
            }
            '''
            chosen_knob_list = macro_output['chosen_knob_list']
            delay_decrease_weight = macro_output['delay_decrease_weight']
            acc_increase_weight = macro_output['acc_increase_weight']


            # 否则直接提取宏观调度阶段输出的结果就行了,不需要更新

            # 微观阶段根据参数来计算新调度策略
            
            self.micro_feedback.update_delay_loss_and_acc_loss(delay_loss = self.macro_search.knowledge_base.cal_delay_loss(delay=real_time_delay),
                                                               acc_loss = self.macro_search.knowledge_base.cal_acc_loss(acc=real_time_acc))
            self.micro_feedback.update_step_coeff(if_meet_cons_before = if_meet_cons_before,
                                                    if_meet_cons_now = if_meet_cons_now)
            self.micro_feedback.update_all_knob_weight(delay_decrease_weight = delay_decrease_weight,
                                                    acc_increase_weight = acc_increase_weight)
            
            # 基于当前最新的cur_policy来得到新的调度策略
            new_policy = self.micro_feedback.get_new_policy_by_feedback(old_policy = cur_policy)
            # 算出new_policy后还不一定要应用。如果负反馈结果会超出时延，那就千万不要用，保留上一次的调度策略就行
            # 设置record_path，记录矫正器信息
            delay, acc = self.macro_search.pre_delay_and_acc(context_info = context_info, conf_info = new_policy,
                                                             record_path = self.correct_record_path)
            
            if_new_policy_is_bad = False
            if real_time_delay is not None and real_time_acc is not None:

                real_time_loss = self.macro_search.knowledge_base.cal_loss(delay = real_time_delay,
                                                                       acc = real_time_acc)
                if real_time_loss < 0.05:
                    # print('旧策略很好，负反馈结果不合适')
                    if_new_policy_is_bad = True
                    new_policy = cur_policy  #沿用旧调度策略
            


            # 完成微观阶段,现在要记录整个调度过程中产生的信息
            '''
                 task_id = None,
                 if_need_macro_search = None,
                 path_record = None,
                 context_history = None,
                 macro_search_delay = None,
                 chosen_knob_list = None,
                 delay_decrease_weight = None,
                 acc_increase_weight = None,
                 delay_loss = None,
                 acc_loss = None,
                 coeff_info = None,
                 if_new_policy_is_bad = None,
                 old_policy = None,
                 new_policy = None,
            '''
            steady_record = SteadyRecord(
                task_id = cur_task_id,
                if_need_macro_search = macro_output['if_need_macro_search'],
                path_record = macro_output['path_record'],
                context_history = macro_output['context_history'],
                macro_search_delay = macro_output['macro_search_delay'],
                chosen_knob_list = macro_output['chosen_knob_list'],
                delay_decrease_weight = macro_output['delay_decrease_weight'],
                acc_increase_weight = macro_output['acc_increase_weight'],
                delay_loss = self.micro_feedback.delay_loss,
                acc_loss = self.micro_feedback.acc_loss,
                coeff_info = self.micro_feedback.coeff_info,
                if_new_policy_is_bad = if_new_policy_is_bad,
                old_policy = cur_policy,
                new_policy = new_policy
            )

            SteadyRecord.write_record(steady_record = steady_record,
                                      file_path = self.steady_record_path)
            
            #print('完成录入')




            # print('最终策略',new_policy)
            return new_policy


