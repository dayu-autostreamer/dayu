 
from .integrated_safe_predictor import IntegratedSafePredictor
from .multi_label_trainer import MultiLabelTrainer
import copy
import math
import threading

# 包含知识库，并提供分类器的使用


class KnowledgeBase():


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
                 raw_meta_data, 
                 stop_threshold,
                 cluster_threshold
                 ):
        
        self.cur_task_id = -1

        self.performance_predictor=IntegratedSafePredictor(kb_path=kb_path,
                                                       service_name_pipeline=service_name_pipeline,
                                                       corrector_param=corrector_param,
                                                       queue_param=queue_param)
        self.knob_value_range_dict = copy.deepcopy(knob_value_range_dict)
        self.optional_knob_name_list = ['fps','resolution', 'buffer_size', 'edge_serv_num']
        '''
        knob_value_range_dict={
                                        'fps':[1,2,3,4,5,10,15,25,30],
                                        'resolution':['240p','360p','480p','540p','720p','900p','1080p'],
                                        'buffer_size':[10,9,8,7,6,5,4,3,2],
                                        'edge_serv_num':[0,2]
                                    }
        '''
        
        self.delay_cons = delay_cons
        self.acc_cons = acc_cons
        self.delay_weight = delay_weight
        self.acc_weight = acc_weight
        
        self.raw_meta_data = copy.deepcopy(raw_meta_data)

        self.stop_threshold = stop_threshold
        self.cluster_threshold = cluster_threshold

        self.context_info_for_classifier_train = None

        self.trainer = MultiLabelTrainer()

        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self.train_new_classifier, daemon=True)
        self._thread.start()
        
    # 更新一个用于制造新分类器的运行时情境
    def update_context_for_classifier_train(self, context_info):
        self.context_info_for_classifier_train = context_info

    # 外部调度器通过这个接口来更新性能预估器本身
    def update_corrector(self, context_info, conf_info, task_info):

        #传入一个最新的task_id
        self.cur_task_id = task_info['task_id']

        self.performance_predictor.update_corrector(context_info=context_info,
                                                    conf_info=conf_info,
                                                    task_info=task_info)

    # 基于选定的chosen_knob_list, 来获取最小化时延和最大化精度的权重，以供负反馈使用
    # 这些权重实际上是最小化时延和最大化精度的方向，最后都归一化
    def get_delay_decrease_and_acc_increase_weight(self, cur_policy, cur_context, chosen_knob_list):

        delay_decrease_weight = {}
        acc_increase_weight = {}
        cur_delay, cur_acc = self.pre_delay_and_acc(context_info = cur_context,
                                                    conf_info = cur_policy)

        for knob in list(cur_policy.keys()):
            
            if knob not in chosen_knob_list:

                delay_decrease_weight[knob] = 0
                acc_increase_weight[knob] = 0
            
            else:
   
                delay_if_add, acc_if_add, new_policy = self.get_delay_and_acc_if_knob_change_dir_units(policy = cur_policy,
                                                                                                    context = cur_context,
                                                                                                    knob = knob,
                                                                                                    dir = 1)
                delay_if_dec, acc_if_dec, new_policy = self.get_delay_and_acc_if_knob_change_dir_units(policy = cur_policy,
                                                                                                    context = cur_context,
                                                                                                    knob = knob,
                                                                                                    dir = -1)
                weight = 0

                if delay_if_add < delay_if_dec:  
                    weight = max(0, cur_delay - delay_if_add) / cur_delay

                elif delay_if_dec < delay_if_add: 
                    weight = 0 - max(0, cur_delay - delay_if_dec) / cur_delay

                else:
                    weight = 0
                
                delay_decrease_weight[knob] = weight

    
                if acc_if_add > acc_if_dec:  
                   weight = max(0, acc_if_add - cur_acc) / acc_if_add
                
                elif acc_if_dec > acc_if_add: 
                    weight = 0 - max(0, acc_if_dec - cur_acc) / acc_if_dec

                else:
                    weight = 0

                acc_increase_weight[knob] = weight
        
        # 对于delay_decrease_weight, acc_increase_weight，要归一化为长度为1的向量
        # 这样做可以防止向量太长或者太短。
        norm_delay_decrease_weight = self.normalize_vector(dictionary = delay_decrease_weight)
        norm_acc_increase_weight = self.normalize_vector(dictionary = acc_increase_weight)


        return norm_delay_decrease_weight, norm_acc_increase_weight

    # 以下函数能够将一个字典对应的向量进行等比例缩放，使之长度为1
    def normalize_vector(self, dictionary):
        # 提取字典中的所有值
        values = list(dictionary.values())
        # 计算向量的长度（欧几里得范数）
        vector_length = math.sqrt(sum(x**2 for x in values))
        # 避免除以零的情况（如果所有值都是0）
        if vector_length == 0: #此时不需要归一化，返回原始结果
            return dictionary
        # 否则，对每个值进行等比例缩放
        normalized_values = [x / vector_length for x in values]
        # 返回新的字典，键保持不变，值为缩放后的值
        return {key: value for key, value in zip(dictionary.keys(), normalized_values)}

    # 评估配置和运行时情境下的时延和精度
    def pre_delay_and_acc(self, context_info, conf_info, record_path = None):
        
        delay = self.performance_predictor.delay_pre(context_info = context_info,
                                                     conf_info=conf_info,
                                                     latest_task_id = self.cur_task_id+1,
                                                     if_correct = True,
                                                     record_path = record_path)
        acc = self.performance_predictor.acc_pre(context_info = context_info,
                                                 conf_info = conf_info,
                                                 if_correct=True)
        
        # 基于fps校正delay

        delay *= ( conf_info['fps'] / self.raw_meta_data['fps'])

        
        return delay, acc

    # 评估某配置旋钮变化后的性能:dir为1就增加knob 1个单位，dir为-1就减小knob 1个单位，dir为0就一动不动。总之就是获取指定knob移动dir个单位后的新时延和精度
    def get_delay_and_acc_if_knob_change_dir_units(self, policy, context, knob, dir):
        cur_knob_value = policy[knob]
        cur_knob_idx = self.knob_value_range_dict[knob].index(cur_knob_value)
        # print()
        # print('当前要修改的配置',knob,cur_knob_value)
        new_knob_idx = min(len(self.knob_value_range_dict[knob])-1, max(0,cur_knob_idx + dir))
        # print('初始索引和修改方向',cur_knob_idx , dir)
        # print('新索引',new_knob_idx)
        new_policy = copy.deepcopy(policy)
        new_policy[knob] = self.knob_value_range_dict[knob][new_knob_idx]
        # print('初始配置',policy)
        # print('修改配置',new_policy)
        # print()
        new_delay,new_acc = self.pre_delay_and_acc(context_info=context, conf_info=new_policy)

        return new_delay,new_acc, new_policy
    

    def cal_delay_loss(self, delay):
        delay_loss = 0
        if delay > self.delay_cons:
            delay_loss =  ((delay - self.delay_cons) / self.delay_cons ) * self.delay_weight
        return delay_loss

    def cal_acc_loss(self, acc):
        acc_loss = 0
        if acc < self.acc_cons:
            acc_loss = ( (self.acc_cons - acc) / self.acc_cons ) * self.acc_weight
        return acc_loss
    
    # 基于时延和精度计算损失
    def cal_loss(self, delay ,acc):
        
        delay_loss = self.cal_delay_loss(delay=delay)
        acc_loss = self.cal_acc_loss(acc=acc)
        
        loss = delay_loss + acc_loss

        return loss
    
    # 贪心搜索，被用于求新的收敛点。新的收敛点可以用于结合当前配置计算待修改的配置旋钮
    def greedy_search(self, policy, cur_context, loss, all_knob_list):
        # print('开始搜索')

        # pathw为搜索路径，需要记录整个搜索路径上全部的配置和相应的loss
        path_policy = copy.deepcopy(policy)
        path_loss = loss

        path_record = []
        path_record.append(
            {
                'policy':path_policy,
                'loss':path_loss
            }
        )


        # 选出一个knob后就不再考虑剩下的knob，有点像深度优先搜索
        left_knob_list = copy.deepcopy(all_knob_list)

        # 开始进行优先队列搜索，依次尝试不同的配置旋钮来进行搜索。每次都基于path_policy开始搜索。
        for i in range(0, len(left_knob_list)):
            
            # 首先选出当前配置下最值得的修改的knob以及相应的方向，注意best_dir可能为0
            best_knob, best_dir, best_loss = self.choose_knob_by_loss(cur_policy = path_policy,
                                                    cur_context = cur_context,
                                                    cur_loss = path_loss,
                                                    knob_list = left_knob_list)
   
            
            # 如果为0，说明搜索可以终止了
            if best_dir == 0:
                break

            # 如果为1或者-1，就沿着这个knob和相应的方向一路搜索下去
            elif best_dir in [1,-1]:

                tmp_policy, tmp_loss = self.get_best_policy_loss_in_one_dir(cur_policy = path_policy,
                                                                              cur_loss = path_loss,
                                                                              cur_context = cur_context,
                                                                              knob = best_knob,
                                                                              dir = best_dir)
                path_record.append(
                    {
                        'policy':tmp_policy,
                        'loss':tmp_loss
                    }
                )
                
                # 如果loss已经小于0.05且十分有限，就停止继续搜索
                if path_loss < 0.05 and tmp_loss < 0.05:
                    if 0 <= (path_loss - tmp_loss) <= self.stop_threshold*path_loss:
                        break
                
                # 如果无法终止搜索，那就继续前行
                path_policy = tmp_policy
                path_loss = tmp_loss
                left_knob_list.remove(best_knob)
            
            else:
                assert('wrong dir in choose_knobs_by_search')
        return path_record
    

    def get_sorted_knob_list(self, cur_policy, cur_context, real_time_delay, real_time_acc):

        '''
        pred_res = {}
        tmp_idx = 0
        for knob_name in self.optional_knob_name_list:
            pred_res[knob_name] = pred_proba[tmp_idx]
            tmp_idx += 1
        return pred_res
        '''
        # 计算每一个旋钮值得被修改的概率
        pred_res = self.use_classifier(cur_policy=cur_policy,
                                       cur_context=cur_context)
        if pred_res == None:
            #print('------基于分类器排序的尝试失败了------')
            return None
        # 按照概率从大到小排序
        sorted_keys = sorted(pred_res, key=lambda k: pred_res[k], reverse = True)
        print('成功基于分类器进行排序', sorted_keys)

        sorted_knob_list = []

        # 优先考虑不敏感的配置
        if real_time_acc > self.acc_cons == 0 and real_time_delay > self.delay_cons:
            for key in sorted_keys:
                if key in ['buffer_size','edge_serv_num']:
                    sorted_knob_list.append(key)
            for key in sorted_keys:
                if key not in ['buffer_size','edge_serv_num']:
                    sorted_knob_list.append(key)
        else:
            sorted_knob_list = sorted_keys
        
        return sorted_knob_list

    # 贪心搜索，用于在分类器提供的配置参数排序的基础上计算待修改的配置旋钮。
    # 
    def sorted_search(self, policy, cur_context, loss, sorted_knob_list):

        # pathw为搜索路径，需要记录整个搜索路径上全部的配置和相应的loss
        path_policy = copy.deepcopy(policy)
        path_loss = loss

        path_record = []
        path_record.append(
            {
                'policy':path_policy,
                'loss':path_loss
            }
        )
        # 开始进行优先队列搜索，依次尝试不同的配置旋钮来进行搜索。每次都基于path_policy开始搜索。
        for knob in sorted_knob_list:

            best_dir = self.get_best_dir_of_knob(cur_policy = path_policy,
                                                 cur_context = cur_context,
                                                 cur_loss = path_loss,
                                                 knob = knob)
            
            # 如果为0，说明搜索可以终止了
            if best_dir == 0:
                break

            # 如果为1或者-1，就沿着这个knob和相应的方向一路搜索下去
            elif best_dir in [1,-1]:
                tmp_policy, tmp_loss = self.get_best_policy_loss_in_one_dir(cur_policy = path_policy,
                                                                              cur_loss = path_loss,
                                                                              cur_context = cur_context,
                                                                              knob = knob,
                                                                              dir = best_dir)
                path_record.append(
                    {
                        'policy':tmp_policy,
                        'loss':tmp_loss
                    }
                )
                # 如果loss已经小于0.05且十分有限，就停止继续搜索
                if path_loss < 0.05 and tmp_loss < 0.05:
                    if 0 <= (path_loss - tmp_loss) <= self.stop_threshold*path_loss:
                        break
                
                # 如果无法终止搜索，那就继续前行
                path_policy = tmp_policy
                path_loss = tmp_loss
            else:
                assert('wrong dir in choose_knobs_by_search')

        return path_record

   # 在搜索过程中，选择能够让loss最小的配置
    def choose_knob_by_loss(self, cur_policy, cur_context, cur_loss, knob_list):
        best_knob = ''
        best_dir = 0
        best_loss = cur_loss
        for knob in knob_list:
            for dir in [-1,1]:
                new_delay, new_acc, new_policy = self.get_delay_and_acc_if_knob_change_dir_units(policy = cur_policy,
                                                                            context = cur_context,
                                                                            knob = knob,
                                                                            dir = dir
                                                                            )
                new_loss = self.cal_loss(delay = new_delay,
                                        acc = new_acc)
                if new_loss < best_loss:
                    best_loss = new_loss
                    best_knob = knob
                    best_dir = dir

        return best_knob, best_dir, best_loss
    
    # 判断当前旋钮朝什么方向调整最有利
    def get_best_dir_of_knob(self, cur_policy, cur_context, cur_loss, knob):
        best_dir = 0
        best_loss = cur_loss
        for dir in [-1,1]:
            new_delay, new_acc, new_policy = self.get_delay_and_acc_if_knob_change_dir_units(policy = cur_policy,
                                                                        context = cur_context,
                                                                        knob = knob,
                                                                        dir = dir
                                                                        )
            new_loss = self.cal_loss(delay = new_delay,
                                    acc = new_acc)
            if new_loss < best_loss:
                best_loss = new_loss
                best_dir = dir

        return best_dir


    # 在搜索过程中，基于一个配置旋钮以及方向，沿着这个方向寻找能够使得loss最小的配置，并返回最终的loss和配置
    def get_best_policy_loss_in_one_dir(self, cur_policy, cur_loss, cur_context, knob, dir):
        if dir not in [1,-1]:
            assert('Wrong dir in change_knob_in_one_dir')
        best_policy = copy.deepcopy(cur_policy)
        best_loss = cur_loss
        dir_num = 0
        while True:
            dir_num += 1
            new_delay,new_acc, new_policy = self.get_delay_and_acc_if_knob_change_dir_units(policy = best_policy,
                                                                                            context = cur_context,
                                                                                            knob = knob,
                                                                                            dir = dir_num*dir)
            new_loss = self.cal_loss(delay = new_delay, acc = new_acc)
            if new_loss < best_loss:
                best_loss = new_loss
                best_policy = new_policy
            else:
                break
        return best_policy, best_loss
    

    # ——————以下是分类器相关——————
    # 用于进行分类器训练的循环
    def train_new_classifier(self):
        while True:

            if self.context_info_for_classifier_train == None:
                pass

            else:
                new_context = copy.deepcopy(self.context_info_for_classifier_train)
                cluster_name, extreme_context, if_belong_cluster = self.process_context_for_cluster(cur_context=new_context)
                # 可以算出聚类名和极端运行时情境，就可以进行训练
                if cluster_name is not  None and extreme_context is not None:

                    print('------开始训练新分类器------')

                    x_data_list, y_data_list = self.get_train_data(extreme_context=extreme_context)
                    print('------训练数据生成完毕------')

                    self.trainer.train(X = x_data_list,
                                       y = y_data_list,
                                       name = cluster_name)
                    print('------训练器训练结束------')
    
    # 用于获取待修改配置旋钮即各自的预测概率
    def use_classifier(self, cur_policy, cur_context):

        print('开始使用新分类器')

        if self.cluster_threshold == 0 :
            print('---------阈值太小，不足以使用分类器------')
            return None

        cluster_name, extreme_context, if_belong_cluster = self.process_context_for_cluster(cur_context=cur_context)
        print('准备使用新分类器:',cluster_name, extreme_context, if_belong_cluster)
        print('当前可用分类器:',self.trainer.list_models())
    
        if cluster_name is not None:
            if if_belong_cluster == 1:

                x_data = self.trans_policy_to_x_data(policy = cur_policy)
                pred_proba = self.trainer.get_pred_proba(x=x_data,
                                                         name=cluster_name)
                if pred_proba == None:
                    return None
                
                # 返回一个预测结果pred_res，每一个配置旋钮对应一个分数
                pred_res = {}
                tmp_idx = 0
                for knob_name in self.optional_knob_name_list:
                    pred_res[knob_name] = pred_proba[tmp_idx]
                    tmp_idx += 1
                return pred_res

        return None
                

    # 判断当前运行时情境应当对应哪一种分类器，其标识符，以及是否在聚类阈值内
    # 首先，必须确保重要的运行时情境都在其中，才可以进行计算
    # 返回值为None的时候说明聚类失败
    def process_context_for_cluster(self, cur_context):

        if_belong_cluster = 1
        extreme_context = {}
        cluster_name = '' # 标识聚类的字符串

        if self.cluster_threshold == 0:
            return None, None, None

        elif ( 'band_Mbps' in cur_context ) and ( 'obj_size_norm' in cur_context ) and \
           ( 'obj_num' in cur_context ) and  ( 'obj_speed' in cur_context):
            
            band_Mbps =  cur_context['band_Mbps']
            obj_size_norm = cur_context['obj_size_norm']
            obj_num = cur_context['obj_num']
            obj_speed = cur_context['obj_speed']

            # 5种带宽。极端值往小了取。
            if band_Mbps < 0.1:
                cluster_name += '0'
                extreme_context['band_Mbps'] = 0
            elif band_Mbps < 1:
                cluster_name += '1'
                extreme_context['band_Mbps'] = 0.1
                if band_Mbps > 0.1 + (1-0.1)*self.cluster_threshold:
                    if_belong_cluster = 0
            elif band_Mbps < 5:
                cluster_name += '2'
                extreme_context['band_Mbps'] = 1
                if band_Mbps > 1 + (5-1)*self.cluster_threshold:
                    if_belong_cluster = 0
            elif band_Mbps < 10:
                cluster_name += '3'
                extreme_context['band_Mbps'] = 5
                if band_Mbps > 5 + (10-5)*self.cluster_threshold:
                    if_belong_cluster = 0
            else:
                cluster_name += '4'
                extreme_context['band_Mbps'] = 10
                if band_Mbps > 10 + 10*self.cluster_threshold:
                    if_belong_cluster = 0

            # 5种大小。极端值往小了取。
            if obj_size_norm < 0.05:
               cluster_name += '0'
               extreme_context['obj_size_norm'] = 0.01
            elif obj_size_norm < 0.1:
                cluster_name += '1'
                extreme_context['obj_size_norm'] = 0.05
                if obj_size_norm > 0.05 + (0.1-0.05)*self.cluster_threshold:
                    if_belong_cluster = 0
            elif obj_size_norm < 0.2:
                cluster_name += '2'
                extreme_context['obj_size_norm'] = 0.1
                if obj_size_norm > 0.1 + (0.2-0.1)*self.cluster_threshold:
                    if_belong_cluster = 0
            elif obj_size_norm < 0.3:
                cluster_name += '3'
                extreme_context['obj_size_norm'] = 0.2
                if obj_size_norm > 0.2 + (0.3-0.2)*self.cluster_threshold:
                    if_belong_cluster = 0
            else:
                cluster_name += '4'
                extreme_context['obj_size_norm'] = 0.3
                if obj_size_norm > 0.3 + 0.3*self.cluster_threshold:
                    if_belong_cluster = 0
            
            # 4种数量。极端值往大了取。
            if obj_num < 1:
                cluster_name += '0'
                extreme_context['obj_num'] = 1
            elif obj_num < 5:
                cluster_name += '1'
                extreme_context['obj_num'] = 5
                if obj_num < 5 - (5-1)*self.cluster_threshold:
                    if_belong_cluster = 0
            elif obj_num < 10:
                cluster_name += '2'
                extreme_context['obj_num'] = 10
                if obj_num < 10 - (10-5)*self.cluster_threshold:
                    if_belong_cluster = 0
            else:
                cluster_name += '3'
                extreme_context['obj_num'] = 20
                if obj_num < 20 - (20-10)*self.cluster_threshold or obj_num > 20:
                    if_belong_cluster = 0
            
            # 4种速度。极端值往大了取。
            if obj_speed < 260:
                cluster_name += '0'
                extreme_context['obj_speed'] = 260
            elif obj_speed < 520:
                cluster_name += '1'
                extreme_context['obj_speed'] = 520
                if obj_speed < 520 - (520-260)*self.cluster_threshold:
                    if_belong_cluster = 0
            elif obj_speed < 780:
                cluster_name += '2'
                extreme_context['obj_speed'] = 780
                if obj_speed < 780 - (780-520)*self.cluster_threshold:
                    if_belong_cluster = 0
            else:
                cluster_name += '3'
                extreme_context['obj_speed'] = 1500
                if obj_speed < 1500 - (1500-780)*self.cluster_threshold or obj_speed > 1500:
                    if_belong_cluster = 0
            
            return cluster_name, extreme_context, if_belong_cluster
        
        else:
            return None, None, None


            
    def get_train_data(self, extreme_context):
        '''
        self.optional_knob_name_list = ['fps','resolution', 'buffer_size', 'edge_serv_num']
        self.knob_value_range_dict={
                                        'fps':[1,2,3,4,5,10,15,25,30],
                                        'resolution':['240p','360p','480p','540p','720p','900p','1080p'],
                                        'buffer_size':[10,9,8,7,6,5,4,3,2],
                                        'edge_serv_num':[0,1,2]
                                    }
        
        '''

        x_data_list = []
        y_data_list = []

        for tmp_fps in self.knob_value_range_dict['fps']:
            for tmp_resolution in self.knob_value_range_dict['resolution']:
                for tmp_buffer_size in self.knob_value_range_dict['buffer_size']:
                    for tmp_edge_serv_num in self.knob_value_range_dict['edge_serv_num']:
                        
                        tmp_policy = {
                            'fps': tmp_fps,
                            'resolution': tmp_resolution,
                            'buffer_size': tmp_buffer_size,
                            'edge_serv_num': tmp_edge_serv_num
                        }

                        # 将调度策略转化为一个向量
                        x_data = self.trans_policy_to_x_data(policy=tmp_policy)
                        # 获取待修改配置旋钮
                        chosen_knob_set = self.get_chosen_knob_set_for_train(start_policy=tmp_policy,
                                                                               cur_context=extreme_context)
                        # 将待修改配置旋钮转化为一个标签
                        y_data = self.trans_knob_set_to_y_data(knob_set=chosen_knob_set)

                        x_data_list.append(x_data)
                        y_data_list.append(y_data)
        
        return x_data_list, y_data_list


    # 化策略为向量。将配置旋钮的取值表示为整数
    def trans_policy_to_x_data(self, policy ):

        x_data = []
        # ['fps','resolution', 'buffer_size', 'edge_serv_num']
        for knob_name in self.optional_knob_name_list:
            tmp_knob_value = policy[knob_name]
            tmp_knob_idx = self.knob_value_range_dict[knob_name].index(tmp_knob_value)
            x_data.append(tmp_knob_idx )

        return x_data
    
    # 化待修改配置集合为标签，存在即为1。
    def trans_knob_set_to_y_data(self, knob_set):

        y_data = []
        for knob_name in self.optional_knob_name_list:
            if knob_name in knob_set:
                y_data.append(1)
            else:
                y_data.append(0)
        return y_data



    # 为了训练分类器, 需要生成训练数据
    # 输入一个调度策略，以及对应运行时情境，判断哪些配置参数需要调整，指导训练数据的生成
    def get_chosen_knob_set_for_train(self, start_policy, cur_context):

        delay,acc = self.pre_delay_and_acc(context_info = cur_context,
                                            conf_info = start_policy)
        start_loss = self.cal_loss(delay=delay,acc=acc)
        all_knob_list = list(self.knob_value_range_dict.keys())
        path_record = self.greedy_search(policy = start_policy,
                                        cur_context = cur_context,
                                        loss = start_loss,
                                        all_knob_list = all_knob_list)
        conv_policy = copy.deepcopy(path_record[-1]['policy'])

        chosen_knob_list = []
        for knob in list(start_policy.keys()):
            if conv_policy[knob] is not start_policy[knob]:
                chosen_knob_list.append(knob)
        
        # 返回一个待修改配置参数构成的集合
        return chosen_knob_list

        
        

        
    

