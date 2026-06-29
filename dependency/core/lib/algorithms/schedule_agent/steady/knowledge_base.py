from .integrated_safe_predictor import IntegratedSafePredictor
from .multi_label_trainer import MultiLabelTrainer
from core.lib.common import LOGGER
import copy
import math
import threading
import time


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

        self.performance_predictor = IntegratedSafePredictor(kb_path=kb_path,
                                                             service_name_pipeline=service_name_pipeline,
                                                             corrector_param=corrector_param,
                                                             queue_param=queue_param)
        self.knob_value_range_dict = copy.deepcopy(knob_value_range_dict)
        self.optional_knob_name_list = ['fps', 'resolution', 'buffer_size', 'edge_serv_num']
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

    def update_context_for_classifier_train(self, context_info):
        self.context_info_for_classifier_train = context_info

    def update_corrector(self, context_info, conf_info, task_info):

        self.cur_task_id = task_info['task_id']

        self.performance_predictor.update_corrector(context_info=context_info,
                                                    conf_info=conf_info,
                                                    task_info=task_info)

    def get_delay_decrease_and_acc_increase_weight(self, cur_policy, cur_context, chosen_knob_list):

        delay_decrease_weight = {}
        acc_increase_weight = {}
        cur_delay, cur_acc = self.pre_delay_and_acc(context_info=cur_context,
                                                    conf_info=cur_policy)

        for knob in list(cur_policy.keys()):

            if knob not in chosen_knob_list:

                delay_decrease_weight[knob] = 0
                acc_increase_weight[knob] = 0

            else:

                delay_if_add, acc_if_add, new_policy = self.get_delay_and_acc_if_knob_change_dir_units(
                    policy=cur_policy,
                    context=cur_context,
                    knob=knob,
                    dir=1)
                delay_if_dec, acc_if_dec, new_policy = self.get_delay_and_acc_if_knob_change_dir_units(
                    policy=cur_policy,
                    context=cur_context,
                    knob=knob,
                    dir=-1)
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

        norm_delay_decrease_weight = self.normalize_vector(dictionary=delay_decrease_weight)
        norm_acc_increase_weight = self.normalize_vector(dictionary=acc_increase_weight)

        return norm_delay_decrease_weight, norm_acc_increase_weight

    def normalize_vector(self, dictionary):
        values = list(dictionary.values())
        vector_length = math.sqrt(sum(x ** 2 for x in values))
        if vector_length == 0:
            return dictionary
        normalized_values = [x / vector_length for x in values]
        return {key: value for key, value in zip(dictionary.keys(), normalized_values)}

    def pre_delay_and_acc(self, context_info, conf_info, record_path=None):

        delay = self.performance_predictor.delay_pre(context_info=context_info,
                                                     conf_info=conf_info,
                                                     latest_task_id=self.cur_task_id + 1,
                                                     if_correct=True,
                                                     record_path=record_path)
        acc = self.performance_predictor.acc_pre(context_info=context_info,
                                                 conf_info=conf_info,
                                                 if_correct=True)

        delay *= (conf_info['fps'] / self.raw_meta_data['fps'])

        return delay, acc

    def get_delay_and_acc_if_knob_change_dir_units(self, policy, context, knob, dir):
        cur_knob_value = policy[knob]
        cur_knob_idx = self.knob_value_range_dict[knob].index(cur_knob_value)
        new_knob_idx = min(len(self.knob_value_range_dict[knob]) - 1, max(0, cur_knob_idx + dir))
        new_policy = copy.deepcopy(policy)
        new_policy[knob] = self.knob_value_range_dict[knob][new_knob_idx]
        new_delay, new_acc = self.pre_delay_and_acc(context_info=context, conf_info=new_policy)

        return new_delay, new_acc, new_policy

    def cal_delay_loss(self, delay):
        delay_loss = 0
        if delay > self.delay_cons:
            delay_loss = ((delay - self.delay_cons) / self.delay_cons) * self.delay_weight
        return delay_loss

    def cal_acc_loss(self, acc):
        acc_loss = 0
        if acc < self.acc_cons:
            acc_loss = ((self.acc_cons - acc) / self.acc_cons) * self.acc_weight
        return acc_loss

    def cal_loss(self, delay, acc):

        delay_loss = self.cal_delay_loss(delay=delay)
        acc_loss = self.cal_acc_loss(acc=acc)

        loss = delay_loss + acc_loss

        return loss

    def greedy_search(self, policy, cur_context, loss, all_knob_list):

        path_policy = copy.deepcopy(policy)
        path_loss = loss

        path_record = []
        path_record.append(
            {
                'policy': path_policy,
                'loss': path_loss
            }
        )

        left_knob_list = copy.deepcopy(all_knob_list)

        for i in range(0, len(left_knob_list)):

            best_knob, best_dir, best_loss = self.choose_knob_by_loss(cur_policy=path_policy,
                                                                      cur_context=cur_context,
                                                                      cur_loss=path_loss,
                                                                      knob_list=left_knob_list)

            if best_dir == 0:
                break

            elif best_dir in [1, -1]:

                tmp_policy, tmp_loss = self.get_best_policy_loss_in_one_dir(cur_policy=path_policy,
                                                                            cur_loss=path_loss,
                                                                            cur_context=cur_context,
                                                                            knob=best_knob,
                                                                            dir=best_dir)
                path_record.append(
                    {
                        'policy': tmp_policy,
                        'loss': tmp_loss
                    }
                )

                if path_loss < 0.05 and tmp_loss < 0.05:
                    if 0 <= (path_loss - tmp_loss) <= self.stop_threshold * path_loss:
                        break

                path_policy = tmp_policy
                path_loss = tmp_loss
                left_knob_list.remove(best_knob)

            else:
                assert ('wrong dir in choose_knobs_by_search')
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
        pred_res = self.use_classifier(cur_policy=cur_policy,
                                       cur_context=cur_context)
        if pred_res == None:
            return None
        sorted_keys = sorted(pred_res, key=lambda k: pred_res[k], reverse=True)
        LOGGER.debug(f'[SteadyKnowledgeBase] Sorted knobs with classifier: {sorted_keys}')

        sorted_knob_list = []

        if real_time_acc > self.acc_cons == 0 and real_time_delay > self.delay_cons:
            for key in sorted_keys:
                if key in ['buffer_size', 'edge_serv_num']:
                    sorted_knob_list.append(key)
            for key in sorted_keys:
                if key not in ['buffer_size', 'edge_serv_num']:
                    sorted_knob_list.append(key)
        else:
            sorted_knob_list = sorted_keys

        return sorted_knob_list

    #
    def sorted_search(self, policy, cur_context, loss, sorted_knob_list):

        path_policy = copy.deepcopy(policy)
        path_loss = loss

        path_record = []
        path_record.append(
            {
                'policy': path_policy,
                'loss': path_loss
            }
        )
        for knob in sorted_knob_list:

            best_dir = self.get_best_dir_of_knob(cur_policy=path_policy,
                                                 cur_context=cur_context,
                                                 cur_loss=path_loss,
                                                 knob=knob)

            if best_dir == 0:
                break

            elif best_dir in [1, -1]:
                tmp_policy, tmp_loss = self.get_best_policy_loss_in_one_dir(cur_policy=path_policy,
                                                                            cur_loss=path_loss,
                                                                            cur_context=cur_context,
                                                                            knob=knob,
                                                                            dir=best_dir)
                path_record.append(
                    {
                        'policy': tmp_policy,
                        'loss': tmp_loss
                    }
                )
                if path_loss < 0.05 and tmp_loss < 0.05:
                    if 0 <= (path_loss - tmp_loss) <= self.stop_threshold * path_loss:
                        break

                path_policy = tmp_policy
                path_loss = tmp_loss
            else:
                assert ('wrong dir in choose_knobs_by_search')

        return path_record

    def choose_knob_by_loss(self, cur_policy, cur_context, cur_loss, knob_list):
        best_knob = ''
        best_dir = 0
        best_loss = cur_loss
        for knob in knob_list:
            for dir in [-1, 1]:
                new_delay, new_acc, new_policy = self.get_delay_and_acc_if_knob_change_dir_units(policy=cur_policy,
                                                                                                 context=cur_context,
                                                                                                 knob=knob,
                                                                                                 dir=dir
                                                                                                 )
                new_loss = self.cal_loss(delay=new_delay,
                                         acc=new_acc)
                if new_loss < best_loss:
                    best_loss = new_loss
                    best_knob = knob
                    best_dir = dir

        return best_knob, best_dir, best_loss

    def get_best_dir_of_knob(self, cur_policy, cur_context, cur_loss, knob):
        best_dir = 0
        best_loss = cur_loss
        for dir in [-1, 1]:
            new_delay, new_acc, new_policy = self.get_delay_and_acc_if_knob_change_dir_units(policy=cur_policy,
                                                                                             context=cur_context,
                                                                                             knob=knob,
                                                                                             dir=dir
                                                                                             )
            new_loss = self.cal_loss(delay=new_delay,
                                     acc=new_acc)
            if new_loss < best_loss:
                best_loss = new_loss
                best_dir = dir

        return best_dir

    def get_best_policy_loss_in_one_dir(self, cur_policy, cur_loss, cur_context, knob, dir):
        if dir not in [1, -1]:
            assert ('Wrong dir in change_knob_in_one_dir')
        best_policy = copy.deepcopy(cur_policy)
        best_loss = cur_loss
        dir_num = 0
        while True:
            dir_num += 1
            new_delay, new_acc, new_policy = self.get_delay_and_acc_if_knob_change_dir_units(policy=best_policy,
                                                                                             context=cur_context,
                                                                                             knob=knob,
                                                                                             dir=dir_num * dir)
            new_loss = self.cal_loss(delay=new_delay, acc=new_acc)
            if new_loss < best_loss:
                best_loss = new_loss
                best_policy = new_policy
            else:
                break
        return best_policy, best_loss

    def train_new_classifier(self):
        while True:

            if self.context_info_for_classifier_train == None:
                time.sleep(1.0)
                continue

            else:
                new_context = copy.deepcopy(self.context_info_for_classifier_train)
                cluster_name, extreme_context, if_belong_cluster = self.process_context_for_cluster(
                    cur_context=new_context)
                if cluster_name is not None and extreme_context is not None:
                    if cluster_name in self.trainer.list_models():
                        time.sleep(1.0)
                        continue

                    LOGGER.debug(f'[SteadyKnowledgeBase] Training classifier for cluster {cluster_name}.')

                    x_data_list, y_data_list = self.get_train_data(extreme_context=extreme_context)
                    LOGGER.debug(f'[SteadyKnowledgeBase] Generated training data for cluster {cluster_name}.')

                    self.trainer.train(X=x_data_list,
                                       y=y_data_list,
                                       name=cluster_name)
                    LOGGER.debug(f'[SteadyKnowledgeBase] Finished classifier training for cluster {cluster_name}.')
            time.sleep(1.0)

    def use_classifier(self, cur_policy, cur_context):

        LOGGER.debug('[SteadyKnowledgeBase] Trying classifier-based knob sorting.')

        if self.cluster_threshold == 0:
            LOGGER.debug('[SteadyKnowledgeBase] Cluster threshold is zero; classifier is disabled.')
            return None

        cluster_name, extreme_context, if_belong_cluster = self.process_context_for_cluster(cur_context=cur_context)
        LOGGER.debug(
            f'[SteadyKnowledgeBase] Classifier candidate: cluster={cluster_name}, '
            f'extreme_context={extreme_context}, belongs={if_belong_cluster}'
        )
        LOGGER.debug(f'[SteadyKnowledgeBase] Available classifiers: {self.trainer.list_models()}')

        if cluster_name is not None:
            if if_belong_cluster == 1:

                x_data = self.trans_policy_to_x_data(policy=cur_policy)
                pred_proba = self.trainer.get_pred_proba(x=x_data,
                                                         name=cluster_name)
                if pred_proba == None:
                    return None

                pred_res = {}
                tmp_idx = 0
                for knob_name in self.optional_knob_name_list:
                    pred_res[knob_name] = pred_proba[tmp_idx]
                    tmp_idx += 1
                return pred_res

        return None

    def process_context_for_cluster(self, cur_context):

        if_belong_cluster = 1
        extreme_context = {}
        cluster_name = ''

        if self.cluster_threshold == 0:
            return None, None, None

        elif ('band_Mbps' in cur_context) and ('obj_size_norm' in cur_context) and \
            ('obj_num' in cur_context) and ('obj_speed' in cur_context):

            band_Mbps = cur_context['band_Mbps']
            obj_size_norm = cur_context['obj_size_norm']
            obj_num = cur_context['obj_num']
            obj_speed = cur_context['obj_speed']

            if band_Mbps < 0.1:
                cluster_name += '0'
                extreme_context['band_Mbps'] = 0
            elif band_Mbps < 1:
                cluster_name += '1'
                extreme_context['band_Mbps'] = 0.1
                if band_Mbps > 0.1 + (1 - 0.1) * self.cluster_threshold:
                    if_belong_cluster = 0
            elif band_Mbps < 5:
                cluster_name += '2'
                extreme_context['band_Mbps'] = 1
                if band_Mbps > 1 + (5 - 1) * self.cluster_threshold:
                    if_belong_cluster = 0
            elif band_Mbps < 10:
                cluster_name += '3'
                extreme_context['band_Mbps'] = 5
                if band_Mbps > 5 + (10 - 5) * self.cluster_threshold:
                    if_belong_cluster = 0
            else:
                cluster_name += '4'
                extreme_context['band_Mbps'] = 10
                if band_Mbps > 10 + 10 * self.cluster_threshold:
                    if_belong_cluster = 0

            if obj_size_norm < 0.05:
                cluster_name += '0'
                extreme_context['obj_size_norm'] = 0.01
            elif obj_size_norm < 0.1:
                cluster_name += '1'
                extreme_context['obj_size_norm'] = 0.05
                if obj_size_norm > 0.05 + (0.1 - 0.05) * self.cluster_threshold:
                    if_belong_cluster = 0
            elif obj_size_norm < 0.2:
                cluster_name += '2'
                extreme_context['obj_size_norm'] = 0.1
                if obj_size_norm > 0.1 + (0.2 - 0.1) * self.cluster_threshold:
                    if_belong_cluster = 0
            elif obj_size_norm < 0.3:
                cluster_name += '3'
                extreme_context['obj_size_norm'] = 0.2
                if obj_size_norm > 0.2 + (0.3 - 0.2) * self.cluster_threshold:
                    if_belong_cluster = 0
            else:
                cluster_name += '4'
                extreme_context['obj_size_norm'] = 0.3
                if obj_size_norm > 0.3 + 0.3 * self.cluster_threshold:
                    if_belong_cluster = 0

            if obj_num < 1:
                cluster_name += '0'
                extreme_context['obj_num'] = 1
            elif obj_num < 5:
                cluster_name += '1'
                extreme_context['obj_num'] = 5
                if obj_num < 5 - (5 - 1) * self.cluster_threshold:
                    if_belong_cluster = 0
            elif obj_num < 10:
                cluster_name += '2'
                extreme_context['obj_num'] = 10
                if obj_num < 10 - (10 - 5) * self.cluster_threshold:
                    if_belong_cluster = 0
            else:
                cluster_name += '3'
                extreme_context['obj_num'] = 20
                if obj_num < 20 - (20 - 10) * self.cluster_threshold or obj_num > 20:
                    if_belong_cluster = 0

            if obj_speed < 260:
                cluster_name += '0'
                extreme_context['obj_speed'] = 260
            elif obj_speed < 520:
                cluster_name += '1'
                extreme_context['obj_speed'] = 520
                if obj_speed < 520 - (520 - 260) * self.cluster_threshold:
                    if_belong_cluster = 0
            elif obj_speed < 780:
                cluster_name += '2'
                extreme_context['obj_speed'] = 780
                if obj_speed < 780 - (780 - 520) * self.cluster_threshold:
                    if_belong_cluster = 0
            else:
                cluster_name += '3'
                extreme_context['obj_speed'] = 1500
                if obj_speed < 1500 - (1500 - 780) * self.cluster_threshold or obj_speed > 1500:
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

                        x_data = self.trans_policy_to_x_data(policy=tmp_policy)
                        chosen_knob_set = self.get_chosen_knob_set_for_train(start_policy=tmp_policy,
                                                                             cur_context=extreme_context)
                        y_data = self.trans_knob_set_to_y_data(knob_set=chosen_knob_set)

                        x_data_list.append(x_data)
                        y_data_list.append(y_data)

        return x_data_list, y_data_list

    def trans_policy_to_x_data(self, policy):

        x_data = []
        # ['fps','resolution', 'buffer_size', 'edge_serv_num']
        for knob_name in self.optional_knob_name_list:
            tmp_knob_value = policy[knob_name]
            tmp_knob_idx = self.knob_value_range_dict[knob_name].index(tmp_knob_value)
            x_data.append(tmp_knob_idx)

        return x_data

    def trans_knob_set_to_y_data(self, knob_set):

        y_data = []
        for knob_name in self.optional_knob_name_list:
            if knob_name in knob_set:
                y_data.append(1)
            else:
                y_data.append(0)
        return y_data

    def get_chosen_knob_set_for_train(self, start_policy, cur_context):

        delay, acc = self.pre_delay_and_acc(context_info=cur_context,
                                            conf_info=start_policy)
        start_loss = self.cal_loss(delay=delay, acc=acc)
        all_knob_list = list(self.knob_value_range_dict.keys())
        path_record = self.greedy_search(policy=start_policy,
                                         cur_context=cur_context,
                                         loss=start_loss,
                                         all_knob_list=all_knob_list)
        conv_policy = copy.deepcopy(path_record[-1]['policy'])

        chosen_knob_list = []
        for knob in list(start_policy.keys()):
            if conv_policy[knob] is not start_policy[knob]:
                chosen_knob_list.append(knob)

        return chosen_knob_list
