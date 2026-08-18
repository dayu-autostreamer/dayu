from .knowledge_base import KnowledgeBase
import asyncio
import threading
import copy
import time
import math


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
                 if_online_train
                 ):
        

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
                 cluster_threshold = cluster_threshold,
                 if_online_train = if_online_train
                 )
        

        self.knob_value_range_dict = copy.deepcopy(knob_value_range_dict)


        self.conv_policy = copy.deepcopy(default_policy)

        self.context_history = {}
        self.history_lenghth = history_lenghth
        for context in context_names:
            self.context_history[context] = []

        self.context_anylze_type = context_anylze_type 
    

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


    def get_anylzed_context(self):

        anylzed_context = {}

        if self.context_anylze_type == 1:

             for context in self.context_history.keys():
                
                if len(self.context_history[context]) == 0:
                    anylzed_context = {}
                    break

                else:
                    anylzed_context[context] = sum(self.context_history[context])/len(self.context_history[context])

        else: 
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

    def update_conv_policy(self, start_policy, real_time_delay, real_time_acc):

        cur_context = self.get_anylzed_context()

        path_record = []

        if cur_context is not None:
            if len(cur_context)>0:

                delay,acc = self.pre_delay_and_acc(context_info = cur_context,
                                                   conf_info = start_policy)
                

                start_loss = self.cal_loss(delay=delay,acc=acc)

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
                        if_greedy_for_path_record = 0 

                if if_greedy_for_path_record == 1:
                    all_knob_list = list(self.knob_value_range_dict.keys())
                    path_record = self.knowledge_base.greedy_search(policy = start_policy,
                                                    cur_context = cur_context,
                                                    loss = start_loss,
                                                    all_knob_list = all_knob_list)
                
                if len(path_record) > 0:
                    self.conv_policy = copy.deepcopy(path_record[-1]['policy'])
                    conv_loss = copy.deepcopy(path_record[-1]['loss'])

        
        return path_record 


    def choose_knobs_by_conv_policy(self,cur_policy):
        chosen_knob_list = []
        for knob in list(cur_policy.keys()):
            if self.conv_policy[knob] is not cur_policy[knob]:
                chosen_knob_list.append(knob)

        return chosen_knob_list


    def update_corrector(self, context_info, conf_info, task_info):

        self.knowledge_base.update_corrector(context_info = context_info,
                                             conf_info = conf_info,
                                             task_info = task_info)



    def get_delay_decrease_and_acc_increase_weight(self, cur_policy, cur_context, chosen_knob_list):
        norm_delay_decrease_weight, norm_acc_increase_weight = self.knowledge_base.get_delay_decrease_and_acc_increase_weight(cur_policy=cur_policy,
                                                                                                                              cur_context=cur_context,
                                                                                                                              chosen_knob_list=chosen_knob_list)
        return norm_delay_decrease_weight, norm_acc_increase_weight


    def pre_delay_and_acc(self, context_info, conf_info, record_path = None):
        delay,acc = self.knowledge_base.pre_delay_and_acc(context_info = context_info,
                                                          conf_info = conf_info,
                                                          record_path = record_path)
        return delay, acc
            

    def cal_loss(self, delay ,acc):
        loss = self.knowledge_base.cal_loss(delay=delay,
                                            acc=acc)
        return loss
           

class MicroFeedback:
    
    def __init__(self,coeff_info, knob_value_range_dict):


        self.coeff_info = copy.deepcopy(coeff_info)
        self.coeff_info['step_coeff']['cur_value'] = self.coeff_info['step_coeff']['start_value']
        
        self.delay_loss = 0
        self.acc_loss = 0
       
        self.all_knob_weight = {}

        self.knob_value_range_dict = copy.deepcopy(knob_value_range_dict)



    def update_delay_loss_and_acc_loss(self,delay_loss, acc_loss):
        self.delay_loss = delay_loss
        self.acc_loss = acc_loss

    def update_step_coeff(self, if_meet_cons_before, if_meet_cons_now):


        if (not if_meet_cons_before) and (not if_meet_cons_now):
            self.update_coeff(coeff_name='step_coeff',if_add = True)

        elif (not if_meet_cons_before) and if_meet_cons_now:
            self.update_coeff(coeff_name='step_coeff',if_add = False)

        elif if_meet_cons_before and (not if_meet_cons_now):
            self.update_coeff(coeff_name='step_coeff',if_add = True)

        elif if_meet_cons_before and if_meet_cons_now:
           pass
        

    def update_all_knob_weight(self, delay_decrease_weight, acc_increase_weight):

        if self.delay_loss + self.acc_loss == 0:
            delay_perf = 0
            acc_perf = 0
        else:
            delay_perf = self.delay_loss / (self.delay_loss + self.acc_loss)
            acc_perf = self.acc_loss / (self.delay_loss + self.acc_loss)

        for knob in delay_decrease_weight.keys():
            self.all_knob_weight[knob] = acc_perf * acc_increase_weight[knob]  + delay_perf * delay_decrease_weight[knob]

    def update_coeff(self, coeff_name, if_add):

        if if_add:
            if self.coeff_info[coeff_name]['cur_value'] + self.coeff_info[coeff_name]['add_interval'] <= self.coeff_info[coeff_name]['max_value']:
                
                self.coeff_info[coeff_name]['cur_value'] += self.coeff_info[coeff_name]['add_interval']
            else:
                self.coeff_info[coeff_name]['cur_value'] = self.coeff_info[coeff_name]['max_value']
        else:
            if (self.coeff_info[coeff_name]['cur_value'] / 2.0 ) >= self.coeff_info[coeff_name]['min_value']:
                
                self.coeff_info[coeff_name]['cur_value'] /= 2.0 
            else:
                self.coeff_info[coeff_name]['cur_value'] = self.coeff_info[coeff_name]['min_value']
    

    def get_new_policy_by_feedback(self, old_policy):

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
                 cluster_threshold,
                 schedule_type,
                 feedback_weight,
                 if_online_train
                 ):
          
        self.knob_value_range_dict = copy.copy(knob_value_range_dict)
        self.default_policy = default_policy
        self.delay_cons = delay_cons
        self.acc_cons = acc_cons
        self.delay_weight = delay_weight
        self.acc_weight = acc_weight
        self.stop_threshold = stop_threshold

        self.latest_real_time_loss = None


        self.cur_policy = None
        self.real_time_context_info = None
        self.real_time_delay = None
        self.real_time_acc = None

        self.if_need_macro_search = None
        self.path_record = None
        self.context_history = None
        self.macro_search_delay = None
        self.chosen_knob_list = None
        self.delay_decrease_weight = None 
        self.acc_increase_weight = None
        
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
                                        cluster_threshold = cluster_threshold,
                                        if_online_train = if_online_train)

        self.micro_feedback = MicroFeedback(coeff_info = coeff_info,
                                            knob_value_range_dict = knob_value_range_dict)
        
        self.macro_update_interval = macro_update_interval
        

        self._loop = asyncio.new_event_loop() 
        self._stop_event = threading.Event() 
        self._thread = threading.Thread(target=self._run_loop, daemon=True)  

        self.schedule_type = schedule_type
        self.feedback_weight = feedback_weight

        if schedule_type == 'macro':   
            self._thread.start() 

    def update_delay_cons(self, delay_cons):
        self.delay_cons = delay_cons
        self.macro_search.knowledge_base.update_delay_cons(delay_cons)

    def update_acc_cons(self, acc_cons):
        self.acc_cons = acc_cons
        self.macro_search.knowledge_base.update_acc_cons(acc_cons)
    
    def update_delay_weight(self, delay_weight):
        self.delay_weight = delay_weight
        self.macro_search.knowledge_base.update_delay_weight(delay_weight)

    def update_acc_weight(self, acc_weight):
        self.acc_weight = acc_weight
        self.macro_search.knowledge_base.update_acc_weight(acc_weight)


    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._compute_loop())
        finally:
            self._loop.close()
    
    _lock = threading.Lock()

    def update_parameter(self, cur_policy, real_time_context_info, real_time_delay, real_time_acc):
        with self._lock:
            self.cur_policy = copy.deepcopy(cur_policy)
            self.real_time_context_info = copy.deepcopy(real_time_context_info)
            self.real_time_delay = real_time_delay
            self.real_time_acc = real_time_acc
    
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
            
            

    def stop(self):
        self._stop_event.set()
        self._thread.join()
    

    async def _compute_loop(self):

        async def compute_f(cur_policy, real_time_context_info):

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

                    if_need_macro_search = True
                else:

                    pass


            path_record = []
            if if_need_macro_search:
                path_record = copy.deepcopy(self.macro_search.update_conv_policy(start_policy = self.macro_search.conv_policy,
                                                                                 real_time_delay = self.real_time_delay,
                                                                                 real_time_acc = self.real_time_acc
                                                                                 ))
            

            chosen_knob_list = copy.deepcopy(self.macro_search.choose_knobs_by_conv_policy(cur_policy = cur_policy))

            delay_decrease_weight, acc_increase_weight = self.macro_search.get_delay_decrease_and_acc_increase_weight(cur_policy = cur_policy,
                                                                                                                        cur_context = real_time_context_info,
                                                                                                                        chosen_knob_list = chosen_knob_list,
                                                                                                                        )
            context_history = copy.deepcopy(self.macro_search.context_history)

            
            
            return if_need_macro_search, path_record, context_history, chosen_knob_list, delay_decrease_weight, acc_increase_weight
        

        while not self._stop_event.is_set():

            start_time = time.time()

            await asyncio.sleep(self.macro_update_interval)

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

    def update_scheduler(self,context_info, conf_info, task_info):

        if context_info is not None and conf_info is not None and task_info is not None:

            self.macro_search.update_context(context_info=context_info)
            self.macro_search.update_corrector(context_info=context_info,
                                               conf_info=conf_info,
                                                task_info=task_info)

    def get_schedule_plan(self, cur_task_id, cur_policy, context_info, real_time_delay, real_time_acc):


        self.update_parameter(cur_policy = cur_policy,
                              real_time_context_info = context_info,
                              real_time_delay = real_time_delay,
                              real_time_acc = real_time_acc
                              )

        if cur_policy is None or context_info is None or real_time_delay is None or real_time_acc is None:

            new_policy = copy.deepcopy(self.default_policy)

            return new_policy
        
        elif self.schedule_type == 'macro':
            macro_output = self.get_macro_output()
            path_record = macro_output['path_record']

            new_policy = copy.deepcopy(self.cur_policy)
            if path_record is not None:
                if len(path_record) > 0:
                    new_policy = copy.deepcopy(path_record[-1]['policy'])
            return new_policy

        elif self.schedule_type == 'micro':

            if_meet_cons_before = False
            if_meet_cons_now = False
            if self.latest_real_time_loss is not None:
                if self.latest_real_time_loss == 0:
                    if_meet_cons_before = True
            
            real_time_loss = self.macro_search.knowledge_base.cal_loss(delay = real_time_delay,
                                                                       acc = real_time_acc)
            
            if real_time_loss == 0: 
                if_meet_cons_now = True
            self.latest_real_time_loss = real_time_loss

            delay_decrease_weight = self.feedback_weight['delay_decrease_weight']
            acc_increase_weight = self.feedback_weight['acc_increase_weight']

            
            self.micro_feedback.update_delay_loss_and_acc_loss(delay_loss = self.macro_search.knowledge_base.cal_delay_loss(delay=real_time_delay),
                                                               acc_loss = self.macro_search.knowledge_base.cal_acc_loss(acc=real_time_acc))
            self.micro_feedback.update_step_coeff(if_meet_cons_before = if_meet_cons_before,
                                                    if_meet_cons_now = if_meet_cons_now)
            self.micro_feedback.update_all_knob_weight(delay_decrease_weight = delay_decrease_weight,
                                                    acc_increase_weight = acc_increase_weight)
            
            new_policy = self.micro_feedback.get_new_policy_by_feedback(old_policy = cur_policy)



            return new_policy


