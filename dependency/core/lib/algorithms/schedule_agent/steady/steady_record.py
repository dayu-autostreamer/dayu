import json
import os


class SteadyRecord:


    def __init__(self,
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
                 ):
        
        self.task_id = task_id
        self.if_need_macro_search = if_need_macro_search
        self.path_record = path_record
        self.context_history = context_history
        self.macro_search_delay = macro_search_delay
        self.chosen_knob_list = chosen_knob_list
        self.delay_decrease_weight = delay_decrease_weight
        self.acc_increase_weight = acc_increase_weight
        self.delay_loss = delay_loss
        self.acc_loss = acc_loss
        self.coeff_info = coeff_info
        self.if_new_policy_is_bad = if_new_policy_is_bad
        self.old_policy = old_policy
        self.new_policy = new_policy    

    def get_value(self):

        return {
            'task_id':self.task_id,
            'if_need_macro_search':self.if_need_macro_search,
            'path_record':self.path_record,
            'context_history':self.context_history,
            'macro_search_delay':self.macro_search_delay,
            'chosen_knob_list':self.chosen_knob_list,
            'delay_decrease_weight':self.delay_decrease_weight,
            'acc_increase_weight':self.acc_increase_weight,
            'delay_loss':self.delay_loss,
            'acc_loss':self.acc_loss,
            'coeff_info':self.coeff_info,
            'if_new_policy_is_bad':self.if_new_policy_is_bad,
            'old_policy':self.old_policy,
            'new_policy':self.new_policy
        }

    @staticmethod
    def serialize(steady_record:'SteadyRecord'):

        return json.dumps({
            'task_id':steady_record.task_id,
            'if_need_macro_search':steady_record.if_need_macro_search,
            'path_record':steady_record.path_record,
            'context_history':steady_record.context_history,
            'macro_search_delay':steady_record.macro_search_delay,
            'chosen_knob_list':steady_record.chosen_knob_list,
            'delay_decrease_weight':steady_record.delay_decrease_weight,
            'acc_increase_weight':steady_record.acc_increase_weight,
            'delay_loss':steady_record.delay_loss,
            'acc_loss':steady_record.acc_loss,
            'coeff_info':steady_record.coeff_info,
            'if_new_policy_is_bad':steady_record.if_new_policy_is_bad,
            'old_policy':steady_record.old_policy,
            'new_policy':steady_record.new_policy
        })
    
    @staticmethod
    def deserialize(data: str):

        data = json.loads(data)
        steady_record = SteadyRecord(
            task_id = data['task_id'],
            if_need_macro_search = data['if_need_macro_search'],
            path_record = data['path_record'],
            context_history = data['context_history'],
            macro_search_delay = data['macro_search_delay'],
            chosen_knob_list = data['chosen_knob_list'],
            delay_decrease_weight = data['delay_decrease_weight'],
            acc_increase_weight = data['acc_increase_weight'],
            delay_loss = data['delay_loss'],
            acc_loss = data['acc_loss'],
            coeff_info = data['coeff_info'],
            if_new_policy_is_bad = data['if_new_policy_is_bad'],
            old_policy = data['old_policy'],
            new_policy = data['new_policy']
            )
        
        return steady_record
    
    @staticmethod
    def write_record(steady_record: 'SteadyRecord', file_path):
        if not os.path.exists(file_path):
            with open(file_path,'w') as f:
                f.write('')
        
        json_str = SteadyRecord.serialize(steady_record) + '\n'
        #print('待写入:',steady_record.old_policy)
        #print('待写入:',steady_record.new_policy)
        with open(file_path,'a') as f:
            f.write(json_str)

    @staticmethod
    def read_record(file_path):
        
        record_list = []
        with open(file_path,'r') as f:
            for line in f:
                stripped_line = line.strip() 
                if stripped_line: 
                    try:  
                        steady_record = SteadyRecord.deserialize(stripped_line)
                        record_list.append(steady_record)  
                    except json.JSONDecodeError:  
                        print(f"Warning: Could not decode JSON from line: {stripped_line}")  
       
        return record_list