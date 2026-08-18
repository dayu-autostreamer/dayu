import numpy as np
import math
import json
import copy
from .context_cluster import ContextCluster
from .accuracy_prediction import AccuracyPrediction2fps,AccuracyPrediction2reso, resolution_wh


class PolynomialFitter:

    def __init__(self):
        pass

    def fit(self, x_list:list, y_list:list, n:int):
 
        x_np = np.array(x_list)
        y_np = np.array(y_list)

        if n==0:

            y_mean = np.mean(y_np)
            y_std = np.std(y_np)
            
            adjusted_mean = max(y_mean, 1e-10)    
            score = max(min(1 - y_std / adjusted_mean, 1), 0)

            return [y_mean],score

        if len(x_list) > 1:

            coeff_np = np.polyfit(x_np, y_np, n)
            coeff_list = list(coeff_np)

            pre_y_np = np.polyval(coeff_np, x_np)

            ss_res = np.sum((y_np - pre_y_np) ** 2)  
            ss_tot = np.sum((y_np - np.mean(y_np)) ** 2)  

            r2 = 1

            if ss_tot == 0:
                r2 = 1
            else:
                r2 = 1 - (ss_res / ss_tot)

            score = r2

            return coeff_list[::-1], score
        

        elif len(x_list)==1 and x_list[0]!=0: 

            coeff_list = [0]
            coeff_list.append(y_list[0]/x_list[0])

            score = 1
            return coeff_list, score
        

        elif len(x_list)==1 and x_list[0]==0: 

            coeff_list = [y_list[0]]
            score = 1
            return coeff_list, score

    def predict(self,coeff_list:list, x):

        coeff_np = np.array(coeff_list[::-1])

        x_np = np.array([x])
        pre_y_np = np.polyval(coeff_np, x_np)
        y = pre_y_np[0]
        
        return y


class SimpleCorrector:

    def __init__(self,n,coeff_window_length,x_sample_window_length, y_sample_window_length, context_names):
        
        self.n = n 
        self.coeff_window_length = coeff_window_length 
        self.x_sample_window_length = x_sample_window_length 
        self.y_sample_window_length = y_sample_window_length 
        self.polynomial_fitter = PolynomialFitter()
        

        self.coeff_window = []
        for i in range(self.n+1):
            self.coeff_window.append([])
        

        self.x_sample_window = []

        self.x_y_sample_window = {}

        self.context_sample_window = {}

        self.context_names = context_names
        for context_name in self.context_names:
            self.context_sample_window[context_name] = {}


    def update_sample_window(self,x_value,y_value, context_info):

        if x_value not in self.x_sample_window:
            self.x_sample_window.append(x_value)
            if len(self.x_sample_window) > self.x_sample_window_length:
                del_x_value = self.x_sample_window.pop(0)
                if del_x_value in self.x_y_sample_window:
                    del self.x_y_sample_window[del_x_value]
                    for context_name in self.context_sample_window:
                        del self.context_sample_window[context_name][del_x_value]
        else:
            pass


        if x_value not in self.x_y_sample_window:
            self.x_y_sample_window[x_value] = [y_value]
            for context_name in self.context_names:
                self.context_sample_window[context_name][x_value] = [context_info[context_name]]

        else:
            self.x_y_sample_window[x_value].append(y_value)
            for context_name in self.context_names:
                self.context_sample_window[context_name][x_value].append(context_info[context_name])
        
            if len(self.x_y_sample_window[x_value]) > self.y_sample_window_length:
                self.x_y_sample_window[x_value].pop(0)
                for context_name in self.context_names:
                    self.context_sample_window[context_name][x_value].pop(0)


    def update_coeff_window(self):
        x_list = []
        y_list = []
        for x in self.x_y_sample_window.keys():
            if len(self.x_y_sample_window[x]) > 0:
                x_list.append(x)
                y_list.append(sum(self.x_y_sample_window[x])/len(self.x_y_sample_window[x]))
                
        if len(x_list) == 0:
            return
        coeff_list = []  

        if len(x_list) == 1:
            coeff_list, score = self.polynomial_fitter.fit(x_list=x_list,
                                                        y_list=y_list,
                                                        n=1)

        elif len(x_list) > 1 and len(x_list) < self.n + 1:
            coeff_list, score = self.polynomial_fitter.fit(x_list=x_list,
                                                        y_list=y_list,
                                                        n=len(x_list)-1)

        elif len(x_list) >= self.n + 1:
            coeff_list, score = self.polynomial_fitter.fit(x_list=x_list,
                                                        y_list=y_list,
                                                        n=self.n)

        for i in range(len(coeff_list),self.n + 1):
            coeff_list.append(0.0)

        for i in range(self.n+1):
            self.coeff_window[i].append(coeff_list[i])
            if len(self.coeff_window[i]) > self.coeff_window_length:
                self.coeff_window[i].pop(0)
    

    def get_cur_coeff_window_and_context_val(self):

        cur_coeff_window = copy.deepcopy(self.coeff_window)
        cur_context_val = {}
        for context_name in self.context_names:

            sample_context_list = [ sample_context for c_lst in self.context_sample_window[context_name].values() for sample_context in c_lst]
            if len(sample_context_list) > 0:
                cur_context_val[context_name] = sum(sample_context_list) / len(sample_context_list)
            else:
                cur_context_val[context_name] = None

        return cur_coeff_window, cur_context_val


    def predict(self,x,cur_coeff_window):

        coeff_window = copy.deepcopy(cur_coeff_window)

        if len(coeff_window[0]) == 0:
            return x

        coeff_list = []
        for i in range(self.n+1):
            coeff_list.append(coeff_window[i][-1])
        
        y = self.polynomial_fitter.predict(coeff_list=coeff_list,
                                           x=x)

        if y<0:
            y = x

        return y


class CorrectedPredictor():

    def __init__(self,kb_path, service_name_pipeline, corrector_param, cluster_threshold):

        self.kb_path = kb_path

        self.service_name_pipeline = []
        for service_name in service_name_pipeline: 
            if service_name == 'end':
                break
            else:
                self.service_name_pipeline.append(service_name)

        self.accuracy_prediction_2_fps = AccuracyPrediction2fps()
        self.accuracy_prediction_2_reso = AccuracyPrediction2reso()

        
        self.exe_pre_detect_dict = self.read_dict(kb_path + '/' + service_name_pipeline[0] + '.json')
        self.exe_pre_classify_dict = {}
        if len(service_name_pipeline)>1:
            self.exe_pre_classify_dict = self.read_dict(kb_path + '/' + service_name_pipeline[1] + '.json')
  

        self.file_size_dict = self.read_dict(kb_path + '/' + 'file_size' + '.json')

        self.context_cluster = ContextCluster(cluster_threshold=cluster_threshold)


        self.corrector_pool = {}
        self.corrector_pool_threshold = corrector_param['corrector_pool_threshold']


        param = corrector_param['detect']
        context_names = []
        self.exe_corrector_detect = SimpleCorrector(n = param['n'],
                                                    coeff_window_length = param['coeff_window_length'],
                                                    x_sample_window_length = param['x_sample_window_length'],
                                                    y_sample_window_length = param['y_sample_window_length'],
                                                    context_names = context_names)
        param = corrector_param['classify']
        context_names = ['obj_num']
        self.exe_corrector_classify = SimpleCorrector(n = param['n'],
                                                    coeff_window_length = param['coeff_window_length'],
                                                    x_sample_window_length = param['x_sample_window_length'],
                                                    y_sample_window_length = param['y_sample_window_length'],
                                                    context_names = context_names)
        param = corrector_param['trans']
        context_names = ['band_Mbps']
        self.trans_corrector = SimpleCorrector(n = param['n'],
                                                coeff_window_length = param['coeff_window_length'],
                                                x_sample_window_length = param['x_sample_window_length'],
                                                y_sample_window_length = param['y_sample_window_length'],
                                                context_names = context_names)
        
        param = corrector_param['acc']
        context_names = ['obj_size_norm','obj_speed']
        self.acc_reso_corrector = SimpleCorrector(n = param['n'],
                                                coeff_window_length = param['coeff_window_length'],
                                                x_sample_window_length = param['x_sample_window_length'],
                                                y_sample_window_length = param['y_sample_window_length'],
                                                context_names = context_names)
        
        

        

    def read_dict(self, file_path):
        dict_data={}
        try:  
            with open(file_path, 'r') as file:  
                dict_data = json.load(file)  
        except FileNotFoundError:   
            dict_data = {} 
        return dict_data
    

    def update_corrector(self, context_info, conf_info, task_info):

        cluster_name, _, if_belong_cluster = self.context_cluster.process_context_for_cluster(cur_context=context_info)

        if 'real_exe_detect' in task_info:
            
            if cluster_name is not None:
                if if_belong_cluster == 1:
                    if cluster_name in self.corrector_pool:
                        y_real_value=task_info['real_exe_detect']
                        y_corr_value = self.exe_pre_detect(context_info=context_info,
                                                           conf_info=conf_info,
                                                           if_correct=True,
                                                           cur_coeff_window=self.corrector_pool[cluster_name]['detect_coeff_window'])
                        if y_real_value > 0:
                            if abs( (y_real_value - y_corr_value) / y_real_value ) > self.corrector_pool_threshold:
                                del self.corrector_pool[cluster_name]


            self.exe_corrector_detect.update_sample_window(x_value=self.exe_pre_detect(context_info=context_info, conf_info=conf_info,if_correct=False),
                                                        y_value=task_info['real_exe_detect'],
                                                        context_info = context_info)
            self.exe_corrector_detect.update_coeff_window()

        if 'real_exe_classify' in task_info:

            if cluster_name is not None:
                if if_belong_cluster == 1:
                    if cluster_name in self.corrector_pool:
                        y_real_value=task_info['real_exe_classify']
                        y_corr_value = self.exe_pre_classify(context_info=context_info,
                                                           conf_info=conf_info,
                                                           if_correct=True,
                                                           cur_coeff_window=self.corrector_pool[cluster_name]['classify_coeff_window'])
                        if y_real_value > 0:
                            if abs( (y_real_value - y_corr_value) / y_real_value ) > self.corrector_pool_threshold:
                                del self.corrector_pool[cluster_name]
        
            self.exe_corrector_classify.update_sample_window(x_value=self.exe_pre_classify(context_info=context_info, conf_info=conf_info,if_correct=False),
                                                            y_value=task_info['real_exe_classify'],
                                                            context_info = context_info)
            self.exe_corrector_classify.update_coeff_window()
        
        if 'real_trans' in task_info:
            
            if cluster_name is not None:
                if if_belong_cluster == 1:
                    if cluster_name in self.corrector_pool:
                        y_real_value=task_info['real_trans']
                        y_corr_value = self.trans_pre(context_info=context_info,
                                                           conf_info=conf_info,
                                                           if_correct=True,
                                                           cur_coeff_window=self.corrector_pool[cluster_name]['trans_coeff_window'])
                        if y_real_value > 0:
                            if abs( (y_real_value - y_corr_value) / y_real_value ) > self.corrector_pool_threshold:
                                del self.corrector_pool[cluster_name]
        
            self.trans_corrector.update_sample_window(x_value=self.trans_pre(context_info=context_info, conf_info=conf_info,if_correct=False),
                                                    y_value=task_info['real_trans'],
                                                    context_info = context_info)
            self.trans_corrector.update_coeff_window()
        
        if 'real_acc_reso' in task_info:

            if cluster_name is not None:
                if if_belong_cluster == 1:
                    if cluster_name in self.corrector_pool:
                        y_real_value=task_info['real_acc_reso']
                        y_corr_value = self.acc_pre(context_info=context_info,
                                                           conf_info=conf_info,
                                                           if_correct=True,
                                                           cur_coeff_window=self.corrector_pool[cluster_name]['acc_coeff_window'])
                        if y_real_value > 0:
                            if abs( (y_real_value - y_corr_value) / y_real_value ) > self.corrector_pool_threshold:
                                del self.corrector_pool[cluster_name]

            self.acc_reso_corrector.update_sample_window(x_value=self.acc_pre(context_info=context_info, conf_info=conf_info,if_correct=False),
                                                    y_value=task_info['real_acc_reso'],
                                                    context_info = context_info)
            self.acc_reso_corrector.update_coeff_window()
        

        detect_coeff_window, detect_context_val = self.exe_corrector_detect.get_cur_coeff_window_and_context_val()
        classify_coeff_window, classify_context_val = self.exe_corrector_classify.get_cur_coeff_window_and_context_val()
        trans_coeff_window, trans_context_val = self.trans_corrector.get_cur_coeff_window_and_context_val()
        acc_coeff_window, acc_context_val = self.acc_reso_corrector.get_cur_coeff_window_and_context_val()


        corrector_context_info = {
            'obj_num':classify_context_val['obj_num'],
            'band_Mbps':trans_context_val['band_Mbps'],
            'obj_size_norm':acc_context_val['obj_size_norm'],
            'obj_speed':acc_context_val['obj_speed']
        }

        cluster_name, _, if_belong_cluster = self.context_cluster.process_context_for_cluster(cur_context=corrector_context_info)
        if cluster_name is not None:
            if if_belong_cluster == 1:
                self.corrector_pool[cluster_name] = {
                    'detect_coeff_window':detect_coeff_window,
                    'classify_coeff_window':classify_coeff_window,
                    'trans_coeff_window':trans_coeff_window,
                    'acc_coeff_window':acc_coeff_window
                }

    def get_all_coeff_window_by_context(self, context_info):

        cluster_name, _, if_belong_cluster = self.context_cluster.process_context_for_cluster(cur_context = context_info)
        
        all_coeff_window = {}
        if cluster_name is not None:
            if if_belong_cluster == 1:
                if cluster_name in self.corrector_pool:
                    all_coeff_window = copy.deepcopy(self.corrector_pool[cluster_name])
                    return all_coeff_window
        
        all_coeff_window = {
            'detect_coeff_window':copy.deepcopy(self.exe_corrector_detect.coeff_window),
            'classify_coeff_window':copy.deepcopy(self.exe_corrector_classify.coeff_window),
            'trans_coeff_window':copy.deepcopy(self.trans_corrector.coeff_window),
            'acc_coeff_window':copy.deepcopy(self.acc_reso_corrector.coeff_window)
        }

        return all_coeff_window
        

    def acc_pre(self, context_info, conf_info, if_correct, cur_coeff_window=None):

        acc_fps = self.accuracy_prediction_2_fps.predict(
                            service_name = self.service_name_pipeline[0],
                            service_conf = {
                                'fps':conf_info['fps'],
                                'resolution':conf_info['resolution'],
                            },
                            obj_size = context_info['obj_size_norm'] * resolution_wh[conf_info['resolution']]['w']*resolution_wh[conf_info['resolution']]['h'],
                            obj_speed = context_info['obj_speed']
                        )
        acc_reso = self.accuracy_prediction_2_reso.predict(
                            service_name = self.service_name_pipeline[0],
                            service_conf = {
                                'fps':conf_info['fps'],
                                'resolution':conf_info['resolution'],
                            },
                            obj_size = context_info['obj_size_norm'] * resolution_wh[conf_info['resolution']]['w']*resolution_wh[conf_info['resolution']]['h'],
                            obj_speed = context_info['obj_speed']
                        )
        
        
        if if_correct:
            coeff_window = {}
            if cur_coeff_window is None:
                coeff_window = copy.deepcopy(self.acc_reso_corrector.coeff_window)
            else:
                coeff_window = copy.deepcopy(cur_coeff_window)
            acc_reso = self.acc_reso_corrector.predict(x = acc_reso,
                                            cur_coeff_window = coeff_window)
        
        acc = acc_fps * acc_reso
        
        return acc

    def exe_pre_detect(self, context_info, conf_info, if_correct, cur_coeff_window=None):
        
        key=''
        if conf_info['edge_serv_num'] > 0:
            key=f"execute_device=edge4#resolution={conf_info['resolution']}"
        else:
            key=f"execute_device=cloud.kubeedge#resolution={conf_info['resolution']}"


        delay = self.exe_pre_detect_dict[key]

        if if_correct:
            coeff_window = {}
            if cur_coeff_window is None:
                coeff_window = copy.deepcopy(self.exe_corrector_detect.coeff_window)
            else:
                coeff_window = copy.deepcopy(cur_coeff_window)
            delay = self.exe_corrector_detect.predict(x = delay,
                                                      cur_coeff_window = coeff_window)

        return delay
    
    def exe_pre_classify(self, context_info, conf_info, if_correct, cur_coeff_window=None):

        key=''
        if conf_info['edge_serv_num'] > 1:
            key=f"execute_device=edge4#resolution={conf_info['resolution']}"
        else:
            key=f"execute_device=cloud.kubeedge#resolution={conf_info['resolution']}"

        delay = (self.exe_pre_classify_dict[key])*context_info['obj_num']

        if if_correct:
            coeff_window = {}
            if cur_coeff_window is None:
                coeff_window = copy.deepcopy(self.exe_corrector_classify.coeff_window)
            else:
                coeff_window = copy.deepcopy(cur_coeff_window)
            delay = self.exe_corrector_classify.predict(x = delay,
                                                        cur_coeff_window = coeff_window)

        return delay
    
    def trans_pre(self, context_info, conf_info, if_correct, cur_coeff_window=None):

        total_service_num = len(self.service_name_pipeline) 
        key=''
        delay = None

        if conf_info['edge_serv_num'] < total_service_num:
            key=f"resolution={conf_info['resolution']}#fps={str(int(conf_info['fps']))}#encoding=mp4v#buffer_size={str(int(conf_info['buffer_size']))}"
            file_size = self.file_size_dict[key]
            x = 0
            if context_info['band_Mbps'] > 0:
                x = file_size / context_info['band_Mbps']
            else:
                x = file_size / 1

            a = 3.5905589868141545
            b = 1.372854018944592

            y=(math.exp(a))*(x**b) 

            delay = y/conf_info['buffer_size']
        
        else:
            delay = 0

        new_delay = delay

        if if_correct and conf_info['edge_serv_num'] >= total_service_num:

            coeff_window = {}
            if cur_coeff_window is None:
                coeff_window = copy.deepcopy(self.trans_corrector.coeff_window)
            else:
                coeff_window = copy.deepcopy(cur_coeff_window)
            new_delay = self.trans_corrector.predict(x = delay,
                                                    cur_coeff_window = coeff_window)
        return new_delay



class SimpleWaitPredictor():

    def __init__(self, max_diff_thr, min_diff_thr, min_anylze_thr, stable_thr, fit_thr, history_window_length):

        self.max_diff_thr = max_diff_thr
        self.min_diff_thr = min_diff_thr
        self.min_anylze_thr = min_anylze_thr

        self.stable_thr = stable_thr

        self.fit_thr = fit_thr

        self.stable_wait_delay = 0
        self.wait_delay_history_window = []
        self.history_window_length = history_window_length
        self.fitter = PolynomialFitter()
    

    def update_wait_delay_history_window(self, task_id, wait_delay):

        self.wait_delay_history_window.append({
            'task_id':task_id,
            'wait_delay':wait_delay
        })
        if len(self.wait_delay_history_window) > self.history_window_length:
            self.wait_delay_history_window.pop(0)

    def predict(self,task_id):
        
        if len(self.wait_delay_history_window) == 0:
            return self.stable_wait_delay
        
        latest_history_task_id = self.wait_delay_history_window[-1]['task_id']

        if task_id - latest_history_task_id >= self.max_diff_thr:
            self.wait_delay_history_window = []
            return self.stable_wait_delay


        x_list = []
        y_list = []
        for item in self.wait_delay_history_window:
            x_list.append(item['task_id'])
            y_list.append(item['wait_delay'])
        y_mean = np.mean(np.array(y_list))
        y_std = np.std(np.array(y_list))
        

        if (task_id - latest_history_task_id <= self.min_diff_thr) and \
           (len(y_list) >= self.min_anylze_thr):
            

            coeff_list, score = self.fitter.fit(x_list=x_list,y_list=y_list,n=1)
            
            if score >= self.fit_thr:

                fit_pre = self.fitter.predict(coeff_list=coeff_list,x=task_id)

                if fit_pre < 0:
                    fit_pre = 0

                return fit_pre
            
            coeff_list, score = self.fitter.fit(x_list=x_list,y_list=y_list,n=0)
            if score >= self.fit_thr:

                fit_pre = self.fitter.predict(coeff_list=coeff_list,x=task_id)
                if fit_pre > 0:
                    self.stable_wait_delay = fit_pre
                
                return fit_pre         

        return y_mean


class WaitDelayPredictor():

    def __init__(self, queue_param):
        
        param = queue_param['detect_edge']
        self.detect_edge_predictor = SimpleWaitPredictor(max_diff_thr=param['max_diff_thr'],
                                                min_diff_thr=param['min_diff_thr'],
                                                min_anylze_thr=param['min_anylze_thr'],
                                                stable_thr=param['stable_thr'],
                                                fit_thr=param['fit_thr'],
                                                history_window_length=param['history_window_length'])
        
        param = queue_param['detect_cloud']
        self.detect_cloud_predictor = SimpleWaitPredictor(max_diff_thr=param['max_diff_thr'],
                                                min_diff_thr=param['min_diff_thr'],
                                                min_anylze_thr=param['min_anylze_thr'],
                                                stable_thr=param['stable_thr'],
                                                fit_thr=param['fit_thr'],
                                                history_window_length=param['history_window_length'])
        
        param = queue_param['classify_edge']
        self.classify_edge_predictor = SimpleWaitPredictor(max_diff_thr=param['max_diff_thr'],
                                                min_diff_thr=param['min_diff_thr'],
                                                min_anylze_thr=param['min_anylze_thr'],
                                                stable_thr=param['stable_thr'],
                                                fit_thr=param['fit_thr'],
                                                history_window_length=param['history_window_length'])
        
        param = queue_param['classify_cloud']
        self.classify_cloud_predictor = SimpleWaitPredictor(max_diff_thr=param['max_diff_thr'],
                                                min_diff_thr=param['min_diff_thr'],
                                                min_anylze_thr=param['min_anylze_thr'],
                                                stable_thr=param['stable_thr'],
                                                fit_thr=param['fit_thr'],
                                                history_window_length=param['history_window_length'])

    def update_history(self, conf_info, task_info):

        if 'detect_wait_delay' in task_info:

            if conf_info['edge_serv_num'] >= 1:
                self.detect_edge_predictor.update_wait_delay_history_window(task_id=task_info['task_id'],
                                                                            wait_delay=task_info['detect_wait_delay'])
            else: 
                self.detect_cloud_predictor.update_wait_delay_history_window(task_id=task_info['task_id'],
                                                                             wait_delay=task_info['detect_wait_delay'])
        
        if 'classify_wait_delay' in task_info:

            if conf_info['edge_serv_num'] >= 2:
                self.classify_edge_predictor.update_wait_delay_history_window(task_id=task_info['task_id'],
                                                                            wait_delay=task_info['classify_wait_delay'])
            else: 
                self.classify_cloud_predictor.update_wait_delay_history_window(task_id=task_info['task_id'],
                                                                            wait_delay=task_info['classify_wait_delay'])


    def pre_detect(self, conf_info, latest_task_id):

        detect_wait_delay = 0
        if conf_info['edge_serv_num'] >= 1:
            detect_wait_delay = self.detect_edge_predictor.predict(latest_task_id)
        else:
            detect_wait_delay = self.detect_cloud_predictor.predict(latest_task_id)
        return detect_wait_delay

    def pre_classify(self, conf_info, latest_task_id):

        classify_wait_delay = 0
        if conf_info['edge_serv_num'] >=2:
            classify_wait_delay = self.classify_edge_predictor.predict(latest_task_id)
        else:
            classify_wait_delay = self.classify_cloud_predictor.predict(latest_task_id)
        return classify_wait_delay



class IntegratedSafePredictor():

    def __init__(self, kb_path, service_name_pipeline, 
                 corrector_param,
                 queue_param,
                 cluster_threshold):
        
        self.service_name_pipeline = []

        for service_name in service_name_pipeline: 
            if service_name == 'end':
                break
            else:
                self.service_name_pipeline.append(service_name)
        

        self.corrected_predictor = CorrectedPredictor(kb_path=kb_path,
                                                      service_name_pipeline=service_name_pipeline,
                                                      corrector_param=corrector_param,
                                                      cluster_threshold = cluster_threshold)
        
        self.wait_delay_predictor = WaitDelayPredictor(queue_param=queue_param)
        
        

    def update_corrector(self, context_info, conf_info, task_info):
        

        self.wait_delay_predictor.update_history(conf_info=conf_info,
                                                 task_info=task_info)

        self.corrected_predictor.update_corrector(context_info = context_info,
                                                 conf_info=conf_info,
                                                 task_info=task_info)

    def acc_pre(self,context_info,conf_info,if_correct):

        all_coeff_window = self.corrected_predictor.get_all_coeff_window_by_context(context_info = context_info)

        coeff_window = all_coeff_window['acc_coeff_window']

        return self.corrected_predictor.acc_pre(context_info=context_info,
                                                conf_info = conf_info,
                                                if_correct=if_correct,
                                                cur_coeff_window=coeff_window)
    

    def delay_pre(self, context_info,conf_info,latest_task_id,if_correct, record_path = None):

        exe_detect = 0 
        wait_detect = 0
        trans = 0
        exe_classify = 0
        wait_classify = 0

        all_coeff_window = self.corrected_predictor.get_all_coeff_window_by_context(context_info = context_info)

        coeff_window = all_coeff_window['detect_coeff_window']
        exe_detect = self.exe_pre_detect(context_info=context_info,
                                        conf_info=conf_info,
                                        if_correct=if_correct,
                                        cur_coeff_window = coeff_window)
        wait_detect = self.wait_detect_pre(conf_info=conf_info,
                                            latest_task_id=latest_task_id)
        

        coeff_window = all_coeff_window['trans_coeff_window']
        trans = self.trans_pre(context_info=context_info,
                                conf_info=conf_info,
                                if_correct=if_correct)
        

        if len(self.service_name_pipeline) > 1:

            coeff_window = all_coeff_window['classify_coeff_window']
            exe_classify = self.exe_pre_classify(context_info=context_info,
                                                conf_info=conf_info,
                                                if_correct=if_correct,
                                                cur_coeff_window = coeff_window)
            wait_classify = self.wait_classify_pre(conf_info=conf_info,
                                                latest_task_id=latest_task_id)




        return exe_detect + exe_classify + trans + wait_detect + wait_classify


    def exe_pre_detect(self,context_info,conf_info,if_correct,cur_coeff_window=None):

        coeff_window = None
        if cur_coeff_window is not None:
            coeff_window = copy.deepcopy(cur_coeff_window)

        return self.corrected_predictor.exe_pre_detect(context_info=context_info,
                                                       conf_info=conf_info,
                                                       if_correct=if_correct,
                                                       cur_coeff_window = coeff_window)

    def exe_pre_classify(self,context_info,conf_info,if_correct,cur_coeff_window=None):

        coeff_window = None
        if cur_coeff_window is not None:
            coeff_window = copy.deepcopy(cur_coeff_window)

        return self.corrected_predictor.exe_pre_classify(context_info=context_info,
                                                         conf_info=conf_info,
                                                         if_correct=if_correct,
                                                         cur_coeff_window = coeff_window)
    

    def trans_pre(self, context_info, conf_info,if_correct,cur_coeff_window=None):

        coeff_window = None
        if cur_coeff_window is not None:
            coeff_window = copy.deepcopy(cur_coeff_window)

        return self.corrected_predictor.trans_pre(context_info=context_info,
                                                  conf_info=conf_info,
                                                  if_correct=if_correct,
                                                  cur_coeff_window = coeff_window)
    
    def wait_detect_pre(self,conf_info,latest_task_id):

        return self.wait_delay_predictor.pre_detect(conf_info=conf_info,
                                                    latest_task_id=latest_task_id)
    
    def wait_classify_pre(self,conf_info,latest_task_id):

        return self.wait_delay_predictor.pre_classify(conf_info=conf_info,
                                                      latest_task_id=latest_task_id)
    
    



