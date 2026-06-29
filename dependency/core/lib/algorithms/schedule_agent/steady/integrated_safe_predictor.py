

import numpy as np
import math
import json
import copy

from .accuracy_prediction import AccuracyPrediction2fps,AccuracyPrediction2reso, resolution_wh
from .correct_record import CorrectRecord


# 线性拟合器
# 可以返回拟合的超参数和拟合的评估结果(越逼近1越好)，返回一个列表和一个浮点数; 也可以基于超参数进行单个值的预测
class PolynomialFitter:

    def __init__(self):
        pass
    
    # 按照n次方多项式来进行拟合，返回结果是列表，列表里包含了n个参数，其中i号索引对应着i次方的系数
    def fit(self, x_list:list, y_list:list, n:int):
        # 采样点数量大于1的时候，按照参数进行拟合   
        x_np = np.array(x_list)
        y_np = np.array(y_list)

        # 注意, 当n=0的时候，r2的传统计算方法不可用。因为n=0的时候会计算平均值，结果r2算出来就是0
        # 因此换一个计算方法，求

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
            # 计算R²
            ss_res = np.sum((y_np - pre_y_np) ** 2)  # 残差平方和
            ss_tot = np.sum((y_np - np.mean(y_np)) ** 2)  # 总平方和

            r2 = 1

            if ss_tot == 0:
                r2 = 1
            else:
                r2 = 1 - (ss_res / ss_tot)

            score = r2
            # 反向切片，使得返回值里的系数是从0次方开始从低到高
            return coeff_list[::-1], score
        
        # 采样点数量等于1且x值不为0的时候，视为y=0+kx。此时最低次方的系数是0，其次才是y_list[0]/x_list[0]
        # 但是注意:如果x_list[0]本身取值为0怎么办呢？理论上这是不应该出现的情况
        elif len(x_list)==1 and x_list[0]!=0: 

            coeff_list = [0]
            coeff_list.append(y_list[0]/x_list[0])

            score = 1
            return coeff_list, score
        
         # 采样点数量等于1且x值不为0的时候，视为y = c ，也就是说0次方返回一个常熟
        elif len(x_list)==1 and x_list[0]==0: 

            coeff_list = [y_list[0]]
            score = 1
            return coeff_list, score

    def predict(self,coeff_list:list, x):

        # 反向切片，得到次方从高到低排列的多项式系数
        coeff_np = np.array(coeff_list[::-1])

        x_np = np.array([x])
        pre_y_np = np.polyval(coeff_np, x_np)
        y = pre_y_np[0]
        
        return y

# 单一的矫正器
class SimpleCorrector:

    def __init__(self,n,coeff_window_length,x_sample_window_length, y_sample_window_length):
        
        self.n = n # 最高次项是N
        self.coeff_window_length = coeff_window_length # 系数窗口长度
        self.x_sample_window_length = x_sample_window_length # x采样点的种类窗口
        self.y_sample_window_length = y_sample_window_length # 每一个x对应的y的种类窗口
        self.polynomial_fitter = PolynomialFitter()
        
        # 存储各个系数的窗口，元素为列表
        self.coeff_window = []
        for i in range(self.n+1):
            self.coeff_window.append([])
        
        # x的采样窗口
        self.x_sample_window = []
        # x对应y的采样窗口
        self.x_y_sample_window = {}
        
    # 首先更新采样点
    def update_sample_window(self,x_value,y_value):

        # 首先更新x_sample_window。如果此前没有出现过x_value，需要更新列表；新添加的时候还要注意是否已经超出了窗口长度，如果超出就要删掉早期出现的x
        # 且删除旧x的时候还要注意对x_y_sample_window的删除
        if x_value not in self.x_sample_window:

            self.x_sample_window.append(x_value)

            if len(self.x_sample_window) > self.x_sample_window_length:
                del_x_value = self.x_sample_window.pop(0)
                if del_x_value in self.x_y_sample_window:
                    del self.x_y_sample_window[del_x_value]
        
        else:
            pass

        # 然后更新 self.x_y_sample_window,一旦超出长度也要删除
        if x_value not in self.x_y_sample_window:
            self.x_y_sample_window[x_value] = [y_value]
        else:
            self.x_y_sample_window[x_value].append(y_value)
            if len(self.x_y_sample_window[x_value]) > self.y_sample_window_length:
                self.x_y_sample_window[x_value].pop(0) #pop(0)表示删除列表中序号为0的第一个元素

    # 利用当前的采样窗口，取出采样结果并进行拟合，得到新的拟合超参数多项式
    # 新的拟合参数计算结果会被存入历史窗口中，当然，只有确实存在足够多的x-y采样点的时候才能拟合得到参数
    # 如果x_list和y_list长度都为0，那根本无法更新历史窗口，因为不存在可拟合的内容
    def update_coeff_window(self):

        x_list = []
        y_list = []

        for x in self.x_y_sample_window.keys():
            
            if len(self.x_y_sample_window[x]) > 0:

                x_list.append(x)
                y_list.append(sum(self.x_y_sample_window[x])/len(self.x_y_sample_window[x]))
        
        # 如果x_list长度为0，那么根本没有拟合的必要

        if len(x_list) == 0:
            return
        
        coeff_list = []  #coeff_list[i]用于表示i次方对应的系数

        # 根据采样点数量选择不同的指数
        if len(x_list) == 1:

            coeff_list, score = self.polynomial_fitter.fit(x_list=x_list,
                                                        y_list=y_list,
                                                        n=1)
        
        # 比1多但是比理想情况少，那么只能拟合len(x_list)-1次方
        elif len(x_list) > 1 and len(x_list) < self.n + 1:
            
            coeff_list, score = self.polynomial_fitter.fit(x_list=x_list,
                                                        y_list=y_list,
                                                        n=len(x_list)-1)
        
        # 只有采样点数量足够多的时候，才能拟合n次方
        elif len(x_list) >= self.n + 1:
            coeff_list, score = self.polynomial_fitter.fit(x_list=x_list,
                                                        y_list=y_list,
                                                        n=self.n)

        # 总共需要self.n+1个参数，如果拟合得到的参数数量无法达标，那就持续增加0作为弥补，用于代表高次方的系数
        for i in range(len(coeff_list),self.n + 1):
            coeff_list.append(0.0)
        
        # 这里是记录每一个次方对应的最新的参数值。每一个次方对应的拟合系数本身也保留为窗口，方便后续使用
        for i in range(self.n+1):
            self.coeff_window[i].append(coeff_list[i])
            if len(self.coeff_window[i]) > self.coeff_window_length:
                self.coeff_window[i].pop(0)
    

    
    # 如果出现了明显过于离谱的预测结果，说明权重很成问题，此时应该及时删除旧的权重预测结果
    def delete_coeff_window_by_one(self):

        for i in range(self.n+1):
            if len(self.coeff_window[i]) > 0:
                self.coeff_window[i].pop(-1)

    # 需要从窗口中提取超参数来进行多项式拟合预测，暂时先用最新版本而非均值
    def predict(self,x,cur_coeff_window):

        coeff_window = copy.deepcopy(cur_coeff_window)

        if len(coeff_window[0]) == 0:
            return x

        coeff_list = []
        for i in range(self.n+1):
            coeff_list.append(coeff_window[i][-1])
            #coeff_list.append(sum(coeff_window[i])/len(coeff_window[i]))
        
        y = self.polynomial_fitter.predict(coeff_list=coeff_list,
                                           x=x)


        # 注意，算出来的y可能小于0，此时有问题
        if y<0:
            y = x

        return y

# 含矫正器的执行时延/传输时延/精度预估器（不含等待时延）
class CorrectedPredictor():

    def __init__(self,kb_path, service_name_pipeline, corrector_param):
        # TODO：为不同的工况建立不同的精度曲线
        self.kb_path = kb_path

        self.service_name_pipeline = []
        # self.service_name_pipeline 形如 ['face-detection','gender-classification']
        for service_name in service_name_pipeline: #里面绝对不能含有end
            if service_name == 'end':
                break
            else:
                self.service_name_pipeline.append(service_name)


        # 完成加载,为检测、分类、file_size各自加载库
        self.accuracy_prediction_2_fps = AccuracyPrediction2fps()
        self.accuracy_prediction_2_reso = AccuracyPrediction2reso()

        
        self.exe_pre_detect_dict = self.read_dict(kb_path + '/' + service_name_pipeline[0] + '.json')
        self.exe_pre_classify_dict = {}
        if len(service_name_pipeline)>1:
            self.exe_pre_classify_dict = self.read_dict(kb_path + '/' + service_name_pipeline[1] + '.json')
  

        self.file_size_dict = self.read_dict(kb_path + '/' + 'file_size' + '.json')

        # 需要分别为检测时延、分类时延、传输时延设置矫正器
        
        param = corrector_param['detect']
        self.exe_corrector_detect = SimpleCorrector(n = param['n'],
                                                    coeff_window_length = param['coeff_window_length'],
                                                    x_sample_window_length = param['x_sample_window_length'],
                                                    y_sample_window_length = param['y_sample_window_length'])
        param = corrector_param['classify']
        self.exe_corrector_classify = SimpleCorrector(n = param['n'],
                                                    coeff_window_length = param['coeff_window_length'],
                                                    x_sample_window_length = param['x_sample_window_length'],
                                                    y_sample_window_length = param['y_sample_window_length'])
        param = corrector_param['trans']
        self.trans_corrector = SimpleCorrector(n = param['n'],
                                                coeff_window_length = param['coeff_window_length'],
                                                x_sample_window_length = param['x_sample_window_length'],
                                                y_sample_window_length = param['y_sample_window_length'])
        
        param = corrector_param['acc']
        self.acc_reso_corrector = SimpleCorrector(n = param['n'],
                                                coeff_window_length = param['coeff_window_length'],
                                                x_sample_window_length = param['x_sample_window_length'],
                                                y_sample_window_length = param['y_sample_window_length'])
        

    def read_dict(self, file_path):
        dict_data={}
        try:  
            with open(file_path, 'r') as file:  
                dict_data = json.load(file)  
        except FileNotFoundError:   
            dict_data = {} 
        return dict_data
    
    # 根据真实性能表现处理矫正器
    def update_corrctor(self, context_info, conf_info, task_info):


        # 更新矫正器相关参数的时候，不需要置入cur_coeff_window参数，默认为None

        # 注意采样x的时候，if_correct都是false，表示初始估计结果

        if 'real_exe_detect' in task_info:

            self.exe_corrector_detect.update_sample_window(x_value=self.exe_pre_detect(context_info=context_info, conf_info=conf_info,if_correct=False),
                                                        y_value=task_info['real_exe_detect'])
            self.exe_corrector_detect.update_coeff_window()

        if 'real_exe_classify' in task_info:
        
            self.exe_corrector_classify.update_sample_window(x_value=self.exe_pre_classify(context_info=context_info, conf_info=conf_info,if_correct=False),
                                                            y_value=task_info['real_exe_classify'])
            self.exe_corrector_classify.update_coeff_window()
        
        if 'real_trans' in task_info:
        
            self.trans_corrector.update_sample_window(x_value=self.trans_pre(context_info=context_info, conf_info=conf_info,if_correct=False),
                                                    y_value=task_info['real_trans'])
            self.trans_corrector.update_coeff_window()
        

        if 'real_acc_reso' in task_info:

            self.acc_reso_corrector.update_sample_window(x_value=self.acc_pre(context_info=context_info, conf_info=conf_info,if_correct=False),
                                                    y_value=task_info['real_acc_reso'])
            self.acc_reso_corrector.update_coeff_window()
    

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
        
        #  "execute_device=edge4#resolution=540p": 0.01659703254699707,
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

        #  "execute_device=edge4#resolution=540p": 0.01659703254699707,
        key=''
        if conf_info['edge_serv_num'] > 1:
            key=f"execute_device=edge4#resolution={conf_info['resolution']}"
        else:
            key=f"execute_device=cloud.kubeedge#resolution={conf_info['resolution']}"

        # 所得delay本身就是平均每一帧的结果
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
        # "resolution=240p#fps=2#encoding=mp4v#buffer_size=7":

        # 应该根据服务数量判断传输时延如何
        # self.service_name_pipeline 形如 ['face-detection','gender-classification']

        total_service_num = len(self.service_name_pipeline) 

        key=''
        delay = None
        # edge_serv_num的数量比总的服务数量小的时候，才会产生云边传输时延
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

            # y算出的结果是整个task的传输时延，必须除以self.buffer_size才能得到平均每一帧的结果
            delay = y/conf_info['buffer_size']
        
        else:
            delay = 0

        new_delay = delay

        # 需要校正且当前确实存在云边传输时延的时候，才需要进行校正
        # 如果需要校正单云边传输时延不存在，没有校正的必要
        if if_correct and conf_info['edge_serv_num'] >= total_service_num:

            # 获取当前传输时延预估器的校正参数
            coeff_window = {}
            if cur_coeff_window is None:
                coeff_window = copy.deepcopy(self.trans_corrector.coeff_window)
            else:
                coeff_window = copy.deepcopy(cur_coeff_window)

            new_delay = self.trans_corrector.predict(x = delay,
                                                    cur_coeff_window = coeff_window)
        
            while new_delay - delay > 1.5:
                self.trans_corrector.delete_coeff_window_by_one()
                new_delay = self.trans_corrector.predict(x = delay,
                                                        cur_coeff_window = coeff_window)

        return new_delay

# 单一的等待时延预估器
class SimpleWaitPredictor():

    def __init__(self, max_diff_thr, min_diff_thr, min_anylze_thr, stable_thr, fit_thr, history_window_length):

        self.max_diff_thr = max_diff_thr
        self.min_diff_thr = min_diff_thr
        self.min_anylze_thr = min_anylze_thr

        # 如果线性拟合结果的一次项系数的绝对值小于stable_thr，则认为是稳定值
        self.stable_thr = stable_thr

        #score高于fit_thr被认为线性拟合成功
        self.fit_thr = fit_thr

        self.stable_wait_delay = 0
        self.wait_delay_history_window = []
        self.history_window_length = history_window_length
        self.fitter = PolynomialFitter()
    
    # 更新当前历史窗口
    def update_wait_delay_history_window(self, task_id, wait_delay):

        self.wait_delay_history_window.append({
            'task_id':task_id,
            'wait_delay':wait_delay
        })
        if len(self.wait_delay_history_window) > self.history_window_length:
            self.wait_delay_history_window.pop(0)
    
    # 预测最新等待时延，需要task_id作为超参数
    def predict(self,task_id):
        # 如果历史窗口为空，或者当前task_id过于新，都说明历史窗口已经过时了
        
        if len(self.wait_delay_history_window) == 0:
            return self.stable_wait_delay
        
        latest_history_task_id = self.wait_delay_history_window[-1]['task_id']

        if task_id - latest_history_task_id >= self.max_diff_thr:
            self.wait_delay_history_window = []
            return self.stable_wait_delay

            
        
        # 历史窗口不为空的时候，需要进行分析
        x_list = []
        y_list = []
        for item in self.wait_delay_history_window:
            x_list.append(item['task_id'])
            y_list.append(item['wait_delay'])
        y_mean = np.mean(np.array(y_list))
        y_std = np.std(np.array(y_list))
        
        # 如果差异不大，且已有数据足够分析，看看是否存在线性关系;没有线性关系，就看看是否稳定。

        if (task_id - latest_history_task_id <= self.min_diff_thr) and \
           (len(y_list) >= self.min_anylze_thr):
            
            # 进行一次方的线性拟合
            coeff_list, score = self.fitter.fit(x_list=x_list,y_list=y_list,n=1)
            
            # 如果确实有线性关系，返回基于线性拟合的预测结果
            if score >= self.fit_thr:

                fit_pre = self.fitter.predict(coeff_list=coeff_list,x=task_id)

                # 但是，如果fit_pre小于0，那就有问题
                if fit_pre < 0:
                    fit_pre = 0

                return fit_pre
            

            
            # 如果不存在线性关系，看看是否是稳定常数值
            coeff_list, score = self.fitter.fit(x_list=x_list,y_list=y_list,n=0)
            if score >= self.fit_thr:

                fit_pre = self.fitter.predict(coeff_list=coeff_list,x=task_id)
                if fit_pre > 0:
                    self.stable_wait_delay = fit_pre
                
                return fit_pre
            



        # 如果差异不大但是可分析数据不够，或者如果差异不够大也不够小，那就没法进行基于线性的预估，直接返回平均值

        return y_mean

# 综合的等待时延预估器
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
            else: #为0的时候，在云端等待
                self.detect_cloud_predictor.update_wait_delay_history_window(task_id=task_info['task_id'],
                                                                             wait_delay=task_info['detect_wait_delay'])
        
        if 'classify_wait_delay' in task_info:

            if conf_info['edge_serv_num'] >= 2:
                self.classify_edge_predictor.update_wait_delay_history_window(task_id=task_info['task_id'],
                                                                            wait_delay=task_info['classify_wait_delay'])
            else: #小于2的时候，在云端等待
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


# 统一的精度、时延综合预估器
class IntegratedSafePredictor():

    def __init__(self, kb_path, service_name_pipeline, 
                 corrector_param,
                 queue_param):
        
        self.service_name_pipeline = []
        # self.service_name_pipeline 形如 ['face-detection','gender-classification']
        for service_name in service_name_pipeline: #里面绝对不能含有end
            if service_name == 'end':
                break
            else:
                self.service_name_pipeline.append(service_name)
        

        self.corrected_predictor = CorrectedPredictor(kb_path=kb_path,
                                                      service_name_pipeline=service_name_pipeline,
                                                      corrector_param=corrector_param)
        
        self.wait_delay_predictor = WaitDelayPredictor(queue_param=queue_param)
        
        
    # 需要根据真实感知结果更新调度器的底层逻辑时，调用此方法，更新矫正器和等待时延预估器相关逻辑
    def update_corrector(self, context_info, conf_info, task_info):
        
        # 然后更新等待时延预估器的历史窗口
        self.wait_delay_predictor.update_history(conf_info=conf_info,
                                                 task_info=task_info)
        # 更新矫正器的内部状态
        self.corrected_predictor.update_corrctor(context_info = context_info,
                                                 conf_info=conf_info,
                                                 task_info=task_info)

    # 精度预估
    def acc_pre(self,context_info,conf_info,if_correct):

        coeff_window = self.corrected_predictor.acc_reso_corrector.coeff_window

        return self.corrected_predictor.acc_pre(context_info=context_info,
                                                conf_info = conf_info,
                                                if_correct=if_correct,
                                                cur_coeff_window=coeff_window)
    
    # 整体时延预估
    def delay_pre(self, context_info,conf_info,latest_task_id,if_correct, record_path = None):

        record_param = {}
        record_param['coeff_window']={}
        record_param['x_y_sample_window']={}
        record_param['wait_delay_history_window']={}
        record_param['wait_delay_history_window']['edge']={}
        record_param['wait_delay_history_window']['cloud']={}

        # 初始化十分重要，不可避免
        exe_detect = 0 
        wait_detect = 0
        trans = 0
        exe_classify = 0
        wait_classify = 0


        # 处理检测的执行时延、等待时延
        coeff_window = self.corrected_predictor.exe_corrector_detect.coeff_window
        record_param['coeff_window']['detect'] = coeff_window
        record_param['x_y_sample_window']['detect'] = self.corrected_predictor.exe_corrector_detect.x_y_sample_window
        exe_detect = self.exe_pre_detect(context_info=context_info,
                                        conf_info=conf_info,
                                        if_correct=if_correct,
                                        cur_coeff_window = coeff_window)
        
        record_param['wait_delay_history_window']['edge']['detect'] = self.wait_delay_predictor.detect_edge_predictor.wait_delay_history_window
        record_param['wait_delay_history_window']['cloud']['detect'] = self.wait_delay_predictor.detect_cloud_predictor.wait_delay_history_window
        wait_detect = self.wait_detect_pre(conf_info=conf_info,
                                            latest_task_id=latest_task_id)
        
       

        # 处理传输时延
        coeff_window = self.corrected_predictor.trans_corrector.coeff_window
        record_param['coeff_window']['trans'] = coeff_window
        record_param['x_y_sample_window']['trans'] = self.corrected_predictor.trans_corrector.x_y_sample_window
        trans = self.trans_pre(context_info=context_info,
                                conf_info=conf_info,
                                if_correct=if_correct)
        
        # 然后是记录当前信息(不包含分类)
        record_param['context_info'] = context_info
        record_param['conf_info'] = conf_info
        record_param['task_total_delay'] = {}
        record_param['task_total_delay']['exe_detect'] = exe_detect
        record_param['task_total_delay']['wait_detect'] = wait_detect
        record_param['task_total_delay']['trans'] = trans

        # 如果服务的数量比1大，那么就要分类的执行时延、等待时延
        if len(self.service_name_pipeline) > 1:

            coeff_window = self.corrected_predictor.exe_corrector_classify.coeff_window
            record_param['coeff_window']['classify'] = coeff_window
            record_param['x_y_sample_window']['classify'] = self.corrected_predictor.exe_corrector_classify.x_y_sample_window
            exe_classify = self.exe_pre_classify(context_info=context_info,
                                                conf_info=conf_info,
                                                if_correct=if_correct,
                                                cur_coeff_window = coeff_window)
            
            record_param['wait_delay_history_window']['edge']['classify'] = self.wait_delay_predictor.classify_edge_predictor.wait_delay_history_window
            record_param['wait_delay_history_window']['cloud']['classify'] = self.wait_delay_predictor.classify_cloud_predictor.wait_delay_history_window
            wait_classify = self.wait_classify_pre(conf_info=conf_info,
                                                latest_task_id=latest_task_id)
            
            record_param['task_total_delay']['exe_classify'] = exe_classify
            record_param['task_total_delay']['wait_classify'] = wait_classify


        # 现在得到record_param了
        # 
        task_id = latest_task_id-1
        task_id_for_pre = latest_task_id

        if record_path is not None:

            correct_record = CorrectRecord(task_id = task_id,
                                        task_id_for_pre = task_id_for_pre,
                                        record_param = record_param)
            
            CorrectRecord.write_record(correct_record = correct_record,
                                       file_path = record_path)

        # 没有分类时延的时候wait_detect + wait_classify是0

        return exe_detect + exe_classify + trans + wait_detect + wait_classify



    # 检测执行时延
    def exe_pre_detect(self,context_info,conf_info,if_correct,cur_coeff_window=None):

        coeff_window = None
        if cur_coeff_window is not None:
            coeff_window = copy.deepcopy(cur_coeff_window)

        return self.corrected_predictor.exe_pre_detect(context_info=context_info,
                                                       conf_info=conf_info,
                                                       if_correct=if_correct,
                                                       cur_coeff_window = coeff_window)
    
    # 分类执行时延
    def exe_pre_classify(self,context_info,conf_info,if_correct,cur_coeff_window=None):

        coeff_window = None
        if cur_coeff_window is not None:
            coeff_window = copy.deepcopy(cur_coeff_window)

        return self.corrected_predictor.exe_pre_classify(context_info=context_info,
                                                         conf_info=conf_info,
                                                         if_correct=if_correct,
                                                         cur_coeff_window = coeff_window)
    
    # 传输时延
    def trans_pre(self, context_info, conf_info,if_correct,cur_coeff_window=None):

        coeff_window = None
        if cur_coeff_window is not None:
            coeff_window = copy.deepcopy(cur_coeff_window)

        return self.corrected_predictor.trans_pre(context_info=context_info,
                                                  conf_info=conf_info,
                                                  if_correct=if_correct,
                                                  cur_coeff_window = coeff_window)
    
    # 检测等待时延
    def wait_detect_pre(self,conf_info,latest_task_id):

        return self.wait_delay_predictor.pre_detect(conf_info=conf_info,
                                                    latest_task_id=latest_task_id)
    
    # 分类等待时延
    def wait_classify_pre(self,conf_info,latest_task_id):

        return self.wait_delay_predictor.pre_classify(conf_info=conf_info,
                                                      latest_task_id=latest_task_id)
    
    



