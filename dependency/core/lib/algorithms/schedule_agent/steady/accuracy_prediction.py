import numpy as np

# 定义指数函数模型
def exponential_function_1(x, a, b, c):
    return a + b * np.exp(c * x)

def exponential_function_2(x, a, b, c, d):
    return a + b * np.exp(c * (x - d))


'''
resolution_dict = {'1080p': (1920, 1080),
                       '900p': (1440, 900),
                       '720p': (1280, 720),
                       '600p': (800, 600),
                       '540p': (960, 540),
                       '480p': (640, 480),
                       '360p': (640, 360),
                       '240p': (320, 240)}
'''
# resolution ['240p', '360p', '480p', '540p', '720p', '900p', '1080p']

resolution_wh = {
    "240p": {
        "w": 320,
        "h": 240
    },
    "360p": {
        "w": 640,
        "h": 360
    },
    "480p": {
        "w": 640,
        "h": 480
    },
    "540p": {
        "w": 960,
        "h": 540
    },
    "600p": {
        "w": 800,
        "h": 600
    },
    "720p": {
        "w": 1280,
        "h": 900
    },
    "900p": {
        "w": 1440,
        "h": 900
    },
    "1080p": {
        "w": 1920,
        "h": 1080
    }
}

'''
# 原版精度
resolution_wh = {
    "240p": {
        "w": 320,
        "h": 240
    },
    "360p": {
        "w": 480,
        "h": 360
    },
    "480p": {
        "w": 640,
        "h": 480
    },
    "540p": {
        "w": 960,
        "h": 540
    },
    "630p": {
        "w": 1120,
        "h": 630
    },
    "720p": {
        "w": 1280,
        "h": 720
    },
    "810p": {
        "w": 1440,
        "h": 810
    },
    "900p": {
        "w": 1600,
        "h": 900
    },
    "990p": {
        "w": 1760,
        "h": 990
    },
    "1080p": {
        "w": 1920,
        "h": 1080
    }
}
'''


# 将基于fps和基于reso的精度预估器分开
# 受采样限制，只有基于reso的预估器能够得到矫正，其他情况下难以做到

class AccuracyPrediction2fps():
    def __init__(self):
        # TODO：为不同的工况建立不同的精度曲线
        pass
    
    def predict(self, service_name, service_conf, obj_size=None, obj_speed=None):
        # if service_name == 'face_detection':
        # 为了增加泛用性，干脆扩大服务的范围
        if  'detection' in service_name:
            temp_fps = service_conf['fps']
            temp_reso = resolution_wh[service_conf['resolution']]['h']
            temp_obj_size = obj_size
            temp_obj_speed = obj_speed
            
            if temp_obj_speed is None or temp_obj_speed == -1:  # 当不存在速度字段时，默认速度为第二档
                a1 = 0.97  
                b1 = -1.20
                c1 = -0.78
                acc_2_fps = exponential_function_1(np.array([temp_fps]), a1, b1, c1)
                acc_2_fps = acc_2_fps[0]
            else:
                if temp_obj_speed <= 260:  # 0.98, -0.75, -0.85
                    a1 = 0.98 
                    b1 = -0.75
                    c1 = -0.85
                    acc_2_fps = exponential_function_1(np.array([temp_fps]), a1, b1, c1)
                    acc_2_fps = acc_2_fps[0]
                elif temp_obj_speed > 260 and temp_obj_speed <= 520:  # 0.97, -1.20, -0.78
                    a1 = 0.97  
                    b1 = -1.20
                    c1 = -0.78
                    acc_2_fps = exponential_function_1(np.array([temp_fps]), a1, b1, c1)
                    acc_2_fps = acc_2_fps[0]
                elif temp_obj_speed > 520 and temp_obj_speed <= 780:  # 0.985, -0.9, -0.29
                    a1 = 0.985
                    b1 = -0.9
                    c1 = -0.29
                    acc_2_fps = exponential_function_1(np.array([temp_fps]), a1, b1, c1)
                    acc_2_fps = acc_2_fps[0]
                else:  # 1.0, -0.84, -0.18
                    a1 = 1.0  
                    b1 = -0.84
                    c1 = -0.18
                    acc_2_fps = exponential_function_1(np.array([temp_fps]), a1, b1, c1)
                    acc_2_fps = acc_2_fps[0]
         
            acc = acc_2_fps
            if acc<0:
                acc = 0
            return acc
        
        else:
            return 1
    
    
class AccuracyPrediction2reso():
    def __init__(self):
        # TODO：为不同的工况建立不同的精度曲线
        pass
    
    def predict(self, service_name, service_conf, obj_size=None, obj_speed=None):
        # if service_name == 'face_detection':
        # 为了增加泛用性，干脆扩大服务的范围
        if  'detection' in service_name:
            temp_fps = service_conf['fps']
            temp_reso = resolution_wh[service_conf['resolution']]['h']
            temp_obj_size = obj_size
            temp_obj_speed = obj_speed
            
            if temp_obj_size is None or temp_obj_size == -1:  # 当不存在目标大小字段时，默认大小为第二档
                a2 = 0.99  
                b2 = -0.47
                c2 = -0.008
                d2 = 350
                
                acc_2_reso = exponential_function_2(np.array([temp_reso]), a2, b2, c2, d2)
                acc_2_reso = acc_2_reso[0]
            elif temp_obj_size == 0:  # 当目标大小为0时，说明场景中此时没有目标，默认目标为最大的大小
                a2 = 0.99  # 计算当前分辨率下的精度
                b2 = -0.2
                c2 = -0.008
                d2 = 350
                
                acc_2_reso = exponential_function_2(np.array([temp_reso]), a2, b2, c2, d2)
                acc_2_reso = acc_2_reso[0]
            else:
                if temp_obj_size <= 50000:  # 0.98, -0.63, -0.006, 350
                    a2 = 0.98  # 计算当前分辨率下的精度
                    b2 = -0.63
                    c2 = -0.006
                    d2 = 350
                    
                    acc_2_reso = exponential_function_2(np.array([temp_reso]), a2, b2, c2, d2)
                    acc_2_reso = acc_2_reso[0]
                elif temp_obj_size > 50000 and temp_obj_size <= 100000:  # 0.99, -0.47, -0.008, 350
                    a2 = 0.99  # 计算当前分辨率下的精度
                    b2 = -0.47
                    c2 = -0.008
                    d2 = 350
                    
                    acc_2_reso = exponential_function_2(np.array([temp_reso]), a2, b2, c2, d2)
                    acc_2_reso = acc_2_reso[0]
                else:  # 0.99, -0.2, -0.008, 350
                    a2 = 0.99  # 计算当前分辨率下的精度
                    b2 = -0.2
                    c2 = -0.008
                    d2 = 350
                    
                    acc_2_reso = exponential_function_2(np.array([temp_reso]), a2, b2, c2, d2)
                    acc_2_reso = acc_2_reso[0]
            
            acc = acc_2_reso  # 将分辨率下的精度与帧率下的精度相乘，得到最终预测的精度
            if acc<0:
                acc = 0
            return acc
        
        else:
            return 1
    
    

class AccuracyPrediction():
    def __init__(self):
        # TODO：为不同的工况建立不同的精度曲线
        pass
    
    def predict(self, service_name, service_conf, obj_size=None, obj_speed=None):
        # if service_name == 'face_detection':
        # 为了增加泛用性，干脆扩大服务的范围
        if  'detection' in service_name:
            temp_fps = service_conf['fps']
            temp_reso = resolution_wh[service_conf['resolution']]['h']
            temp_obj_size = obj_size
            temp_obj_speed = obj_speed
            
            if temp_obj_speed is None or temp_obj_speed == -1:  # 当不存在速度字段时，默认速度为第二档
                a1 = 0.97  
                b1 = -1.20
                c1 = -0.78
                acc_2_fps = exponential_function_1(np.array([temp_fps]), a1, b1, c1)
                acc_2_fps = acc_2_fps[0]
            else:
                if temp_obj_speed <= 260:  # 0.98, -0.75, -0.85
                    a1 = 0.98 
                    b1 = -0.75
                    c1 = -0.85
                    acc_2_fps = exponential_function_1(np.array([temp_fps]), a1, b1, c1)
                    acc_2_fps = acc_2_fps[0]
                elif temp_obj_speed > 260 and temp_obj_speed <= 520:  # 0.97, -1.20, -0.78
                    a1 = 0.97  
                    b1 = -1.20
                    c1 = -0.78
                    acc_2_fps = exponential_function_1(np.array([temp_fps]), a1, b1, c1)
                    acc_2_fps = acc_2_fps[0]
                elif temp_obj_speed > 520 and temp_obj_speed <= 780:  # 0.985, -0.9, -0.29
                    a1 = 0.985
                    b1 = -0.9
                    c1 = -0.29
                    acc_2_fps = exponential_function_1(np.array([temp_fps]), a1, b1, c1)
                    acc_2_fps = acc_2_fps[0]
                else:  # 1.0, -0.84, -0.18
                    a1 = 1.0  
                    b1 = -0.84
                    c1 = -0.18
                    acc_2_fps = exponential_function_1(np.array([temp_fps]), a1, b1, c1)
                    acc_2_fps = acc_2_fps[0]
                
            if temp_obj_size is None or temp_obj_size == -1:  # 当不存在目标大小字段时，默认大小为第二档
                a2 = 0.99  
                b2 = -0.47
                c2 = -0.008
                d2 = 350
                
                acc_2_reso = exponential_function_2(np.array([temp_reso]), a2, b2, c2, d2)
                acc_2_reso = acc_2_reso[0]
            elif temp_obj_size == 0:  # 当目标大小为0时，说明场景中此时没有目标，默认目标为最大的大小
                a2 = 0.99  # 计算当前分辨率下的精度
                b2 = -0.2
                c2 = -0.008
                d2 = 350
                
                acc_2_reso = exponential_function_2(np.array([temp_reso]), a2, b2, c2, d2)
                acc_2_reso = acc_2_reso[0]
            else:
                if temp_obj_size <= 50000:  # 0.98, -0.63, -0.006, 350
                    a2 = 0.98  # 计算当前分辨率下的精度
                    b2 = -0.63
                    c2 = -0.006
                    d2 = 350
                    
                    acc_2_reso = exponential_function_2(np.array([temp_reso]), a2, b2, c2, d2)
                    acc_2_reso = acc_2_reso[0]
                elif temp_obj_size > 50000 and temp_obj_size <= 100000:  # 0.99, -0.47, -0.008, 350
                    a2 = 0.99  # 计算当前分辨率下的精度
                    b2 = -0.47
                    c2 = -0.008
                    d2 = 350
                    
                    acc_2_reso = exponential_function_2(np.array([temp_reso]), a2, b2, c2, d2)
                    acc_2_reso = acc_2_reso[0]
                else:  # 0.99, -0.2, -0.008, 350
                    a2 = 0.99  # 计算当前分辨率下的精度
                    b2 = -0.2
                    c2 = -0.008
                    d2 = 350
                    
                    acc_2_reso = exponential_function_2(np.array([temp_reso]), a2, b2, c2, d2)
                    acc_2_reso = acc_2_reso[0]
            
            acc = acc_2_fps * acc_2_reso  # 将分辨率下的精度与帧率下的精度相乘，得到最终预测的精度
            

            if acc<0:
                acc = 0
            
            return acc
        
        else:
            return 1
    
    