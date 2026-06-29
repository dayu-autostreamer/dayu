import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier

# 随机森林多分类器
class MultiLabelTrainer:
    def __init__(self):
        self.models = {}  

    # 训练的时候给定X和y，以及对应分类器的名字
    def train(self, X, y, name):
        model = OneVsRestClassifier(
            RandomForestClassifier(
                n_estimators=50,       # 100→50，数据量小，50足够，避免过拟合
                max_depth=5,           # 限制树深，防止记忆训练数据
                min_samples_leaf=3,    # 叶子节点最少3条，2000条数据很容易触达
                min_samples_split=5,   # 分裂需要至少5条
                n_jobs=-1,
                random_state=42
            )
        )
        model.fit(X, y)

        self.models[name] = {
            "model": model,
            "get_proba": lambda x: model.predict_proba(x)
        }
        # print(f"[已保存到内存] {name}")
        return name
    
    # 直接获取预测结果和预测的概率
    def get_pred_proba(self, x, name):


        # 测试正确性
        input_x = [x]
        
        data = self.load(name)
        if data == None:
            return None
        
        get_proba = data["get_proba"]
        output_pred_proba = get_proba(input_x)
        pred_proba = [] 
        for proba in output_pred_proba[0]:
            pred_proba.append(float(proba))
        return pred_proba

    # 使用的时候直接获取模型
    def load(self, name):
        if name not in self.models:
            return None 
        return self.models[name]

    # 重要：获取所有的模型名称
    def list_models(self):
        return list(self.models.keys())

    # 有必要的时候删除
    def remove(self, name):
        if name in self.models:
            del self.models[name]
    