import numpy as np

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.multiclass import OneVsRestClassifier

    SKLEARN_IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    RandomForestClassifier = None
    OneVsRestClassifier = None
    SKLEARN_IMPORT_ERROR = exc


class MultiLabelTrainer:
    def __init__(self):
        self.models = {}

    def train(self, X, y, name):
        if SKLEARN_IMPORT_ERROR is not None:
            raise ModuleNotFoundError(
                "scikit-learn is required to train steady scheduler classifiers."
            ) from SKLEARN_IMPORT_ERROR

        model = OneVsRestClassifier(
            RandomForestClassifier(
                n_estimators=50,
                max_depth=5,
                min_samples_leaf=3,
                min_samples_split=5,
                n_jobs=-1,
                random_state=42
            )
        )
        model.fit(X, y)

        self.models[name] = {
            "model": model,
            "get_proba": lambda x: model.predict_proba(x)
        }
        return name

    def get_pred_proba(self, x, name):

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

    def load(self, name):
        if name not in self.models:
            return None
        return self.models[name]

    def list_models(self):
        return list(self.models.keys())

    def remove(self, name):
        if name in self.models:
            del self.models[name]
