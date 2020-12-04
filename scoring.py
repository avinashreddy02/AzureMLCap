import json
import numpy as np
import os
import joblib
from azureml.core.model import Model



def init():
    global model
    model_root = Model.get_model_path('house_sale')
    print(model_root)
    model_path = os.path.join(model_root, 'model.pkl')
    print(model_path)
    model = joblib.load(model_path)

def run(data):
    try:
        data = np.array(json.loads(data))
        result = model.predict(data)
        # You can return any data type, as long as it is JSON serializable.
        return result
    except Exception as e:
        error = str(e)
        return error