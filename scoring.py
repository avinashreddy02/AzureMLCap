import json
import numpy as np
import os
import joblib
from azureml.core.model import Model



def init():
    global model
    model_root = Model.get_model_path('house_sales1')
    print(model_root)
    model_path = os.path.join(model_root, 'model.pkl')
    print(model_path)
    model = joblib.load(model_path)

def run(data):
    try:
        print(data)
        data = np.array(json.loads(data))
        result = model.predict(data)
        # You can return any data type, as long as it is JSON serializable.
        return result.tolist()
    except Exception as e:
        error = str(e)
        return error