from sklearn.ensemble import RandomForestRegressor
import argparse
import os
import numpy as np
from sklearn.metrics import mean_absolute_error
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.workspace import Workspace
from azureml.core.dataset import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from azureml.core.run import Run

# azureml-core of version 1.0.72 or higher is required
# azureml-dataprep[pandas] of version 1.1.34 or higher is required
from azureml.core import Workspace, Dataset


def clean_data(df):
    df.drop("id",inplace = True,axis = 1)
    df.drop("zipcode",inplace = True,axis = 1)
    df.drop("date",inplace = True,axis = 1)
    y_df = df.pop("price")
    
    

    
    
    return df,y_df


def main():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--n_estimators',type = int, default = 100, help = "number of estimators")
    parser.add_argument('--max_depth',type = int,default = 8, help = "depth of the tree")
    parser.add_argument('--min_samples_split',type = int, default = 2, help = "the minimum number of samples required to split an internal node")
    
    args = parser.parse_args()
    
    run.log("Number of Estimators", np.int(args.n_estimators))
    run.log("maximum depth", np.int(args.max_depth))
    run.log("minimum samples split",np.int(args.min_samples_split))
    
    model = RandomForestRegressor(n_estimators = args.n_estimators,max_depth = args.max_depth,
                                 min_samples_split = args.min_samples_split).fit(x_train,y_train)
    accuracy = model.score(x_test,y_test)
    run.log("accuracy",np.float(accuracy))
    
    y_pred = model.predict(x_test)
    mae = mean_absolute_error(y_test,y_pred)
    run.log("MAE",np.float(mae))
    
    os.makedirs('outputs',exist_ok = True)
    joblib.dump(model,'outputs/model.joblib')
    


subscription_id = '7395406a-64a8-4774-b0c2-0d5dafb2a8ce'
resource_group = 'aml-quickstarts-129197'
workspace_name = 'quick-starts-ws-129197'

workspace = Workspace(subscription_id, resource_group, workspace_name)

dataset = Dataset.get_by_name(workspace, name='house_sales')
ds = dataset.to_pandas_dataframe()
    
x, y = clean_data(ds)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3, random_state = 42)

run = Run.get_context()

if __name__ == '__main__':
    
    main()
    



