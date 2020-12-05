# Train and Deploy Models with Azure ML

## Table of Contents
- [Problem Statement](##problem)
- [Data Set](##dataset)
    - [Task](###task)
    - [Access](###access)
- [Set Up and Installation](##setup)
- [Automated ML](##automl)
    - [Result](##automl_result)
    - [Thoughts about Improvement](##automl_improve)
- [Hyperparameter Tuning](##hyperdrive)
    - [Result](##hyperdrive_result)
    - [Thoughts about Improvement](##hyperdrive_improve)
 - [Model Deployment](##deployment)
 - [Recreen Recording](##recording) 
 - [Standout Suggestions](##standout)

# House Sales in King County USA <a name="problem"></a>

In This Project i have Considered a Regression problem, it's a process of predicting a continuous value instead of discrete value given an input data, i have used two different techniques to train the model , one is using the AutoML which will trigger the training on multiple different models and second one using Hyperdrive configuration with single model

## Project Set Up and Installation
*OPTIONAL:* If your project has any special installation steps, this is where you should put it. To turn this project into a professional portfolio project, you are encouraged to explain how to set up this project in AzureML.

## Dataset <a name="dataset"></a>

### Overview
In this project i have considered housing sales data set of king county, i have found this data set on [kaggle](https://www.kaggle.com/harlfoxem/housesalesprediction) , this data contains homes sold between may 2014 to may 2015, it requires minimal data cleaning and has an understandable list of variables, this enables me to focus more on required configuration to work with AzureML 

### Task <a name="task"></a>
My Objective is to build a prediction model that predicts the housing prices from the set of given house features like , number of bedrooms, number of bathrooms , i will be perfomring this using Regression Task 

### Access
i have downloaded the housing sale dataset from kaggle first and uploaded the csv file to datastore , once that dataset is available in Azure i have used the below code to access the data from the datastore 

## Automated ML <a name="automl"></a>
While setting up the AutomL run we first need to define the automl configuration which has different parameters like task type whether it's regression or classification and label column, primary metric, and number of cross validations 

here are the automl settings i have selected for the automl run 
```
automl_settings = {
    "iteration_timeout_minutes": 10,
    "experiment_timeout_hours" : 0.3,
    "enable_early_stopping": True,
    "primary_metric" : 'normalized_mean_absolute_error',
    "featurization": 'auto',
    "verbosity": logging.INFO,
    "n_cross_validations": 5
}

# TODO: Put your automl config here
automl_config = AutoMLConfig(
    task = 'regression',
    debug_log = 'automl_reg_errors.log',
    training_data = x_train,
    label_column_name = "price",
    **automl_settings
)

```

next we submit the automl run which we can monitor using run widgets in notebook or also azure ML studio UI , below are the screen shots of the progress 
```
remote_run = experiment.submit(automl_config, show_output = True)

RunDetails(remote_run).show()
remote_run.get_status()
remote_run.wait_for_completion()
```
![Run Details1](runwidget_1.PNG)

### Results <a name="automl_result"></a>
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning <a name="hyperdrive"></a>
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results <a name="hyperdrive_result"></a> 
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment <a name="deployment"></a>
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording <a name="recording"></a>
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
