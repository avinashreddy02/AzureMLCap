*NOTE:* This file is a template that you can use to create the README for your project. The *TODO* comments below will highlight the information you should be sure to include.

# House Sales in King County USA

In This Project i have Considered a Regression problem, it's a process of predicting a continuous value instead of discrete value given an input data, i have used two different techniques to train the model , one is using the AutoML which will trigger the training on multiple different models and second one using Hyperdrive configuration with single model

## Project Set Up and Installation
*OPTIONAL:* If your project has any special installation steps, this is where you should put it. To turn this project into a professional portfolio project, you are encouraged to explain how to set up this project in AzureML.

## Dataset

### Overview
In this project i have considered housing sales data set of king county, i have found this data set on kaggle , this data contains homes sold between may 2014 to may 2015, it requires minimal data cleaning and has an understandable list of variables, this enables me to focus more on required configuration to work with AzureML 

### Task
My Objective is to build a prediction model that predicts the housing prices from the set of given house features like , number of bedrooms, number of bathrooms , i will be perfomring this using Regression Task 

### Access
i have downloaded the housing sale dataset from kaggle first and uploaded the csv file to datastore , once that dataset is available in Azure i have used the below code to access the data from the datastore 

## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
