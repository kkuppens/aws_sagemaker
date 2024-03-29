# Welcome to the Project page part of Udacity AWS Machine Learning Engineer Nanodegree

## Project: Image Classification for Vehicle Detection and Routing Optimization

In this project, I created an image classification model for Scones Unlimited, a delivery service company. 
The goal was to automatically detect the type of vehicle delivery drivers use, enabling the company to route 
them efficiently to the correct loading bay and orders. By assigning bicycle-riding delivery professionals to 
nearby orders and providing motorcyclists with orders that are farther away, Scones Unlimited aimed to optimize 
its operations.

As a Machine Learning Engineer, my objective was to deliver a scalable and reliable model. Once deployed, 
the model needed to be capable of handling varying demand while also implementing safeguards to monitor and 
manage performance drift or degradation.

Throughout the project, I utilized AWS SageMaker to develop an image classification model capable of 
distinguishing between bicycles and motorcycles. Deployment involved the use of AWS Lambda functions to 
create supporting services, and AWS Step Functions to orchestrate the model and services into an 
event-driven application. 


Project Steps Overview:

1. Data staging
2. Model training and deployment
3. Integration of Lambdas and Step Function workflow
4. Testing and evaluation
5. Optional challenge
6. Cleanup of cloud resources

Project submission artifacts:
- starter.ipynb, completed solution in notebook with key results displayed
- Lambda.py, all 3 lambda functions copied into one python file  
- MyStateMachine-yzv1v5pi8.asl.json, Stepfunction exported to json
- MyStateMachine-yzv1v5pi8-stepfunction-screenshot.png, Screencaps of working Step Function