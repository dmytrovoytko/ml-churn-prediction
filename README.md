# ML project Predicting Customer Churn for a Gym

Midterm project for DataTalks.Club Machine Learning ZoomCamp`24:

![ML project Churn prediction](/EDA/feature-importance.png)

Project can be tested and deployed in **GitHub CodeSpaces** (the easiest option, and free), cloud virtual machine (AWS, Azure, GCP), or just locally.
For GitHub CodeSpace option you don't need to use anything extra at all - just your favorite web browser + GitHub account is totally enough.

## Problem statement

Subscription-based businesses are all around us - from classics like telecom to cloud services, Netflix and ChatGPT. Customer retention is a critical factor for the long-term success of such companies. Acquiring new customers is often significantly more costly - "from 5 to 25 times more expensive than retaining an existing one." (Harvard Business Review) - "It makes sense: you don’t have to spend time and resources going out and finding a new client — you just have to keep the one you have happy". Therefore increase of customer retention can lead to significant growth of profits over time. 
Businesses need accurately predict customer churn, so they can proactively implement targeted retention strategies to reduce customer attrition and increase revenue.
I decided to use Machine Learning to predict customer churn, and chose a [Gym customers features and churn dataset from Kaggle](https://www.kaggle.com/datasets/adrianvinueza/gym-customers-features-and-churn).

## 🎯 Goals

This is my Midterm project in [Machine Learning ZoomCamp](https://github.com/DataTalksClub/machine-learning-zoomcamp)'24.

**The main goal** is straight-forward: build an end-to-end Machine Learning project:
- choose interesting dataset
- load data, conduct exploratory data analysis (EDA), clean it
- train & test ML model(s)
- deploy the model (as a web service) using containerization

## 🔢 Dataset

[CSV file](/data/gym_churn_us.csv) includes 4000 records with 14 columns.

**Structure**: 
- customer features: gender, near_location, partner, promo_friends, phone, group_visits, age
- financial: contract_period, avg_additional_charges_total, month_to_end_contract, lifetime, avg_class_frequency_total, avg_class_frequency_current_month
- labels: churn - 1 if customer unsubscribed

You can find more details in the beginning of [model-training-log.txt](/model-training-log.txt) and screenshots below.

## 📊 EDA

Dataset is well prepared, without duplicates and null values.
You can explore detailed information in [Jupyter notebook](/churn-prediction-3.ipynb)

Overview distribution of all features:
![Overview distribution of all features](/EDA/distribution-high-view.jpg)

Churn rate distribution:
![Churn rate distribution](/EDA/churn-distribution-overview.png)

Key features distribution:
![Key features distribution](/EDA/feature-importance.png)

Correlation matrix:
![Correlation matrix](/EDA/correlation-heatmap.png)

## 🎛 Model training

I started with 3 classifiers used for prediction - linear regression and 2 tree-based:
- LogisticRegression
- RandomForestClassifier
- AdaBoostClassifier

Also I experimented with hyperparameter tuning to improve metrics.

Comparison of modeling in jupyter notebook:
![Models comparison](/EDA/model-comparison.png)

## Python scripts for data pre-processing and training

- [preprocess.py](/prediction_service/preprocess.py)
- [train_model.py](/prediction_service/train_model.py)

`train_model.py` includes a more advanced hyperparameter tuning for all models (even 4, + DecisionTreeClassifier)
I used GridSearchCV and measured time for training each classifier.
You can find results in [model-training-log.txt](/model-training-log.txt)


## 🚀 Instructions to reproduce

- [Setup environment](#hammer_and_wrench-setup-environment)
- [Train model](#train-model)
- [Test prediction service](#test-prediction-service)
- [Deployment](#deployment)


### :hammer_and_wrench: Setup environment

1. **Fork this repo on GitHub**. Or use `git clone https://github.com/dmytrovoytko/ml-churn-prediction.git` command to clone it locally, then `ml-churn-prediction`.
2. Create GitHub CodeSpace from the repo.
3. **Start CodeSpace**
4. **Go to the app directory** `ml-churn-prediction`
5. The app works in docker container, **you don't need to install packages locally to test it**.
6. Only if you want to develop the project locally, you can run `pip install -r requirements.txt` (project tested on python 3.11/3.12).

### Train model

1. **Run `bash deploy.sh` to build and start app container**. The dataset is quite small, required libraries too, so it should be ready to serve quickly enough. When new log messages stop appearing, press enter to return to a command line.

![docker-compose up](/screenshots/docker-compose-00.png)

When you see these messages app is ready

![docker-compose up](/screenshots/docker-compose-01.png)

2. To reproduce training process run `bash train.sh` which starts model training in docker container. If you run it locally, execute `python train_model.py`. 

As a result you will see log similar to [model-training-log.txt](/model-training-log.txt)

### Test prediction service

1. **Run `bash test-api.sh` to execute test calls to prediction web service**. If you run it locally, execute `python test-api.py`. 

![Testing prediction service in dockerl](/screenshots/prediction-service-test-1.png)

![Testing prediction service in dockerl](/screenshots/prediction-service-test-dataset-1.png)


### Deployment

As application is fully containerized, it can be deployed on any virtual machine (AWS, Azure, GCP).

