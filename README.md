# ML project Predicting Customer Churn for a Gym

Midterm project for DataTalks.Club Machine Learning ZoomCamp`24:

![ML project Churn prediction](/EDA/model-comparison.png)

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

## EDA

Dataset is well prepared - without duplicates and null values.
You can explore detailed information in [notebook](/churn-prediction-3.ipynb)

Overview distribution of all features:
![Overview distribution of all features](/EDA/distribution-high-view.jpg)

Churn rate distribution:
![Churn rate distribution](/EDA/churn-distribution-overview.png)

Key features distribution:
![Key features distribution](/EDA/feature-importance.png)

Correlation matrix:
![Correlation matrix](/EDA/correlation-heatmap.png)

## Model training

I started with 3 classifiers used for prediction - linear regression and 2 tree-based:
- LogisticRegression
- RandomForestClassifier
- AdaBoostClassifier

Also I experimented with hyperparameter tuning to improve metrics.

Comparison of modeling in jupyter notebook:
![Models comparison](/EDA/model-comparison.png)

