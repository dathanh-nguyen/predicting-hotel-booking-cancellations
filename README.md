# Predicting Hotel Booking Cancellations
## Project Overview
This project centers on the prediction of cancellations of hotel bookings using machine learning techniques. It utilizes a reduced version of the 'Hotel booking demand' dataset from [Kaggle](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand). This dataset has been reduced and split into two parts - the training dataset (`bookings_train.Rdata`) and the test dataset (`bookings_test_solutions.Rdata`). The training dataset comprises 24 035 bookings, and the training dataset comprises further 16 025 bookings. The dataset contains a total of 25 variables, including our target variable `is_cancelled`. The goal of the project was to reach at least 85% accuracy, while reaching the highest sensitivity possible. 

In this project, I was responsible for the model assessment set-up, the creation of the recipe and the development of the Random Forest model.

## EDA
An exploratory data analysis was first performed to identify potential issues with the dataset. At this stage, data cleaning was performed, the correct data types were assigned to the variables and feature engineering was done to reduce the number of features and get rid of correlated variables.

Since the dataset also has a large class imbalance in regards to the `is_cancelled`, I decided to deal with this issue by using downsampling. This was compared with a non-downsampled model, and the sensitivity increased considerably using the downsampling model, while the other metrics remained more or less the same. Numeric variables are standardized.

## Modelling
Multiple models were fitted and assessed:
- K-nearest Neighbours Classifier
- Regularized Logistic Regression
- Random Forest Classifier

Each of the models was tuned using a 10-fold cross validation. The best model was selected based on the performance on the test set, using the highest accuracy achieved. In the code, only Random Forest and the Regularized Logistic Regression Model are included, since KNN's tuning process is largely similar to the RF model.

## Final Results
The best performing model was the random forest model.

| Accuracy | Cohen’s Kappa | Sensitivity | Specificity |
|----------|---------------|-------------|-------------|
|   0.88   |     0.72      |     0.92    |     0.86    |

