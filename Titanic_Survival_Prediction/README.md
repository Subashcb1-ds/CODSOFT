# Predicting Titanic Survival: Analyzing Passenger Data to Forecast Survival Outcomes

## Introduction

### Description

This project involves analyzing Titanic passenger data to build a predictive model for forecasting survival outcomes based on features like age, sex, and passenger class.

### Objective

To develop a machine learning model that predicts Titanic passenger survival based on various attributes, using data analysis and classification techniques.

## Data Description

### Overview

The dataset includes information on Titanic passengers and their survival status. It contains features such as:

- **PassengerId**: Unique identifier for each passenger.
- **Survived**: Survival status (0 = Not survived, 1 = Survived).
- **Pclass**: Passenger class (1 = First class, 2 = Second class, 3 = Third class).
- **Name**: Name of the passenger.
- **Sex**: Gender of the passenger (male or female).
- **Age**: Age of the passenger.
- **SibSp**: Number of siblings or spouses aboard.
- **Parch**: Number of parents or children aboard.
- **Ticket**: Ticket number.
- **Fare**: Fare paid by the passenger.
- **Cabin**: Cabin number (often missing).
- **Embarked**: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton).

### Additional Features

- **Title_Master**: Title "Master".
- **Title_Miss**: Title "Miss".
- **Title_Mr**: Title "Mr".
- **Title_Mrs**: Title "Mrs".
- **Title_Rare**: Less common titles.

## Data Preprocessing

Data preprocessing involves several steps to prepare the dataset for modeling:

1. **Loading the Data**: Read the dataset from a CSV file.
2. **Handling Missing Values**: Fill missing values in the `Age` column with the median age.
3. **Feature Extraction**: Extract and encode titles from the `Name` column.
4. **Encoding Categorical Variables**: Convert `Sex` and `Embarked` to numerical values.
5. **Saving the Preprocessed Data**: Save the cleaned data to a new CSV file.

The detailed preprocessing code is provided in a separate Python file: `preprocessing_titanic.py`.

## Python Code

For detailed data preprocessing, refer to the `preprocessing_titanic.py` file.

## Exploratory Data Analysis (EDA)

EDA involves analyzing the dataset to understand its structure, patterns, and relationships between features. The key steps include:

1. **Loading the Data**: Read the preprocessed dataset from a CSV file.
2. **Descriptive Statistics**: Calculate summary statistics for numerical features.
3. **Data Visualization**: Create plots to visualize distributions and relationships, such as histograms, boxplots, and correlation matrices.
4. **Identifying Trends and Patterns**: Analyze visualizations to uncover trends and patterns related to survival rates.

The detailed EDA code is provided in a separate Python file: `eda_titanic.py`.

## Python Code

For detailed exploratory data analysis, refer to the `eda_titanic.py` file.


## Model Training

Model training involves building and evaluating a machine learning model to predict survival outcomes. The key steps include:

1. **Loading the Data**: Read the preprocessed dataset from a CSV file.
2. **Splitting the Data**: Divide the dataset into training and test sets.
3. **Choosing a Model**: Select a classification algorithm (e.g., Logistic Regression, Random Forest).
4. **Training the Model**: Fit the model on the training data.
5. **Evaluating the Model**: Assess the model’s performance using metrics like accuracy, precision, recall, and F1-score.
6. **Saving the Model**: Save the trained model for future use.

The detailed model training code is provided in a separate Python file: `model_training_titanic.py`.

## Python Code

For detailed model training, refer to the `model_training_titanic.py` file.

