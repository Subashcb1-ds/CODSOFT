# Iris Flower Species Classification Using K-Nearest Neighbors Algorithm

## Introduction

**Description:**
- This project aims to develop a machine learning model using the K-Nearest Neighbors (KNN) algorithm to classify iris flowers into three species: Setosa, Versicolor, and Virginica.
- Classification is based on sepal and petal measurements.

**Objective:**
- Predict the species of an iris flower based on sepal length, sepal width, petal length, and petal width using the KNN algorithm.

## Data Description

The Iris dataset includes 150 samples with four features:
- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

Each sample is labeled with one of three species:
- Setosa
- Versicolor
- Virginica

## Data Preprocessing

- Checked for missing values and found none.
- No data cleaning was needed as the dataset was already prepared.

## Exploratory Data Analysis (EDA)

- Basic statistics and histograms were used to summarize and visualize the dataset.
- Pair plots and a correlation heatmap provided insights into feature relationships and species distribution.

### EDA Results
- **Histograms**: Showed the distribution of features across the dataset.
- **Pair Plots**: Illustrated the relationships between features and species.
- **Heatmap**: Displayed the correlation between features.

## Model Building and Evaluation

- **Model**: K-Nearest Neighbors (KNN) with `k=3`.
- **Performance**: Achieved 100% accuracy in classifying iris species.
- **Confusion Matrix**: All predictions were correct, indicating no errors.
- **Metrics**:
  - **Precision**: 100%
  - **Recall**: 100%
  - **F1-Score**: 100%

### Visualization of Results
- **Pair Plots**: Highlighted feature distributions by species.
- **Confusion Matrix Heatmap**: Confirmed perfect classification performance.

## Conclusion

The KNN model effectively classified all iris species with perfect accuracy, demonstrating its reliability for this classification task.
