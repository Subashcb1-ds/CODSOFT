# Sales Prediction Using Multiple Linear Regression: Analyzing the Impact of Advertising on Sales

## Introduction

**Brief Description:**

- This project aims to predict sales based on advertising expenditures across three media channels: TV, Radio, and Newspapers.
- Using Multiple Linear Regression, the goal is to build an accurate forecasting model and analyze how each advertising channel affects sales.

**Objectives and Goals:**

- Develop a model to predict sales from advertising expenditure data.
- Analyze the impact of TV, Radio, and Newspaper advertising on sales.
- Build and evaluate a Multiple Linear Regression model.
- Provide actionable insights to optimize advertising strategies and budget allocation.

**Importance of Sales Prediction for Businesses:**

- Helps in optimizing advertising budgets.
- Informs decision-making on resource allocation.
- Enhances marketing strategies and maximizes revenue potential by targeting effective advertising channels.

## Data Description

The dataset includes four columns:

- `TV`: Advertising expenditure on TV
- `Radio`: Advertising expenditure on Radio
- `Newspaper`: Advertising expenditure on Newspapers
- `Sales`: Sales figures

## Data Preprocessing

- Checked for and confirmed no missing values.
- The dataset was ready for analysis without additional cleaning.

## Exploratory Data Analysis (EDA)

### Summary

- **Descriptive Statistics**: Provided basic statistics for each column, including mean, standard deviation, minimum, and maximum values.
- **Visualizations**:
  - **Histogram**: Showed the distribution of sales.
  - **Heatmap**: Displayed the correlation matrix of features.

### Output

- **Histogram**: Illustrates the distribution of sales.
- **Heatmap**: Revealed the correlations between advertising expenditures and sales.

## Analysis of Sales over Different Advertising Channels Using PowerBI

- Various PowerBI visualizations were used to analyze sales across advertising channels, including:
  - Distribution and impact of different advertising expenditures on sales.

## Model Building and Evaluation

### Model

- **Algorithm**: Multiple Linear Regression
- **Performance Metrics**:
  - **Mean Squared Error (MSE)**: Measures the average squared difference between actual and predicted sales.
  - **R-squared (R²)**: Indicates the proportion of variance in sales explained by the model.

### Results

- **Coefficients**:
  - **TV**: 0.05 (Small positive impact)
  - **Radio**: 0.10 (Largest positive impact)
  - **Newspaper**: 0.00 (No significant impact)
  - **Intercept**: 4.71 (Baseline sales with no advertising)

- **Model Evaluation**:
  - **MSE**: 2.91, indicating a good fit as it’s lower than the variance of actual sales.
  - **R-squared**: 0.91, suggesting that the model explains 91% of the variability in sales.

### Visualization

- **Actual vs. Predicted Sales Plot**: Showed how well the model’s predictions matched actual sales values.

## Discussion

### Coefficients and Their Impacts

- **TV**: Contributes a modest increase in sales.
- **Radio**: Has the most significant impact on sales, making it the most effective channel.
- **Newspaper**: Shows no meaningful impact on sales.

### Model’s Performance

- **Accuracy**: The low MSE and high R² value indicate that the model performs well in predicting sales.

## Data-Driven Decisions

- **Increase Investment in Radio**: Based on the model, reallocating more budget to Radio is likely to yield higher sales.
- **Monitor Other Channels**: Continue evaluating TV and Newspaper channels and consider exploring other factors affecting sales.

## Business Impact

- **Increased ROI**: Optimizing the advertising budget, especially towards Radio, is expected to enhance the return on investment.
- **Efficient Resource Allocation**: Focus on the most impactful advertising channel to maximize the efficiency of marketing expenditures.

## Conclusion

The analysis highlights that Radio advertising has the most significant effect on sales. Increasing investment in Radio advertising is expected to drive higher sales and maximize the return on marketing investments.
