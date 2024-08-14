import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv(r"C:\Users\Subashcb1\Downloads\advertising - advertising.csv (1).csv")  

# Features and target variable
X = data[['TV', 'Radio', 'Newspaper']]
y = data['Sales']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Extract and print coefficients
coefficients = model.coef_
intercept = model.intercept_
print("Coefficients:")
print(f'TV: {coefficients[0]:.2f}')
print(f'Radio: {coefficients[1]:.2f}')
print(f'Newspaper: {coefficients[2]:.2f}')
print(f'Intercept: {intercept:.2f}')

# Interpret coefficients
# Radio has the highest impact on sales (0.10), followed by TV (0.05). Newspaper has no significant impact (0.00).

# Predict sales on the test set
y_pred = model.predict(X_test)

# Calculate and print metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse:.2f}')
print(f'R-squared: {r2:.2f}')

# Plot actual vs predicted sales with a regression line
plt.scatter(y_test, y_pred, label='Actual vs Predicted')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2, label='y=x')
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual vs Predicted Sales')
plt.legend()
plt.show()

# Function to predict sales based on user input
def predict_sales(tv, radio, newspaper):
    user_data = pd.DataFrame({'TV': [tv], 'Radio': [radio], 'Newspaper': [newspaper]})
    return model.predict(user_data)[0]

# Get user input and predict sales
tv_input = float(input("Enter TV advertising budget (in thousands): "))
radio_input = float(input("Enter Radio advertising budget (in thousands): "))
newspaper_input = float(input("Enter Newspaper advertising budget (in thousands): "))

predicted_sales = predict_sales(tv_input, radio_input, newspaper_input)
print(f'Predicted Sales: {predicted_sales:.2f}')
