import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv(r"C:\Users\Subashcb1\Downloads\advertising - advertising.csv (1).csv")

# Check for missing values
print("\nMissing values in each column:")
print(data.isnull().sum())

# Calculate and display basic descriptive statistics
print("\nDescriptive statistics for numeric columns:")
print(data.describe())

# Visualizations
# Histogram for distribution of Sales
plt.figure(figsize=(10, 6))
plt.hist(data['Sales'], bins=20, edgecolor='black')
plt.xlabel('Sales')
plt.ylabel('Frequency')
plt.title('Distribution of Sales')
plt.grid(True)
plt.tight_layout()  # Ensure labels are not cut off
plt.show()

# Heatmap of the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.tight_layout()  # Ensure labels are not cut off
plt.show()
