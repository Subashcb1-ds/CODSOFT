import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file
data = pd.read_csv(r"IRIS.csv")

# Check for missing values
print("\nMissing values in each column:\n", data.isnull().sum())

# Display basic descriptive statistics
print("\nDescriptive Statistics:\n", data.describe())

# Plot histograms for each feature
data.hist(bins=20, figsize=(12, 8))
plt.suptitle('Histograms of Features')
plt.show()
