import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv(r"titanic_modified_encoded.csv")

# Display basic information about the dataset
print("Dataset Information:")
print(data.info())
print("\n")

# Display first few rows of the dataset
print("First 5 rows of the dataset:")
print(data.head())
print("\n")

# Summary statistics
print("Summary Statistics:")
print(data.describe(include='all'))
print("\n")

# Set up the matplotlib figure
fig, axes = plt.subplots(3, 2, figsize=(15, 15))
fig.suptitle('Titanic Dataset EDA', fontsize=16)

# Distribution of Survived
sns.countplot(x='Survived', data=data, ax=axes[0, 0])
axes[0, 0].set_title('Distribution of Survived')

# Distribution of Pclass
sns.countplot(x='Pclass', data=data, ax=axes[0, 1])
axes[0, 1].set_title('Distribution of Passenger Class')

# Distribution of Sex
sns.countplot(x='Sex', data=data, ax=axes[1, 0])
axes[1, 0].set_title('Distribution of Sex')

# Distribution of Age
sns.histplot(data['Age'], kde=True, bins=30, ax=axes[1, 1])
axes[1, 1].set_title('Distribution of Age')

# Distribution of Fare
sns.histplot(data['Fare'], kde=True, bins=30, ax=axes[2, 0])
axes[2, 0].set_title('Distribution of Fare')

# Distribution of Embarked
sns.countplot(x='Embarked', data=data, ax=axes[2, 1])
axes[2, 1].set_title('Distribution of Embarked')

# Show the plot
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# Set up another figure for relationship plots
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle('Relationships with Survived', fontsize=16)

# Relationship between Pclass and Survived
sns.barplot(x='Pclass', y='Survived', data=data, ax=axes[0])
axes[0].set_title('Survival Rate by Passenger Class')

# Relationship between Sex and Survived
sns.barplot(x='Sex', y='Survived', data=data, ax=axes[1])
axes[1].set_title('Survival Rate by Sex')

# Show the plot
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
