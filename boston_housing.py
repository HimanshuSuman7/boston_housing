import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_boston

pd.set_option('display.width', 300)
pd.set_option('display.max_columns', 30)

boston_dataset = load_boston()
# print(boston_dataset.keys())

boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
# print(boston.head())

boston['MEDV'] = boston_dataset.target
# print(boston.head())

# print(boston.isnull().sum())

# Plot distribution of the target variable
sns.set(rc={
    'figure.figsize': (11.7, 8.27)
})
sns.distplot(boston['MEDV'], bins=30)
plt.show()

# Correlation matrix
corr_matrix = boston.corr().round(2)
sns.heatmap(data=corr_matrix, annot=True)
plt.show()

# Observations
plt.scatter(boston['LSTAT'], boston['MEDV'], marker='o')
plt.title("LSTAT")
plt.xlabel('LSTAT')
plt.ylabel('MEDV')
plt.show()

plt.scatter(boston['RM'], boston['MEDV'], marker='o')
plt.title("RM")
plt.xlabel('RM')
plt.ylabel('MEDV')
plt.show()

# Applying
