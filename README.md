# Boston Housing
### Linear Regression
Linear Regression is a machine learning algorithm based on supervised learning. It performs a regression task. Regression models a target prediction value based on independent variables. It is mostly used for finding out the relationship between variables and forecasting.

### About the Dataset
We will use a housing dataset which contains information about different houses in Boston. This dataset is available in the _scikit-learn_ library. There are 506 samples and 13 feature variables in the dataset. The objective is to predict the value of the prices of the houses using the given features.

### Required Libraries
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_boston
```

### Exploratory Analysis
We perform exploratory analysis to derive useful information from the dataset. This step is useful as the quality of the dataset directly affects the ability of the model to learn.
```python
# Loading the dataset
boston_dataset = load_boston()

# Check values of boston_dataset
boston_dataset.keys()
```

When we print the above code, it returns
```
dict_keys(['data', 'target', 'feature_names', 'DESCR'])
```

Here,
1. _data_: contains the information for various houses
2. _target_: prices of the house
3. _feature_names_: names of the features
4. _DESCR_: describes the dataset

The description of all the features is given below:
```
CRIM: Per capita crime rate by town
ZN: Proportion of residential land zoned for lots over 25,000 sq. ft
INDUS: Proportion of non-retail business acres per town
CHAS: Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
NOX: Nitric oxide concentration (parts per 10 million)
RM: Average number of rooms per dwelling
AGE: Proportion of owner-occupied units built prior to 1940
DIS: Weighted distances to five Boston employment centers
RAD: Index of accessibility to radial highways
TAX: Full-value property tax rate per $10,000
PTRATIO: Pupil-teacher ratio by town
B: 1000(Bk — 0.63)², where Bk is the proportion of [people of African American descent] by town
LSTAT: Percentage of lower status of the population
MEDV: Median value of owner-occupied homes in $1000s
```

Now, we will use some visualizations to understand the relationship of the target variable with other features. We will use _displot_ function from the _seaborn_ library.
```python
sns.set(rc={
    'figure.figsize': (11.7, 8.27)
})
sns.distplot(boston['MEDV'], bins=30)
plt.show()
```

We see that the values of MEDV are distributed normally with few outliers.

<img src="contents/exploratory_medv.png" alt="MEDV Output" width="827" height="550">

Now, we will create a correlation matrix using _corr_ function from the pandas dataframe library. We will use heat-map function from the seaborn library to plot the correlation matrix.
```python
corr_matrix = boston.corr().round(2)
sns.heatmap(data=corr_matrix, annot=True)
plt.show()
```

<img src="contents/corr_matrix.png" alt="Correlation Matrix" width="827" height="550">
