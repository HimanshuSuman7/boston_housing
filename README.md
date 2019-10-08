# Boston Housing
### Linear Regression
Linear Regression is a machine learning algorithm based on supervised learning. It performs a regression task. Regression models a target prediction value based on independent variables. It is mostly used for finding out the relationship between variables and forecasting.

### About the Dataset
I have used housing dataset from Scikit - Learn library which contains information about different houses in Boston. There are 506 samples and 13 feature variables in the dataset. The objective is to predict the value of the prices of the houses using the given features.

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
ZN: Proportion of residential land zoned for lots over 25, 000 sq. ft
INDUS: Proportion of non - retail business acres per town
CHAS: Charles River dummy variable(=1 if tract bounds river; 0 otherwise)
NOX: Nitric oxide concentration(parts per 10 million)
RM: Average number of rooms per dwelling
AGE: Proportion of owner - occupied units built prior to 1940
DIS: Weighted distances to five Boston employment centers
RAD: Index of accessibility to radial highways
TAX: Full - value property tax rate per $10, 000
PTRATIO: Pupil - teacher ratio by town
B: 1000(Bk — 0.63)², where Bk is the proportion of[people of African American descent] by town
LSTAT: Percentage of lower status of the population
MEDV: Median value of owner - occupied homes in $1000s
```

### Required Imports
```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
```

_boston_hosusing.py_ file contains the complete code for your reference.
