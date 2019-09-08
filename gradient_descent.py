from math import isclose

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# x = np.random.random_integers(1000, 1700, 100)
# y = np.random.random_integers(1000, 1700, 100)

dataset = pd.read_csv("dataset\\linear_dataset.csv", sep=',', header=None).to_numpy()
x = dataset[..., 0:1]
y = dataset[..., 1:2]

num = len(x)
flag1, flag2 = 1, 1
# slope1, const1, slope2, const2 = 1, 0, 1, 0
slope, const = 1, 0
new_slope, new_const = 0, 0

# for k in range(1000):

while isclose(flag1, flag2, rel_tol=0.0005):
    diff_slope, diff_const = 0, 0

    for i in range(0, num):
        diff_slope += ((-2 / num) * (y[i] - (const + slope * x[i])))
    step_size_slope = diff_slope * 0.01
    new_slope = slope - step_size_slope

    for j in range(0, num):
        diff_const += ((-2 / num) * x[j] * (y[j] - (const + slope * x[j])))
    step_size_const = diff_const * 0.01
    new_const = const - step_size_const

    slope1 = new_slope
    const2 = new_const

    flag1 = step_size_slope
    flag2 = step_size_const

print("new_slope : {}, new_const : {}".format(str(new_slope), str(new_const)))

max_value = np.max(x)
min_value = np.min(x)

a = np.linspace(min_value, max_value, 100)
b = new_const + new_slope * a

plt.plot(a, b, color="#58b970", label="Regression line")
plt.scatter(x, y, c='#ef5423', label="Scatter points")
plt.xlabel("Independent")
plt.ylabel("Dependent")
plt.legend()
plt.show()

# Prediction for y

y_pred = []
for i in range(len(x)):
    pred = new_const + new_slope * x[i]
    y_pred.append(pred)

# Accuracy calculation

mse = np.sum((y_pred - y) ** 2)
rmse = np.sqrt(mse / num)
print("Mean squared error: {}\n".format(mse),
      "Root mean squared error: {}\n".format(rmse))

ssr = np.sum((y_pred - y) ** 2)
sst = np.sum((y - np.mean(y)) ** 2)
r2_score = 1 - (ssr / sst)
print("Square of residuals: {}\n".format(ssr),
      "Sum of squares: {}\n".format(sst),
      "R2 score or coeff of determination: {}".format(r2_score))
