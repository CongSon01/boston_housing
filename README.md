# Linear regression on the Boston housing dataset

The [Boston housing dataset](https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html) is based on 506 entries 
consisting of 14 different variables each. In the following tutorial, the median house value (MEDV) is predicted based on
the remaining 13 variables by means of Linear Regression (LR). That is, LR is first directly computed by means of normal 
equation. Second, LR is found approximative via Gradient Descent (GD).

## Tutorial

Clone the repository and change to the project directory (e.g. `cd ~/Projects/boston_housing/`). Next, you can execute 
the python script by the following command:

```
python linear_regression.py 
```

### Preprocessing 

In the first step, the script automatically reads and normalises the CSV file (i.e. `mass_boston.csv`). That is, all 
features (0 to 12 in the table below) are scaled into [0,1] by means of a min-max normalisation.

Index | Name    | Description
----- | --------|--------------
0     | CRIM    | Per capita crime rate by town
1     | ZN      | Proportion of residential land zoned for lots over 25,000 sq.ft.
2     | INDUS   | Proportion of non-retail business acres per town
3     | CHAS    | Charles River dummy variable (1 if tract bounds river; 0 otherwise)
4     | NOX     | Nitric oxides concentration (parts per 10 million)
5     | RM      | Average number of rooms per dwelling
6     | AGE     | Proportion of owner-occupied units built prior to 1940
7     | DIS     | Weighted distances to five Boston employment centres
8     | RAD     | Index of accessibility to radial highways
9     | TAX     | Full-value property-tax rate per $10,000
10    | PTRATIO | Pupil-teacher ratio by town 
11    | B       | 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
12    | LSTAT   | % lower status of the population
13    | MEDV    | Median value of owner-occupied homes in $1000's

To find the most meaningful features for LR, we first conduct a covariance analysis of all features:

![Covariance Matrix](imgs/covariance_matrix.png)

```
Correlation with MEDV [-0.388  0.36  -0.484  0.175 -0.427  0.695 -0.377  0.25  -0.382 -0.469
 -0.508  0.333 -0.738  1.   ]
```

It turns out, that RM (feature 5) has a strong positive correlation with MEDV (feature 13), while PTRATIO (feature 10) 
and LSTAT (feature 12) have a strong negative correlation with MEDV. Hence, for the further evaluation we consider 
these three features only.

Next, we add a bias (1) to each row in the dataset and split the dataset into training set (80% of the data) and 
an test set (20% of the data).

### LR by means of normal equation

Using the training set, we can directly compute the weights (i.e. regression coefficients) of the LR (i.e. one 
weight for the bias and three weights for the selected features) using normal equation. The trained weights can then 
be used to predict MEDV. That is, LR is evaluated by means of Mean Squared Error (MSE) and Root Mean Squared Error 
(RMSE) on both training and test set:

```
Linear regression via normal equation
Weights [ 17.183  29.569  -6.752 -18.133]
Train: MSE 28.0405 / RMSE 5.2953
Test: MSE 26.9861 / RMSE 5.1948
```

### LR by means of gradient descent

However, rather than computing the regression coefficients of LR directly, one could also train a single artificial 
neuron to approximate LR. That is, the weights of a neuron are trained by means of gradient descent to approximate 
the regression coefficients of LR.

In the following, we have two different scenarios, viz. one with a higher learning rate (i.e. 0.00100) and one with 
a smaller learning rate (i.e. 0.00001). In both scenarios, we train the neuron by 1000 epochs.

In case of higher learning rate, we can observe that the neuron is able to approximate LR correctly. That is, both 
the weights as well as the measures (i.e. MSE and RMSE) are almost identical when compared with LR by means of normal 
equation.

```
Linear regression via gradient descent (learning rate 0.00100)
Weights [ 17.37   29.325  -6.787 -18.268]
Train: MSE 28.0412 / RMSE 5.2954
Test: MSE 26.7880 / RMSE 5.1757
```

In case of lower learning rate, we can observe that the neuron is not able to achieve the same measures and weights.

```
Linear regression via gradient descent (learning rate 0.00001)
Weights [15.388 11.707  4.218 -0.701]
Train: MSE 72.1460 / RMSE 8.4939
Test: MSE 90.8260 / RMSE 9.5303
```

This effect can also be observed by plotting the RMSE for each epoch during training (see figure below). That is, if 
the learning rate is high enough, the neuron or more precisely gradient descent is able to converge the weights quickly. 
As a result, we can observe that the RMSE of the neuron (highlighted in blue) converges to the optimal RMSE 
(highlighted in red) after few epochs. Note that the optimal RMSE represents LR by means of normal equation. However, 
if the learning rate is too low, gradient descent converges only very slowly. Similarly, the learning rate can also be 
too large. As a result, gradient descent would miss the optimal weight settings completely.

![RMSE_EPOCH](imgs/rmse_epoch.png)

## Requirements

The following code has been tested using [Python](https://www.python.org/) 3.7.1, [NumPy](https://www.numpy.org/) 1.15.4, 
and [Matplotlib](https://matplotlib.org/) 3.0.2 using [Anaconda](https://www.anaconda.com/distribution/#download-section).
