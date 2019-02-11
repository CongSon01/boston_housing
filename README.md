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

In the first step, the script automatically reads, cleans and normalises the CSV file (i.e. `mass_boston.csv`). That is, 
all entries where MEDV is equal to 50.0 are removed, as this variable has been censored for higher values. Next, all 
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
Correlation with MEDV [-0.45   0.405 -0.6    0.075 -0.524  0.687 -0.493  0.369 -0.476 -0.572
 -0.519  0.365 -0.76   1.   ]
```

For the further evaluation we consider only features that have a positive or negative correlation that is greater than 
0.5:

Index | Name    | Correlation with MEDV
----- | --------|-------------
2     | INDUS   | -0.600
4     | NOX     | -0.524
5     | RM      | +0.687
9     | TAX     | -0.572
10    | PTRATIO | -0.519 
12    | LSTAT   | -0.760

Next, we add a bias of 1 to each entry in the dataset and split the dataset into training set (80% of the data) and 
an test set (20% of the data).

### LR by means of normal equation

Using the training set, we can directly compute the weights (i.e. regression coefficients) of the LR (i.e. one 
weight for the bias and six weights for the selected features) using normal equation. The trained weights can then 
be used to predict MEDV. That is, LR is evaluated by means of Mean Squared Error (MSE) and Root Mean Squared Error 
(RMSE) on both training and test set:

```
Linear regression via normal equation
Weights [ 18.448  -1.121   0.734  25.841  -4.089  -6.367 -13.639]
Train: MSE 16.4714 / RMSE 4.0585
Test: MSE 18.8056 / RMSE 4.3365
```

### LR by means of gradient descent

However, rather than computing the regression coefficients of LR directly, one could also train a single artificial 
neuron to approximate LR. That is, the weights of a neuron are trained by means of gradient descent to approximate 
the regression coefficients of LR.

In the following, we have two different scenarios, viz. one with a higher learning rate (i.e. 0.00100) and one with 
a smaller learning rate (i.e. 0.00001). In both scenarios, we train the neuron in 1000 epochs.

In case of higher learning rate, we can observe that the neuron is able to approximate LR quickly. That is, both 
the weights as well as the measures (i.e. MSE and RMSE) are almost identical when compared with LR by means of normal 
equation.

```
Linear regression via gradient descent (learning rate 0.00100)
Weights [ 18.647  -1.144   0.755  25.564  -4.077  -6.394 -13.788]
Train: MSE 16.4722 / RMSE 4.0586
Test: MSE 18.7043 / RMSE 4.3248
```

In case of lower learning rate, we can observe that the neuron is not able to converge so quickly.

```
Linear regression via gradient descent (learning rate 0.00001)
Weights [14.385  0.052  0.602 10.423 -0.091  4.264 -0.685]
Train: MSE 54.5219 / RMSE 7.3839
Test: MSE 69.2193 / RMSE 8.3198
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
