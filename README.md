# Linear regression on the Boston housing dataset

The [Boston housing dataset](https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html) is based on 506 entries 
consisting of 14 different variables each. In the following example, the median house value (medv) is predicted based on
the remaining 13 variables by means of Linear Regression (LR). That is, LR is first directly computed by means of normal 
equation. Second, LR is found approximative via Gradient Descent (GD).

## Tutorial

Clone the repository and change to the project directory (e.g. `cd ~/Projects/boston_housing/`). Next, you can execute 
the python script with the following command:

```
python linear_regression.py 
```

In the first step, the script automatically reads and normalises the CSV file (i.e. `mass_boston.csv`). In order to 
find the *best* features for the LR, we first conduct a covariance analysis of the following variables:


Index | Name    | Description
----- | --------|--------------
0     | CRIM    |
1     | ZN      |
2     | INDUS   |
3     | CHAS    |
4     | NOX     |
5     | RM      |
6     | AGE     |
7     | DIS     |
8     | RAD     |
9     | TAX     |
10    | PTRATIO |
11    | B       |
12    | LSTAT   |
13    | MEDV    |


![Covariance Matrix](imgs/covariance_matrix.png)

![Covariance Matrix](imgs/rmse_epoch.png)




## Requirements

The following code has been tested using [Python](https://www.python.org/) 3.7.1, [NumPy](https://www.numpy.org/) 1.15.4, 
and [Matplotlib](https://matplotlib.org/) 3.0.2 using [Anaconda](https://www.anaconda.com/distribution/#download-section).
