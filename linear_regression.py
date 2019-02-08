import numpy as np
import matplotlib.pyplot as plt


def mean_squared_error(y_prediction, y_true):
    return np.average((y_true - y_prediction) ** 2)


def root_mean_squared_error(y_prediction, y_true):
    return np.sqrt(mean_squared_error(y_true, y_prediction))


def read_csv_file(file):
    data = np.genfromtxt(file, delimiter=',', dtype=np.float, skip_header=True)
    return data


def train_test_split(x, y, train_size=0.8):
    train_size = int(len(x) * train_size)

    x_train = x[:train_size]
    y_train = y[:train_size]

    x_test = x[train_size:]
    y_test = y[train_size:]

    return x_train, x_test, y_train, y_test


def normal_equation(x, y):
    return np.matmul(np.matmul(np.linalg.inv(np.matmul(x.T, x)), x.T), y)


def neuron(x, y, learning_rate=0.001, nr_of_epochs=10000):
    weights = np.zeros(x.shape[1])

    for i in range(nr_of_epochs):
        output = np.dot(x, weights)
        errors = (y - output)

        gradient = np.dot(x.T, errors)

        weights += learning_rate * gradient

        # print("Train: MSE {} / RMSE {}".format(mean_squared_error(output, y_train), rmse(output, y_train)))

    return weights


def main():

    boston_housing_data = read_csv_file('mass_boston.csv')

    # Split features and output variable (y = median house value)
    features = boston_housing_data[:, 0:13]
    y = boston_housing_data[:, 13]

    # Normalise features by min-max
    min = features.min(axis=0)
    max = features.max(axis=0)
    norm_features = (features - min) / (max - min)

    # Covariance matrix to find features with highest covariance with output variable
    cov = np.corrcoef(boston_housing_data.T)

    plt.matshow(cov, cmap='seismic')
    plt.xticks(np.arange(14))
    plt.yticks(np.arange(14))
    plt.show()

    # Select features with highest covariance (5 = rm, 10 = ptratio, 12 = lstat)
    selected_features = norm_features[:, [5, 10, 12]]

    # Add bias
    x = np.c_[np.ones((len(selected_features))), selected_features]

    # Split data into train and test
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)

    print("Linear regression via normal equation")

    weights = normal_equation(x_train, y_train)

    predictions = np.dot(x_train, weights)

    print("Weights {}".format(weights))

    print("Train: MSE {} / RMSE {}"
          .format(mean_squared_error(predictions, y_train), root_mean_squared_error(predictions, y_train)))

    predictions = np.dot(x_test, weights)

    print("Test: MSE {} / RMSE {}"
          .format(mean_squared_error(predictions, y_test), root_mean_squared_error(predictions, y_test)))

    print("Linear regression via gradient descent")

    nr_of_epochs = 10000
    learning_rate = 0.001

    weights = neuron(x_train, y_train, learning_rate, nr_of_epochs)

    predictions = np.dot(x_train, weights)

    print("Weights {}".format(weights))

    print("Train: MSE {} / RMSE {}"
          .format(mean_squared_error(predictions, y_train), root_mean_squared_error(predictions, y_train)))

    predictions = np.dot(x_test, weights)

    print("Test: MSE {} / RMSE {}"
          .format(mean_squared_error(predictions, y_test), root_mean_squared_error(predictions, y_test)))

    # TODO Show effect if learning rate is too low or too high

    # TODO Add plots for RMSE and linear regression


if __name__ == '__main__':
    main()
