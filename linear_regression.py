import numpy as np
import matplotlib.pyplot as plt


def read_csv_file(file):
    # Read data from CSV file
    data = np.genfromtxt(file, delimiter=',', dtype=np.float, skip_header=True)
    return data


def mse(y_prediction, y_true):
    # Calculate mean square error (mse)
    return np.average((y_true - y_prediction) ** 2)


def rmse(y_prediction, y_true):
    # Calculate root mean square error (mse)
    return np.sqrt(mse(y_prediction, y_true))


def train_test_split(x, y, train_size=0.8):
    # Split data into training and test set
    train_size = int(len(x) * train_size)

    x_train = x[:train_size]
    y_train = y[:train_size]

    x_test = x[train_size:]
    y_test = y[train_size:]

    return x_train, x_test, y_train, y_test


def normal_equation(x, y):
    # Linear regression via normal equation
    return np.matmul(np.matmul(np.linalg.inv(np.matmul(x.T, x)), x.T), y)


def gradient_descent(x, y, learning_rate=0.001, nr_of_epochs=10000):
    # Linear regression via gradient descent
    weights = np.zeros(x.shape[1])
    error_mse = []
    error_rmse = []

    for i in range(nr_of_epochs):

        # Calculate output (= prediction) of activation function
        prediction = predict(x, weights)

        # Calculate error between prediction and output variable
        errors = (y - prediction)

        # Calculate gradient descent to update weights
        gradient = np.dot(x.T, errors)
        weights += learning_rate * gradient

        error_mse.append(mse(prediction, y))
        error_rmse.append(rmse(prediction, y))

    return weights, error_mse, error_rmse


def predict(x, weights):
    # Predict output (i.e. MEDV) based on given features (x) and weights
    return np.dot(x, weights)


def print_subplot(axs, error_rmse, rmse_normal_equation, learning_rate):
    axs.plot(error_rmse)
    axs.set_title('Learning rate {:0.5f}'.format(learning_rate))
    axs.set_xlabel('Nr. of Epochs')
    axs.set_ylabel('RMSE')
    axs.axhline(rmse_normal_equation, color='red', linestyle='dotted')
    axs.set_ylim([0, 30])


def main():

    boston_housing_data = read_csv_file('mass_boston.csv')

    # Change printing precision of matrices
    np.set_printoptions(precision=3)

    # Remove entries where MEDV is equal to 50.0, as this variable has been censored for higher values
    boston_housing_data = boston_housing_data[boston_housing_data[:, 13] != 50.0]

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

    print("Correlation with MEDV {}\n".format(cov[13]))

    # Select features with highest covariance (2 = INDUS, 4 = NOX, 5 = RM, 9 = TAX, 10 = PTRATIO, 12 = LSTAT)
    selected_features = norm_features[:, [2, 4, 5, 9, 10, 12]]

    # Add bias
    x = np.c_[np.ones((len(selected_features))), selected_features]

    # Split data into train and test
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)

    print("Linear regression via normal equation")

    weights = normal_equation(x_train, y_train)
    predictions = predict(x_train, weights)

    print("Weights {}".format(weights))
    rmse_normal_equation = rmse(predictions, y_train)

    print("Train: MSE {:2.4f} / RMSE {:1.4f}".format(mse(predictions, y_train), rmse(predictions, y_train)))

    predictions = predict(x_test, weights)

    print("Test: MSE {:2.4f} / RMSE {:1.4f}\n".format(mse(predictions, y_test), rmse(predictions, y_test)))

    fig, axs = plt.subplots(2)

    # Linear regression via gradient descent with optimal learning rate
    nr_of_epochs = 1000
    learning_rate = 0.001

    print("Linear regression via gradient descent (learning rate {:0.5f})".format(learning_rate))

    weights, error_mse, error_rmse = gradient_descent(x_train, y_train, learning_rate, nr_of_epochs)
    predictions = predict(x_train, weights)

    print("Weights {}".format(weights))

    print("Train: MSE {:2.4f} / RMSE {:1.4f}".format(mse(predictions, y_train), rmse(predictions, y_train)))

    predictions = predict(x_test, weights)

    print("Test: MSE {:2.4f} / RMSE {:1.4f}\n".format(mse(predictions, y_test), rmse(predictions, y_test)))

    print_subplot(axs[0], error_rmse, rmse_normal_equation, learning_rate)

    # Linear regression via gradient descent with small learning rate
    learning_rate = 0.00001

    print("Linear regression via gradient descent (learning rate {:0.5f})".format(learning_rate))

    weights, error_mse, error_rmse = gradient_descent(x_train, y_train, learning_rate, nr_of_epochs)
    predictions = predict(x_train, weights)

    print("Weights {}".format(weights))

    print("Train: MSE {:2.4f} / RMSE {:1.4f}".format(mse(predictions, y_train), rmse(predictions, y_train)))

    predictions = predict(x_test, weights)

    print("Test: MSE {:2.4f} / RMSE {:1.4f}\n".format(mse(predictions, y_test), rmse(predictions, y_test)))

    print_subplot(axs[1], error_rmse, rmse_normal_equation, learning_rate)

    plt.show()


if __name__ == '__main__':
    main()
