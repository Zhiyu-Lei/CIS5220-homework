import numpy as np


class LinearRegression:
    """
    A linear regression model that uses closed form solution to fit the model.
    """

    w: np.ndarray
    b: float

    def __init__(self):
        self.w = None
        self.b = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the model using closed form solution

        Arguments:
            X (np.ndarray): The input data.
            y (np.ndarray): The output data.

        Returns:
            None
        """
        X_extend = np.hstack((np.ones((X.shape[0], 1)), X))
        w_extend = np.linalg.pinv(X_extend.T @ X_extend) @ X_extend.T @ y
        self.w = w_extend[1:]
        self.b = w_extend[0]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input.

        Arguments:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted output.
        """
        return X @ self.w + self.b


class GradientDescentLinearRegression(LinearRegression):
    """
    A linear regression model that uses gradient descent to fit the model.
    """

    def fit(
        self, X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 1000
    ) -> None:
        """
        Fit the model using gradient descent

        Arguments:
            X (np.ndarray): The input data.
            y (np.ndarray): The output data.
            lr (float): The learning rate.
            epochs (int): The epochs.

        Returns:
            None
        """
        X_extend = np.hstack((np.ones((X.shape[0], 1)), X))
        w_extend = np.random.rand(X_extend.shape[1])
        for _ in range(epochs):
            y_pred = X_extend @ w_extend
            dw = 2 * X_extend.T @ (y_pred - y)
            w_extend -= lr * dw
        self.w = w_extend[1:]
        self.b = w_extend[0]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input.

        Arguments:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted output.
        """
        return X @ self.w + self.b
