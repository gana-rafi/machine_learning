import numpy as np
from scipy.optimize import fmin_tnc
import pdb


class LogisticRegressionUsingGD:

    @staticmethod
    def sigmoid(x):
        # Activation function used to map any real value between 0 and 1
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def net_input(theta, x):
        # Computes the weighted sum of inputs Similar to Linear Regression

        return np.dot(x, theta)

    def probability(self, theta, x):
        # Calculates the probability that an instance belongs to a particular class

        return self.sigmoid(self.net_input(theta, x))

    def cost_function(self, theta, x, y, weights, learning_coeff):
        """the weights that we added, changes the cost"""
        # Computes the cost function for all the training samples
        m = x.shape[0]
        total_cost = -(1 / m) * np.sum(
            weights * (y * np.log(self.probability(theta, x)) + (1 - y) * np.log(
                1 - self.probability(theta, x))))
        return total_cost

    def gradient(self, theta, x, y, weights, learning_coeff):
        # Computes the gradient of the cost function at the point theta
        m = x.shape[0]
        theta = theta.reshape(-1,1)
        return learning_coeff * (1 / m) * np.dot(x.T, weights * self.sigmoid(self.net_input(theta, x)) - y)

    def fit(self, x, y, sample_weight):
        """trains the model from the training data
        Uses the fmin_tnc function that is used to find the minimum for any function
        It takes arguments as
            1) func : function to minimize
            2) x0 : initial values for the parameters
            3) fprime: gradient for the function defined by 'func'
            4) args: arguments passed to the function
        Parameters
        ----------
        x: array-like, shape = [n_samples, n_features]
            Training samples
        y: array-like, shape = [n_samples, n_target_values]
            Target classes
        sample_weight: initial weights
        Returns
        -------
        self: An instance of self
        """
        """ the difference is that the weights are being optimized by the model. the coeefficiants(theta) are optimized from random"""
        theta = np.random.rand(x.shape[1] + 1) #generate random vector
        x_with_extra_freedom = np.c_[np.ones(x.shape[0]), x]
        y[y == -1] = 0
        learning_coeff = 0.01

        opt_weights = fmin_tnc(func=self.cost_function, x0=theta, fprime=self.gradient,
                               args=(x_with_extra_freedom, y.reshape(-1, 1), sample_weight.reshape(-1, 1), learning_coeff))
        self.w_ = opt_weights[0]
        return self

    def predict(self, x):
        """ Predicts the class labels
        Parameters
        ----------
        x: array-like, shape = [n_samples, n_features]
            Test samples
        Returns
        -------
        predicted class labels
        """
        theta = self.w_[:, np.newaxis]
        x_with_extra_freedom = np.c_[np.ones(x.shape[0]), x]
        """map predictions to logistic values (classes)"""
        result = self.probability(theta, x_with_extra_freedom).round()
        result[result == 0] = -1

        return result.flatten()

    def accuracy(self, x, actual_classes, probab_threshold=0.5):
        """Computes the accuracy of the classifier
        Parameters
        ----------
        x: array-like, shape = [n_samples, n_features]
            Training samples
        actual_classes : class labels from the training data set
        probab_threshold: threshold/cutoff to categorize the samples into different classes
        Returns
        -------
        accuracy: accuracy of the model
        """
        predicted_classes = (self.predict(x) >= probab_threshold).astype(int)
        predicted_classes = predicted_classes.flatten()
        accuracy = np.mean(predicted_classes == actual_classes)
        return accuracy * 100
