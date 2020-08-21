import numpy as np
from scipy.optimize import fmin_tnc


class LogisticRegressionUsingGD:

    @staticmethod
    def sigmoid(x):
        # Activation function used to map any real value between 0 and 1
        #print(f"in sigmoid:{1/(1+np.exp(-x))}")
        #print(f'x in sigmooid={x}')
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def net_input(theta, x):
        # Computes the weighted sum of inputs Similar to Linear Regression

        return np.dot(x, theta)

    def probability(self, theta, x):
        # Calculates the probability that an instance belongs to a particular class
        #print(f'net_input={self.net_input(theta, x)}')
        #print(f'sigmoid={self.sigmoid(self.net_input(theta, x))}')
        return self.sigmoid(self.net_input(theta, x))

    def cost_function(self, theta, x, y, sample_weights): 
        # Computes the cost function for all the training samples
        m = x.shape[0]
        
        total_cost = -(1 / m) * np.sum(
            (y * np.log(self.probability(theta, x)) + (1 - y) * np.log(
                1 - self.probability(theta, x))))
        
        return total_cost
    

    def gradient(self, theta, x, y, sample_weights):
        m = x.shape[0]
        theta=theta.reshape(-1,1)
        # Computes the gradient of the cost function at the point theta
        
        return (1/ m) * np.dot(x.T, (self.sigmoid(self.net_input(theta, x)) - y))


    def gradient_converge(self, x, y, theta):
        if(np.all((self.gradient(theta, x, y)==0))):
            return True
        return False

    def fit(self, x, y, theta, sample_weights):
        """trains the model from the training data
        x: array-like, shape = [n_samples, n_features]
            Training samples
        y: array-like, shape = [n_samples, n_target_values]
            Target classes
        theta: initial weights
        Returns
        -------
        self: An instance of self
        """
        m=x.shape[0]
        
        x_with_one_column = np.c_[np.ones((x.shape[0], 1)), x]
        sample_weights=sample_weights.reshape(-1,1)
        y=y.reshape(-1,1)
        samples_output = np.where(y==-1, 0, y) 
        #print(f'samples_output={samples_output}')
        for i in range(100000):
            print(f'cost func={self.cost_function(theta, x_with_one_column, samples_output, sample_weights)}')
            #self.cost_function(theta, x_with_one_column, samples_output, sample_weights)
            theta = theta - (0.001)*self.gradient(theta, x_with_one_column, samples_output, sample_weights)
            
        self.w_=theta.flatten()
        
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
        #theta = self.w_[:, np.newaxis]
        #return self.probability(theta, x)
        
        x_with_one_column = np.c_[np.ones((x.shape[0], 1)), x]
        theta = self.w_[:, np.newaxis]

        predicts = self.probability(theta, x_with_one_column)
        
        predicts[predicts>=0.5]=1
        predicts[predicts<0.5]=0
            
        
    
        return predicts.flatten().astype(int)

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
        samples_output = np.where(actual_classes==-1, 0, actual_classes)
        predicted_classes = (self.predict(x) >= probab_threshold).astype(int)
        predicted_classes = predicted_classes.flatten()
        print(f'actual actual classes={samples_output}')
        print(f'predict={self.predict(x) }')
        print(f'acuuracy={np.sum(predicted_classes == samples_output)}')
        accuracy = np.mean(predicted_classes == samples_output)
        return accuracy * 100