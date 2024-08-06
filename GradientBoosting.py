import numpy as np 
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from copy import deepcopy

from TensorDecisionTreeRegressor import *
#Debugging import
import importlib
var = 'TensorDecisionTreeRegressor'
package = importlib.import_module(var)
for name, value in package.__dict__.items():
    if not name.startswith("__"):
        globals()[name] = value

# Gradient Boosting Regressor Class
class GradientBoostingRegressor:
    def __init__(self, n_estimators, learning_rate, weak_learner, n_iterations=1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        #self.loss_function = loss_function
        self.weak_learner = weak_learner
        self.models = [deepcopy(self.weak_learner) for _ in range(n_estimators)]
        self.initial_model = None
        self.n_iterations = n_iterations
        self.pruning = False
        
    def fit(self, X, y, X_test=None, y_test=None, method='mean'):
        # Initialize model with a constant value
        self.initial_model = self.initialize_model(y)
        current_pred = np.mean(y)*np.ones(shape=y.shape)
        # Iteratively add weak learners
        for j in range(self.n_iterations):
            for i in range(self.n_estimators):
                # Calculate predictions of the current model
                current_pred = self.predict(X,regression_method=method)
                #print(j,i,'',current_pred)
                # Calculate negative gradient (residual)
                residual = y - current_pred
                print(j,i,'training MSE ===',np.mean(residual**2))
                if (X_test is not None) and (y_test is not None):
                    current_pred_test = self.predict(X_test,regression_method=method)
                    residual_test = y_test - current_pred_test
                    print(j,i,'testing MSE ===',np.mean(residual_test**2))
                
                #derivative = self.loss_derivative(y, self.predict(X))
                
                # Fit weak learner to residual
                self.models[i].fit(X, residual)
                if self.pruning:
                    self.models[i].prune()
                # Add the fitted weak learner to the ensemble
                #self.models.append(learner)

    def predict(self, X, regression_method='mean'):
        # Start with initial model prediction
        y_pred = self.initial_model * np.zeros(X.shape[0])
        
        # Add predictions from all weak learners
        for learner in self.models:
            #print(learner.predict(X))
            learner_pred = learner.predict(X,regression_method)
            if learner_pred is not None:
                y_pred += self.learning_rate * learner_pred
        
        return y_pred

    def initialize_model(self, y):
        # Typically initialize with mean of y for regression tasks
        return np.mean(y)

    def loss_derivative(self, y, y_pred):
        # Example: derivative of the mean squared error loss
        if self.loss_function=='mse':
            return -2 * (y - y_pred)
        else:
            raise Exception('Not implemented: loss_derivative corresponding to ',self.loss_function)

from sklearn.tree import DecisionTreeRegressor
import numpy as np
from copy import deepcopy

class GeneralizedBoostingRegressor:
    def __init__(self, n_estimators, learning_rate, weak_learner, adaboost_resampling_proportion=0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        #self.loss_function = loss_function
        self.weak_learner = weak_learner
        assert adaboost_resampling_proportion <=1
        self.adaboost_resampling_proportion = adaboost_resampling_proportion
        self.models = [deepcopy(self.weak_learner) for _ in range(n_estimators)]
        self.initial_model = None

    def fit(self, X, y):
        # Initialize model with a constant value (mean of y)
        self.initial_model = np.mean(y)
        current_pred = np.full(y.shape, self.initial_model)

        # Initialize sample weights for resampling (if enabled)
        sample_weights = np.ones(X.shape[0]) / X.shape[0]

        # Iteratively fit weak learners
        for i in range(self.n_estimators):
            # Calculate residuals
            residuals = y - current_pred

            # Resample based on current sample weights (if resampling is enabled)
            if self.adaboost_resampling_proportion>0:
                indices = np.random.choice(np.arange(X.shape[0]), size=int(X.shape[0]*self.adaboost_resampling_proportion), p=sample_weights)
                X_resampled, residuals_resampled = X[indices], residuals[indices]
                self.models[i].fit(X_resampled, residuals_resampled)
            else:
                self.models[i].fit(X, residuals)

            # Update predictions
            update = self.models[i].predict(X)
            current_pred += self.learning_rate * update

            # Update sample weights (if resampling is enabled)
            if self.adaboost_resampling_proportion>0:
                # Increase weights for samples with larger residuals
                sample_weights *= np.exp(np.abs(residuals))
                sample_weights /= np.sum(sample_weights)

    def predict(self, X):
        # Start with the initial model's prediction
        y_pred = np.full(X.shape[0], self.initial_model)

        # Add predictions from all weak learners
        for model in self.models:
            y_pred += self.learning_rate * model.predict(X)

        return y_pred

from sklearn.utils import resample
class RandomForestRegressor:
    def __init__(self, n_estimators, loss_function, weak_learner):
        self.n_estimators = n_estimators
        self.loss_function = loss_function
        self.weak_learner = weak_learner
        self.models = [deepcopy(self.weak_learner) for _ in range(n_estimators)]

    def fit(self, X, y):
        for i in range(self.n_estimators):
            X_sample, y_sample = resample(X, y)  # Bootstrap sampling
            self.models[i].fit(X_sample, y_sample)

    def predict(self, X):
        predictions = np.array([model.predict(X) for model in self.models])
        return np.mean(predictions, axis=0)  # Averaging predictions

    def loss_derivative(self, y, y_pred):
        # Example: derivative of the mean squared error loss
        if self.loss_function == 'mse':
            return -2 * (y - y_pred)
        else:
            raise Exception('Not implemented: loss_derivative corresponding to ', self.loss_function)

from sklearn.utils import resample
class RandomForestRegressor:
    def __init__(self, n_estimators, loss_function, weak_learner, max_features=None):
        self.n_estimators = n_estimators
        self.loss_function = loss_function
        self.weak_learner = weak_learner
        self.models = [deepcopy(self.weak_learner) for _ in range(n_estimators)]
        self.max_features = max_features  # The number of features to consider when looking for the best split

    def fit(self, X, y):
        n_features = X.shape[1]
        if self.max_features is None:
            # By default, use sqrt(n_features) as in the classic random forest
            self.max_features = int(np.sqrt(n_features))
        elif isinstance(self.max_features, float) and self.max_features <= 1.0:
            # If max_features is a float, use it as a percentage of the total
            self.max_features = int(self.max_features * n_features)

        for i in range(self.n_estimators):
            # Bootstrap sampling for rows
            X_sample, y_sample = resample(X, y)
            # Randomly select features for each tree
            features_idx = np.random.choice(range(n_features), self.max_features, replace=False)
            X_sample_features = X_sample[:, features_idx]
            # Fit the weak learner to the bootstrapped sample and selected features
            self.models[i].fit(X_sample_features, y_sample)

    def predict(self, X):
        # Aggregate predictions from each model
        predictions = np.array([model.predict(X[:, model.feature_importances_ > 0]) for model in self.models])
        # Average predictions
        return np.mean(predictions, axis=0)

    def loss_derivative(self, y, y_pred):
        # Example: derivative of the mean squared error loss
        if self.loss_function == 'mse':
            return -2 * (y - y_pred)
        else:
            raise Exception('Not implemented: loss_derivative corresponding to ', self.loss_function)
