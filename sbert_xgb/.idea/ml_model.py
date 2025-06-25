import numpy as np 

class LinearRegression:
    def __init__(self, learning_rate=0.01, max_iter=1000):
        self.learning_rate = learning_rate 
        self.max_iter = max_iter
        
    
    def fit(self, X, y):
        n, d = X.shape 

        self.w = np.zeros(d)
        self.b = 0 


        for _ in range(self.max_iter):
            y_pred = X @ self.w + self.b 

            dw = (2 / n) * np.sum(X.T @ (y_pred - y))
            db = (2 / n) * np.sum(y_pred - y)

            self.w -= self.learning_rate * dw 
            self.b -= self.learning_rate * db 

    def predict(self, X):
        return X @ self.w + self.b 

class LogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.w = None 
        self.b = 0 
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def fit(self, X, y):
        n_samples, n_features = X.shape 
        self.w = np.zeros(n_features)

        for epoch in range(self.epochs):
            y_pred = self.sigmoid(np.dot(X, self.w) + self.b)

            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.w -= self.learning_rate * dw 
            self.b -= self.learning_rate * db 
    
    def predict_proba(self, X):
        return self.sigmoid(np.dot(X, self.w) + self.b)

    def predict(self, X, threshold=0.5):
        probabilities = self.predict_proba(X)

        return (probabilities >= threshold).astype(int)

class LogisticRegressionWithRegularization:
    def __init__(self, learning_rate=0.01, epochs=1000, reg_lambda=0.1):
        self.learning_rate = learning_rate
        self.epochs = epochs 
        self.reg_lambda = reg_lambda 
        self.w = None 
        self.b = 0 

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def fit(self, X, y):
        n_samples, n_features = X.shape 
        self.w = np.zeros(n_features)

        for epoch in range(self.epochs):
            
            y_pred = self.sigmoid(np.dot(X, self.w) + self.b)

            # L2 regularization
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y)) + (self.reg_lambda / n_samples) * self.w 

            # L1 regularization
            # dw = (1 / n_samples) * np.dot(X.T, (y_pred - y)) + (self.reg_lambda / n_samples) * np.sign(self.w)
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.w -= self.learning_rate * dw 
            self.b -= self.learning_rate * db 
    
    def predict_proba(self, X):
        return self.sigmoid(np.dot(X, self.w) + self.b)

    def predict(self, X, threshold=0.5):
        probabilities = self.predict_proba(X)

        return (probabilities >= threshold).astype(int)
    

# Decision Tree

import numpy as np 

class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None 

    def gini(self, y):
        classes = np.unique(y)
        gini = 1.0 
        for cls in classes:
            p = np.sum(y == cls) / len(y) 
            gini -= p ** 2
        return gini 
    
    def entropy(self, y):
        classes, counts = np.unique(y, return_counts=True)

        probabilities = counts / len(y)
        return -np.sum(probabilities * np.log2(probabilities))

    def split(self, X, y, feature_index, threshold):
        left_indices = np.where(X[:, feature_index] <= threshold)[0]
        right_indices = np.where(X[:, feature_index] > threshold)[0]
        return left_indices, right_indices
    
    def find_best_split(self, X, y):
        best_gini = float('inf')
        best_split = None 

        n_samples, n_features = X.shape 

        for feature_index in range(n_features):
            thresholds = np.unique(X[:, feature_index])

            for threshold in thresholds:
                left_indices, right_indices = self.split(X, y, feature_index, threshold)

                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue 
                
                gini_left = self.gini(y[left_indices])
                gini_right = self.gini(y[right_indices])

                weighted_gini = (len(left_indices) / n_samples) * gini_left + (len(right_indices) / n_samples) * gini_right

                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_split = {
                        'feature_index': feature_index,
                        'threshold': threshold,
                        'left_indices': left_indices,
                        'right_indices': right_indices
                    }
        return best_split

    def build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape 
        num_classes = len(np.unique(y))

        if depth >= self.max_depth or n_samples < self.min_samples_split or num_classes == 1:
            leaf_value = np.bincount(y).argmax() # The leaf node is the major class 
            return {'leaf': True, 'value': leaf_value}
        
        best_split = self.find_best_split(X, y)
        if best_split is None:
            leaf_value = np.bincount(y).argmax() # The leaf node is the major class 
            return {'leaf': True, 'value': leaf_value}
        
        left_subtree = self.build_tree(X[best_split['left_indices'], :], y[best_split['left_indices']], depth + 1)
        right_subtree = self.build_tree(X[best_split['right_indices'], :], y[best_split['right_indices']], depth + 1)

        return {
            'leaf': False,
            'feature_index': best_split['feature_index'],
            'threshold': best_split['threshold'],
            'left': left_subtree,
            'right': right_subtree
        }
    
    def fit(self, X, y):
        self.tree = self.build_tree(X, y)
    
    def predict_sample(self, x, tree):
        """one sample prediction"""
        if tree['leaf']:
            return tree['value']
        feature_index = tree['feature_index']
        threshold = tree['threshold']
        if x[feature_index] <= threshold:
            return self.predict_sample(x, tree['left'])
        else:
            return self.predict_sample(x, tree['right'])

    def predict(self, X):
        """multi-sample prediction"""
        return np.array([self.predict_sample(x, self.tree) for x in X])
    
# Neural Network
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.learning_rate = learning_rate 

        self.W1 = np.random.randn(input_size, hidden_size) * 0.01 
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01 
        self.b2 = np.zeros((1, output_size))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z):
        return z * (1 - z)

    def forward(self, X):

        self.Z1 = np.dot(X, self.W1) + self.b1 
        self.A1 = self.sigmoid(self.Z1)

        self.Z2 = np.dot(self.A1, self.W2) + self.b2 
        self.A2 = self.sigmoid(self.Z2)

        return self.A2 

    def backward(self, X, y, output):
        n_samples = X.shape[0]

        error_output = output - y 
        dW2 = np.dot(self.A1.T, error_output) / n_samples
        db2 = np.sum(error_output, axis=0, keepdims=True) / n_samples

        error_hidden = np.dot(error_output, self.W2.T) * self.sigmoid_derivative(self.A1)
        dW1 = np.dot(X.T, error_hidden) / n_samples
        db1 = np.sum(error_hidden, axis=0, keepdims=True) / n_samples

        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1

    def train(self, X, y, epochs=1000):
        
        for epoch in range(epochs):
            
            output = self.forward(X)
            
            self.backward(X, y, output)

            if epoch % 100 == 0:
                loss = np.mean(-y * np.log(output) - (1 - y) * np.log(1 - output))
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def predict(self, X):
        
        output = self.forward(X)
        return (output > 0.5).astype(int)




