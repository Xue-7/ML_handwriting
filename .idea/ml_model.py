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


