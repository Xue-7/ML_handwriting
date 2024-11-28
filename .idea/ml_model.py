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


