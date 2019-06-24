import numpy as np
from sklearn.preprocessing import PolynomialFeatures

class SVM:
    def __init__(self, kernel='linear', C=1, tol=0.1, maxiter=100, k=2):
        self.tol = tol
        self.C = C
        self.maxiter = maxiter
        
        if kernel == 'linear':
            self.kernel = np.dot
        elif kernel == 'polynomial':
            self.kernel = lambda x, y: (np.dot(x, y) + 1) ** k
        elif kernel == 'rbf':
            self.kernel = lambda x, y: np.exp(-k * np.linalg.norm(x - y) ** 2)
        else:
            raise ValueError('kernel={}'.format(kernel))
    
    def fit(self, X, Y):
        self.X = X
        self.Y = Y
        self.n = X.shape[0]
        self.alpha = np.zeros(self.n)
        self.b = 0
        
        alpha = self.alpha
        it = 0
        while it < self.maxiter:
            alpha_old = np.copy(alpha)
            
            for i in range(self.n):
                E_i = self.f(X[i]) - Y[i]
                if ((Y[i] * E_i < -self.tol and alpha[i] < self.C) or 
                        (Y[i] * E_i > self.tol and alpha[i] > 0)):
                    
                    j = i
                    while j == i:
                        j = np.random.randint(0, self.n)
                    E_j = self.f(X[j]) - Y[j]
                    
                    alpha_i_old = alpha[i]
                    alpha_j_old = alpha[j]
                    
                    if Y[i] != Y[j]:
                        L = max(0, alpha[j] - alpha[i])
                        H = min(self.C, self.C + alpha[j] - alpha[i])
                    else:
                        L = max(0, alpha[j] + alpha[i] - self.C)
                        H = min(self.C, alpha[j] + alpha[i])
                    
                    if L == H:
                        continue
                    
                    p_ii = self.kernel(X[i], X[i])
                    p_ij = self.kernel(X[i], X[j])
                    p_jj = self.kernel(X[j], X[j])
                    
                    eta = 2 * p_ij - p_ii - p_jj
                    
                    if eta >= 0:
                        continue
                    
                    alpha[j] -= Y[j] * (E_i - E_j) / eta
                    alpha[j] = min(alpha[j], H)
                    alpha[j] = max(alpha[j], L)
                    
                    if alpha[j] - alpha_j_old < 10e-5:
                        continue
                    
                    alpha[i] += Y[i] * Y[j] * (alpha_j_old - alpha[j])
                    b1 = (self.b - E_i - Y[i] * (alpha[i] - alpha_i_old) * p_ii - 
                          Y[j] * (alpha[j] -alpha_j_old) * p_ij)
                    b2 = (self.b - E_j - Y[i] * (alpha[i] - alpha_i_old) * p_ij - 
                          Y[j] * (alpha[j] -alpha_j_old) * p_jj)
                    
                    if 0 < alpha[i] < self.C:
                        self.b = b1
                    elif 0 < alpha[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2
            
            if np.linalg.norm(alpha - alpha_old) < self.tol:
                break
            it += 1
    
    def predict(self, X):
        return [np.sign(self.f(x)) for x in X]
    
    def f(self, x):
        return np.dot(self.alpha * self.Y, self.kernel(self.X, x)) + self.b


class PolynomialRegression:
    def __init__(self, order):
        self.poly = PolynomialFeatures(order)
    
    def fit(self, X, Y):
        transformed_X = self.poly.fit_transform(X)
        self.coef = np.linalg.lstsq(transformed_X, Y, rcond=None)[0]
    
    def predict(self, X):
        return np.dot(self.poly.fit_transform(X), self.coef)
