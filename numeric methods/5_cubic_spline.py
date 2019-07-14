import matplotlib.pyplot as plt 
import numpy as np

def TDMAsolve(a, b, c, f):
    n = len(f)
    alpha = [-c[0] / b[0]]
    beta  = [ f[0] / b[0]]
    
    for i in range(1, n):
        alpha.append(-c[i] / (b[i] + a[i] * alpha[i - 1]))
        beta.append((f[i] - a[i] * beta[i - 1]) / (b[i] + a[i] * alpha[i - 1]))
    
    x = [None] * n
    x[n - 1] = beta[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = alpha[i] * x[i + 1] + beta[i]
    
    return x

class Spline:
    def __init__(self, x0, h, y):
        n = len(y)
        self.x0 = x0
        self.h = h
        self.coef = [[y[i], None, 0, None] for i in range(n - 1)]
        
        a = [h for i in range(2, n)]
        a[0] = 0
        b = [4 * h for i in range(2, n)]
        c = [h for i in range(2, n)]
        c[n - 3] = 0
        f = [3 * ((y[i] - y[i - 1]) / h - (y[i - 1] - y[i - 2]) / h) for i in range(2, n)]
        
        ci = TDMAsolve(a, b, c, f)
        for i in range(1, n - 1):
            self.coef[i][2] = ci[i - 1]
        
        for i in range(n - 2):
            self.coef[i][1] = ((y[i + 1] - y[i]) / h 
                               - h * (self.coef[i + 1][2] 
                               + 2 * self.coef[i][2]) / 3)
                              
            self.coef[i][3] = (self.coef[i + 1][2] 
                              - self.coef[i][2]) / 3 / h
        
        self.coef[n - 2][1] = ((y[n - 1] - y[n - 2]) / h 
                               - 2 / 3 * h * self.coef[n - 2][2])
        
        self.coef[n - 2][3] = -self.coef[n - 2][2] / 3 / h
        
        
    def __call__(self, x):
        index = int((x - self.x0) / self.h)
        x = x - self.x0 - index * self.h

        coef = self.coef[index]
        
        return coef[0] + coef[1] * x + coef[2] * x ** 2 + coef[3] * x ** 3
    
            
if __name__ == '__main__':
    y = [6, 1.8415, 2.9093, 3.1411, 6.2432, 3, 2, 4, 4, 2, 9]
    x_start = 0
    x_step = 1
    s = Spline(x_start, x_step, y)
    
    xrange = np.arange(x_start, x_start + len(y) * x_step - x_step, 0.01)
    
    plt.plot(np.arange(x_start, x_start + len(y) * x_step, x_step), y, 'ro')
    plt.plot(xrange, [s(x) for x in xrange])
    plt.grid()
    
    plt.show()
