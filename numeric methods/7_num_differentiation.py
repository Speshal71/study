import matplotlib.pyplot as plt 
import numpy as np

def combination(l, n, start=0):
    if n < 0:
        return []
    elif n == 0:
        return [[]]
    elif n == 1:
        return [[l[i]] for i in range(start, len(l))]
    
    ret = []
    for i in range(start, len(l) - n + 1):
        comb = combination(l, n - 1, start=i+1)
        for j in comb:
            ret.append([l[i]] + j)

    return ret

def der(x, coef, order):
    y = 1
    
    if order == 0:
        for xi in coef:
            y *= x - xi
    else:
        y = 0
        
        for icoef in combination(coef, len(coef) - 1):
            y += der(x, icoef, order - 1)
        
    return y


class NewtonPolynomial:
    def __init__(self, x, y):
        n = min(len(x), len(y))
        self.xi = x[:n]
        
        self.coef = [y[:n]]
        for i in range(1, n):
            self.coef.append([])
            for j in range(n - i):
                self.coef[i].append((self.coef[i - 1][j + 1] - self.coef[i - 1][j]) 
                                    / (x[j + i] - x[j]))
    
    def __call__(self, x):
        y = 0
        
        for i in range(len(self.xi)):
            coef = 1
            for j in range(i):
                coef *= (x - self.xi[j])
            y += coef * self.coef[i][0]
        
        return y
    
    def derivative(self, x, order):
        y = 0
        
        for i in range(order, len(self.xi)):
            y += self.coef[i][0] * der(x,self.xi[:i], order)
        
        return y
          

if __name__ == '__main__':
    xrange = np.arange(0, np.pi, 0.05)
    
    x = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi]
    y = [0, np.sqrt(2) / 2, 1, np.sqrt(2) / 2, 0]

    p = NewtonPolynomial(x, y)

    plt.plot(x, y, 'ro')
    plt.plot(xrange, [p(xi) for xi in xrange])
    plt.grid()
    
    plt.show()