import matplotlib.pyplot as plt 
import numpy as np

class LagrangePolynomial:
    def __init__(self, x, y):
        n = min(len(x), len(y))
        self.xi = x[:n]
        self.coef = y[:n]
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    self.coef[i] /= (self.xi[i] - self.xi[j])
    
    def __call__(self, x):
        y = 0
        
        for i in range(len(self.coef)):
            coef = 1
            for j in range(len(self.xi)):
                if i != j:
                    coef *= (x - self.xi[j])
            y += coef * self.coef[i]
        
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


if __name__ == '__main__':
    xrange = np.arange(0, 3 * np.pi / 8, 0.01)
    
    x1 = [0, np.pi / 8, 2 * np.pi / 8, 3 * np.pi / 8]
    y1 = [np.tan(x) + x for x in x1]
    P1 = LagrangePolynomial(x1, y1)

    x2 = [0, np.pi / 8, np.pi / 3, 3 * np.pi / 8]
    y2 = [np.tan(x) + x for x in x2]
    P2 = NewtonPolynomial(x2, y2)
    
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,3))
    
    ax1.plot(x1, y1, 'ro')
    ax1.plot(xrange, [P1(x) for x in xrange])
    ax1.set_title('Lagrange interpolation')
    ax1.grid()
    
    ax2.plot(x2, y2, 'ro')
    ax2.plot(xrange, [P2(x) for x in xrange])
    ax2.set_title('Newton interpolation')
    ax2.grid()
    
    plt.show()
            