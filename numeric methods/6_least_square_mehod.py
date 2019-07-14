import matplotlib.pyplot as plt 
import numpy as np

class LSAproximation:
    def __init__(self, x, y, n):
        self.n = n
        self.coef = []
        
        P = [[x[i] ** j for j in range(n + 1)] for i in range(len(x))]
        self.coef = np.linalg.solve(np.dot(np.transpose(P), P), 
                                    np.dot(np.transpose(P), y))
        
        self.error = sum(map(lambda x: x ** 2, np.dot(P, self.coef) - y))
    
    def __call__(self, x):
        return sum(k * x ** i for i, k in enumerate(self.coef))
    
    def getError(self):
        return self.error


if __name__ == '__main__':
    xrange = np.arange(-1, 4, 0.1)
    
    x = [-0.9, 0, 0.9, 1.8, 2.7, 3.6]
    y = [-1.2689, 0, 1.2689, 2.6541, 4.4856, 9.9138]
    
    f1 = LSAproximation(x, y, 1)
    f2 = LSAproximation(x, y, 5)
    
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,3))

    ax1.plot(x, y, 'ro')
    ax1.plot(xrange, [f1(x) for x in xrange])
    ax1.set_title('error = {}'.format(f1.getError()))
    ax1.grid()
    
    ax2.plot(x, y, 'ro')
    ax2.plot(xrange, [f2(x) for x in xrange])
    ax2.set_title('error = {}'.format(f2.getError()))
    ax2.grid()
    
    plt.show()