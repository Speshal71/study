import matplotlib.pyplot as plt
import numpy as np

def EulerMethod(f, f0, a, b, h):
    y = [f0]
    y_prev = f0[:]
    y_cur  = [None] * len(f)
    x = a
    while x < b - h:
        for i in range(len(f)):
            y_cur[i] = y_prev[i] + h * f[i](x, *y_prev)

        y.append(y_cur[:])
        y_prev, y_cur = y_cur, y_prev
        x += h
    
    return y


def RungeKuttaMethod(f, g, f0, g0, a, b, h):
    y = [[f0], [g0]]
    f_prev = f0
    g_prev = g0
    x = a
    while x < b - h:
        K1 = h * f(x, f_prev, g_prev)
        L1 = h * g(x, f_prev, g_prev)
        K2 = h * f(x + h / 2, f_prev + K1 / 2, g_prev + L1 / 2)
        L2 = h * g(x + h / 2, f_prev + K1 / 2, g_prev + L1 / 2)
        K3 = h * f(x + h / 2, f_prev + K2 / 2, g_prev + L2 / 2)
        L3 = h * g(x + h / 2, f_prev + K2 / 2, g_prev + L2 / 2)
        K4 = h * f(x + h, f_prev + K3, g_prev + L3)
        L4 = h * g(x + h, f_prev + K3, g_prev + L3)
        f_prev = f_prev + (K1 + 2 * K2 + 2 * K3 + K4) / 6
        g_prev = g_prev + (L1 + 2 * L2 + 2 * L3 + L4) / 6
        
        y[0].append(f_prev)
        y[1].append(g_prev)
        x += h
    
    return y

    
def AdamsMethod(f, f0, a, b, h):
    if len(f0) != 4:
        return
    
    y = [*f0]
    y_cur  = [None] * len(f)
    x = a + 3 * h
    while x < b - h:
        for i in range(len(f)):
            last = len(y) - 1
            y_cur[i] = (y[last][i] 
                        + h / 24 * (55 * f[i](x, *y[last]) 
                        - 59 * f[i](x - h, *y[last - 1]) 
                        + 37 * f[i](x - 2 * h, *y[last - 2]) 
                        - 9 * f[i](x - 3 * h, *y[last - 3])))

        y.append(y_cur[:])
        x += h
    
    return y


if __name__ == '__main__':
    #отрезок, на котором мы ищем решение:
    a = 1
    b = 2
    h = 0.1
    x = np.arange(a, b, h)
    
    #решение задачи Коши
    y = (-0.9783 * np.cos(2 * x) + 0.4776 * np.sin(2 * x)) / np.sin(x)
    
    #известные производные
    f = lambda x, y, z: z
    g = lambda x, y, z: -2 * z / np.tan(x) - 3 * y
    
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12,4))
    
    #Euler
    ax1.plot(x, y)
    ax1.plot(x, [i[0] for i in EulerMethod([f, g], [1, 1], a, b, h)])
    ax1.set_title('Euler method')
    ax1.grid()
    
    #Runge-Kutta
    ax2.plot(x, y)
    ax2.plot(x, RungeKuttaMethod(f, g, 1, 1, a, b, h)[0])
    ax2.set_title('Runge-Kutta method')
    ax2.grid()
    
    #Adams
    r = RungeKuttaMethod(f, g, 1, 1, a, b, h)
    table = [[r[0][i], r[1][i]] for i in range(4)]
    
    ax3.plot(x, y)
    ax3.plot(x, [i[0] for i in AdamsMethod([f, g], table, a, b, h)])
    ax3.set_title('Adams method')
    ax3.grid()
    
    plt.show()