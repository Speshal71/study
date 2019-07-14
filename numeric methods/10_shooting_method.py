import matplotlib.pyplot as plt
import numpy as np

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

def ShootingMethod(f, g, a, b, h):
    eps = 0.01
    f0_der = 1
    cond = lambda sol: sol[1] - sol[0] - 1
    
    p_prev = 20
    solution = RungeKuttaMethod(f, g, p_prev,  f0_der, a, b, h)
    sol_prev  = (solution[0][-1], solution[1][-1])
    p_cur  = 18
    solution = RungeKuttaMethod(f, g, p_cur,  f0_der, a, b, h)
    sol_cur  = (solution[0][-1], solution[1][-1])
    
    while abs(cond(sol_cur)) > eps:
        tmp = p_cur
        p_cur = (p_cur - p_prev) / (cond(sol_cur) - cond(sol_prev)) * cond(sol_cur)
        p_prev = tmp
        solution  = RungeKuttaMethod(f, g, p_cur,  f0_der, a, b, h)
        sol_cur   = (solution[0][-1], solution[1][-1])
        
    return solution


if __name__ == '__main__':
    #отрезок, на котором мы ищем решение:
    a = 0
    b = 1
    h = 0.01
    x = np.arange(a, b, h)
    
    #известные производные
    f = lambda x, y, z: z
    g = lambda x, y, z: (2 * z + np.e ** x * y) / (np.e ** x + 1)
    
    
    plt.plot(x, np.e ** x - 1)
    plt.plot(x, ShootingMethod(f, g, a, b, h)[0])
    plt.grid()
    
    plt.show()