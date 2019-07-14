def integrate(f, a, b, h, method='rectangle'):
    res = 0
    
    if method == 'rectangle':
        xi = a
        while xi <= b:
            res += f((xi + xi + h) / 2) * h
            xi += h
    elif method == 'trapeze':
        xi = a
        while xi <= b:
            res += (f(xi) + f(xi + h)) * h / 2
            xi += h
    elif method == 'parabola':
        xi = a
        while xi <= b:
            res += (f(xi) + 4 * f((xi + xi + h) / 2) + f(xi + h)) * h / 6
            xi += h
            
    return res

def precision(I1, I2, h1, h2, p):
    return (I1 - I2) / ((h2 / h1) ** p - 1)
            

if __name__ == '__main__':
    f = lambda x: 1 / (x ** 4 + 16)
    x0 = 0
    xk = 2
    h1 = 0.25
    h2 = 0.5
    
    methods = {'rectangle' : 2, 'trapeze' : 2, 'parabola' : 4}
    print('{:10s}   {:8.2f}   {:8.2f}   {:10s}'.format('', h1, h2, 'precision'))
    for mthd in methods:
        I1 = integrate(f, x0, xk, h1, method=mthd)
        I2 = integrate(f, x0, xk, h2, method=mthd)
        print('{:10s}   {:8.6f}   {:8.6f}   {:8.6f}'.format(mthd, I1, I2, precision(I1, I2, h1, h2, methods[mthd])))