import matplotlib.pyplot as plt
import numpy as np
import time

def extended_euclidean_algorithm(a, b):
    s, old_s = 0, 1
    t, old_t = 1, 0
    r, old_r = b, a

    while r != 0:
        quotient = old_r // r
        old_r, r = r, old_r - quotient * r
        old_s, s = s, old_s - quotient * s
        old_t, t = t, old_t - quotient * t

    return old_r, old_s, old_t

def inverse_of(n, p):
    gcd, x, y = extended_euclidean_algorithm(n, p)
    assert (n * x + p * y) % p == gcd

    if gcd != 1:
        raise ValueError(
            '{} has no multiplicative inverse '
            'modulo {}'.format(n, p))
    else:
        return x % p


class Elliptic_curve:
    def __init__(self, a, b):
        self.a = a
        self.b = b
        
    def __call__(self, x):
        return x ** 3 + self.a * x + self.b

    def add(self, P1, P2, p):
        if P1 == 0:
            return P2
        if P2 == 0:
            return P1

        x1, y1 = P1
        x2, y2 = P2
        
        if x1 == x2 and y1 == -y2 % p:
            return 0
        
        if x1 == x2 and y1 == y2:
            k = (3 * x1 ** 2 + self.a) * inverse_of(2 * y1, p) % p
        else:
            k = (y2 - y1) * inverse_of(x2 - x1, p) % p
        
        x3 = (k ** 2 - x1 - x2) % p
        y3 = (-y1 + k * (x1 - x3)) % p
        
        return (x3, y3)
    
    def order_of_point(self, P, p):
        order = 1
        Q = P
        while Q:
            Q = self.add(P, Q, p)
            order += 1
        return order


if __name__ == '__main__':
    a = 2
    b = 4
    e = Elliptic_curve(a, b)
    P = (0, 2)
    
    primes = [1031, 2053, 4099, 8209, 16411, 32771, 65537, 131101, 262147, 524309]
    time_intervals = []
    
    for p in primes:
        start = time.process_time_ns()
        e.order_of_point(P, p)
        end = time.process_time_ns()
        time_intervals.append((end - start) / 10 ** 9)
        
    poly = np.polyfit(primes, time_intervals, 1)
    approx_line = np.poly1d(poly)
    
    pp = (600 - poly[1]) / poly[0]
    print(pp)
    
    plt.figure(figsize=(10, 7))
    plt.rcParams.update({'font.size': 14})
    
    plt.plot(primes, time_intervals, 'ro')
    plt.plot(primes, approx_line(primes))
    plt.grid()
    plt.xlabel('вычет p')
    plt.ylabel('время нахождения порядка, cек')
    plt.legend(['Фактическое время', 'Приблизительное время'])
    
    plt.show()