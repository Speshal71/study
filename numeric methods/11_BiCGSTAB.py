import numpy as np
import scipy.sparse.linalg
import scipy.sparse as ss

np.seterr(all='raise')

class SparseMatrix:
    def __init__(self, n):
        self.rows = [SparseRow() for i in range(n)]

    def __getitem__(self, key):
        return self.rows[key]
    
    def dot(self, v):
        if isinstance(v, (np.ndarray, list)):
            n = len(v)
            ret = np.zeros(n)
            
            for i in range(n):
                for j in self.rows[i].elems:
                    ret[i] += self.rows[i][j] * v[j]
            
            return ret
        return None


class SparseRow(dict):
    def __init__(self):
        self.elems = dict()
    
    def __setitem__(self, key, val):
        self.elems[key] = val
    
    def __getitem__(self, key):
        return self.elems.get(key, 0.)
    
    def dot(self, v):
        if isinstance(v, (np.ndarray, list)):
            ret = 0
            for i in self.elems:
                ret += self.elems[i] * v[i]
            return ret
        return None

class RhoBreakdown(Exception): pass
class OmegaBreakdown(Exception): pass

def BiCGSTAB(A, b, eps=0.00001):
    n = len(b)
    iter_count = 0
    iter_max = 5 * n
    
    x = np.zeros(n)
    r_hat = np.array(b)
    r = np.array(b)
    
    rho = alpha = omega = 1
    rho_next = np.dot(r, r)
    
    v = np.zeros(n)
    p = np.zeros(n)
    
    while iter_count < iter_max and x[0] == x[0]:
        beta = rho_next / rho * alpha / omega
        
        rho = rho_next
        if rho == 0.0:
            raise RhoBreakdown
        
        p *= beta
        p -= beta * omega * v
        p += r
    
        v = A.dot(p)

        alpha = rho / np.dot(r_hat, v)
        s = r - alpha * v
        
        if np.linalg.norm(s) < eps:
            x += alpha * p
            break
        
        t = A.dot(s)
        
        omega = np.dot(t,s) / np.dot(t,t)
        if omega == 0.0:
            raise OmegaBreakdown
        
        rho_next = -omega * np.dot(r_hat,t)
        
        r = s - omega * t
        
        x += omega * s
        x += alpha * p
        
        if np.linalg.norm(r) < eps * np.linalg.norm(b):
            break
        
        iter_count += 1
    
    return x, iter_count

def main(from_file=True):
    if from_file:
        row = []
        col = []
        data = []
        file = open('matrix.txt', 'r')
    
        n = int(file.readline())
        m = int(file.readline())
        A = SparseMatrix(n)
    
        for k in range(m):
            i, j, val = file.readline().split()
            i, j, val = int(i), int(j), float(val)
            A[i][j] = val
            row.append(i)
            col.append(j)
            data.append(val)
        
        b = []
        for k in range(n):
            b.append(float(file.readline()))
            
        S = ss.csc_matrix((data, (row, col)), shape=(n, n))
    else:
        A = np.array([[1, 2, 3],
                      [2, 3, 4],
                      [4, 6, 8]])
    
        b = np.array([14, 20, 40])
        
        row = [0, 0, 0, 1, 1, 1, 2, 2, 2]
        col = [0, 1 ,2, 0, 1, 2, 0, 1 ,2]
        data = [1, 2, 3, 2, 3, 4, 4, 6, 8]
        
        S = ss.csc_matrix((data, (row, col)), shape=(3, 3))
    try:
        print('scipy bicgstab:')
        sol, flag = ss.linalg.bicgstab(S, b)
        print(sol[-5:], flag)
        
        print('my bicgstab:')
        sol, icnt = BiCGSTAB(A, b)
        print(sol[-5:], icnt)
    except RhoBreakdown:
        print('rho breakdown')
    except OmegaBreakdown:
        print('omega breakdown')
    except:
        print('incorrect system, floating point breakdown')

if __name__ == '__main__':
    main()
