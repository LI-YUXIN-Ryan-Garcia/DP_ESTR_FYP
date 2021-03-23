import numpy as np

def subset(x):
    """ Return subsets of x differing in at most one"""
    n = len(x)
    out = np.empty((n,n-1), dtype='float32')
    for i in range(n):
        out[i] = np.append(x[:i], x[i+1:], axis=0)
    #print(out)
    return np.flip(out, 0)

def delta_f(x, func=np.mean):
    """ Global sensitivity of x """
    n = len(x)
    y = subset(x)
    m = np.absolute(func(x) - func(y[1])) # max
    for i in range(n):
        tmp = np.absolute(func(x) - func(y[i]))
        if m < tmp:
            m = tmp
    return m

def delta_v(x, func=np.mean):
    n = x.shape[0]
    m = np.absolute(func(x[1]) - func(x[2])) # max
    for i in range(n-1):
        for j in range(i+1,n):
            tmp = np.absolute(func(x[i]) - func(x[j]))
        if m < tmp:
            m = tmp
    return m

def max_epsilon(data, func=np.mean, rho=0):
    x = subset(data)
    n = x.shape[0]
    df = np.max(list(map(delta_f, x[:])))
    dv = delta_v(x,func)
    r = 1 / n if rho == 0 else rho
    e = df * np.log((n-1)*r/(1-r)) / dv
    print(n, df, dv, r, e, (n-1)*r)
    return e

data = np.array([1,2,3,10])
x = subset(data)
print(x)
df = np.max(list(map(delta_f, x[:])))
dv = delta_v(x)
print(df,dv)
max_e = max_epsilon(data, func=np.mean)
print(max_e)