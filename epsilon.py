"""
This file calculates Epsilon -- privacy budget
"""

import numpy as np
import pandas as pd

def get_subset(data:pd.DataFrame) -> pd.DataFrame:
    """ Given a vector of n elements, return subsets of the vector differing
        in at most one record"""
    n = data.shape[0]
    names = [i for i in range(n)]
    out = pd.DataFrame(index=np.arange(n-1), columns=names)
    for i in range(n):
        new_subset = data.drop(i).to_numpy()
        out[i] = pd.DataFrame(new_subset) 
    return out


def get_glb_sens(data:pd.DataFrame, func=pd.DataFrame.mean) -> float:
    """ Global sensitivity of a vector of n elements """
    subset = get_subset(data)
    sensitivity = func(data).to_numpy() - func(subset)
    return sensitivity.abs().max()


def get_subset_dist(subset:pd.DataFrame, func=pd.DataFrame.mean) -> float:
    """ Return the maximum distance among subsets"""
    n = subset.shape[1]
    results = func(subset) # query results
    dist = pd.concat([results]*n, axis=1)
    for i in range(n):
        tmp = results[i]
        dist[i] -= tmp
    return dist.abs().max().max()


def max_epsilon(subset:pd.DataFrame, func=pd.DataFrame.mean, rho=0):
    n = subset.shape[1]
    glb_sens = subset.apply(lambda x: get_glb_sens(pd.DataFrame(x), func), axis=0)
    glb_sens = glb_sens.max()
    max_dist = get_subset_dist(subset, func=func)
    r = 1 / n if rho == 0 else rho
    e = glb_sens * np.log((n-1)*r/(1-r)) / max_dist
    return (e, glb_sens, max_dist, r, n)


if __name__ == "__main__":
    data = {'data': [1,2,3,10]}
    df = pd.DataFrame(data, columns=['data'])
    subset = get_subset(df)
    f = pd.DataFrame.mean
    max_sub_dist = get_subset_dist(subset, func=f)
    para = max_epsilon(subset, func=f, rho=1/3)
    print(para) # (0.38293926876882173, 2.833333333333333, 3.0, 0.3333333333333333, 4)
