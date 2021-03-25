"""
This file calculates Epsilon -- privacy budget
"""

import numpy as np
import pandas as pd

def get_subset(data:np.ndarray) -> np.ndarray:
    """ Given a vector of n elements, return subsets of the vector differing
        in at most one record"""
    n = data.shape[0]
    data = data.reshape(n,1)
    out = np.zeros((n-1, n))
    for i in range(n):
        new_subset = np.delete(data, i, axis=0)
        out[:,i:i+1] = new_subset
    return out


def get_glb_sens(data:np.ndarray, func=np.mean) -> float:
    """ Global sensitivity of a vector of n elements """
    subset = get_subset(data)
    sensitivity = func(data) - func(subset, axis=0)
    return np.abs(sensitivity).max()


def get_subset_dist(subset:np.ndarray, func=np.mean) -> float:
    """ Return the maximum distance among subsets"""
    n = subset.shape[1]
    results = func(subset, axis=0) # query results
    dist = np.concatenate([[results]*n], axis=0) # TODO
    dist -= dist.transpose()
    return np.abs(dist).max()


def max_epsilon(subset:np.ndarray, func=np.mean, rho=0):
    n = subset.shape[1]
    glb_sens = np.apply_along_axis(get_glb_sens, 0, subset, func)
    glb_sens = glb_sens.max()
    max_dist = get_subset_dist(subset, func=func)
    r = 1 / n if rho == 0 else rho
    e = glb_sens * np.log((n-1)*r/(1-r)) / max_dist
    return (e, glb_sens, max_dist, r, n)


if __name__ == "__main__":
    # preprocess data
    test_data = {'data': [1,2,3,10]}
    df = pd.DataFrame(test_data, columns=['data'])
    data = df.to_numpy()

    subset = get_subset(data)
    f = np.mean
    max_sub_dist = get_subset_dist(subset, func=f)
    para = max_epsilon(subset, func=f, rho=1/3)
    print(para) # (0.38293926876882173, 2.833333333333333, 3.0, 0.3333333333333333, 4)
