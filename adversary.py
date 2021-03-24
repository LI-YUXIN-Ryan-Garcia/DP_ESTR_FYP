import numpy as np
import pandas as pd
from scipy.stats import laplace
from epsilon import get_subset

class Adversary():
    def __init__(self):
        pass
    def guess(self):
        pass

class MLEAdversary(Adversary):
    """ 
    Adversary makes a guess based on the maximum posterior probability.

    Assume an adversary knows the dataset as much as possible: original dataset
    configuration `dataset`, privacy budget `epsilon`, global sensitivity `gs`
    (or noted by `df`). And the adversary also knows that subsets differ from 
    the original dataset at most one record. 
    
    The adversary queries against a dataset through a differential privacy 
    mechanism then computes posterior probabiliy of all possible configurations
    of the dataset it queried according to the response from the mechanism. It
    guesses the dataset is the one with the largest posterior probability.
    """
    def __init__(self):
        super(MLEAdversary, self).__init__()

    def guess(self, res, query_func, epsilon, global_sensitivity, data:pd.DataFrame):
        # subset query result
        subset = get_subset(data)
        qry_res = subset.apply(query_func, axis=0).to_numpy() 

        # posterior probability
        s = global_sensitivity / epsilon
        posterior = laplace.pdf(res - qry_res, loc=0, scale=s)
        posterior /= posterior.sum()
        
        m = posterior.max()
        m_index = np.argmax(posterior)
        print('max post prob: {}, subset index: {}'.format(m, m_index))
        print("MLEAdversary thinks the dataset config is")
        print(subset[m_index])

