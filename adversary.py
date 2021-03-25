import numpy as np
import pandas as pd
from scipy.stats import laplace
from epsilon import get_subset

def sigmoid(x): return 1 / (1 + np.exp(-x))


class Adversary():
    def __init__(self):
        pass
    def add_data(self):
        pass
    def guess(self):
        pass
    def show_config(self):
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

    def add_data(self, data:np.ndarray):
        self.data = data

    def guess(self, res, query_func, epsilon, global_sensitivity):
        # subset query result
        subset = get_subset(self.data)
        qry_res = query_func(subset, axis=0) 

        # posterior probability
        s = global_sensitivity / epsilon
        posterior = laplace.pdf(res - qry_res, loc=0, scale=s)
        # posterior = sigmoid(posterior)
        posterior /= posterior.sum()
        
        # make a guess
        m = posterior.max() # 0.001036295052801138
        m_index = np.argmax(posterior)
        print('max post prob: {}, subset index: {}'.format(m, m_index))
        
        m_subset = subset[:, m_index]
        values, counts = np.unique(m_subset, return_counts=True)
        config = dict(zip(values, counts))
        print("MLE Adversary thinks the dataset config is {}".format(config))
        

class ImbalanceMLEAdversary(MLEAdversary):
    def __init__(self):
        super(ImbalanceMLEAdversary, self).__init__()

    def guess(self, res, query_func, epsilon, global_sensitivity):
        data = self.data
        # subset query result
        subset = get_subset(data)
        qry_res = query_func(subset, axis=0)

        # imbalance-class posterior probability
        s = global_sensitivity / epsilon
        posterior = (laplace.pdf(res - qry_res, loc=0, scale=s))
        unique, counts = np.unique(posterior, return_counts=True)
        max_cls = counts.max()
        weights = max_cls / counts
        weight_dict = dict(zip(unique, weights))
        def add_weights(x, wdict):
            return x * wdict[x]
        posterior = np.vectorize(add_weights)(posterior, weight_dict)
        
        posterior /= posterior.sum()
        m = posterior.max() #  0.025454860961990372
        m_index = np.argmax(posterior)
        print('max post prob: {}, subset index: {}'.format(m, m_index))
        
        m_subset = subset[:, m_index]
        values, counts = np.unique(m_subset, return_counts=True)
        config = dict(zip(values, counts))
        print("Imbalance-class MLE Adversary thinks the dataset config is {}".format(config))



