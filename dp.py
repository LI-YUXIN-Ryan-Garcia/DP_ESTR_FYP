import os
import numpy as np
import pandas as pd
from epsilon import get_subset, max_epsilon
from adversary import MLEAdversary

class DifferentialPrivacy():
    def __init__(self):
        pass

    def read_data(self, file:str):
        self.data = pd.read_csv(file) # dataframe

    def query(self, func, column, real):
        data = self.data[column]
        subset = get_subset(data)
        e, df, dv, _, _ = max_epsilon(subset, func=func, rho=1/3)

        np.random.seed(1155107874)
        s = df / e
        gamma = real + np.random.laplace(scale=s)
        return {
            "response":gamma.to_numpy()[0], 
            'epsilon':e, 
            'global sensitivity':df, 
            'max distance':dv}


if __name__ == '__main__':
    data = pd.DataFrame({'test':[1,2,3,10]})
    dpm = DifferentialPrivacy()
    dpm.data = data # read data
    
    query_func = pd.DataFrame.mean
    para = dpm.query(query_func, column='test', real=2)
    res = para.get('response')
    epsilon = para.get('epsilon')
    gs = para.get('global sensitivity')

    print(res, epsilon, gs)

    mle_adv = MLEAdversary()
    mle_adv.guess(res, query_func, epsilon, gs, data)

