import os
import numpy as np
import pandas as pd
from epsilon import get_subset, max_epsilon
from adversary import Adversary, MLEAdversary, ImbalanceMLEAdversary

class DifferentialPrivacy():
    def __init__(self):
        pass

    def read_data(self, file:str):
        self.data = pd.read_csv(file) # dataframe

    def query(self, func, column, real):
        data = self.data[column].to_numpy()
        self.numpy_data = data
        subset = get_subset(data)
        e, df, dv, _, _ = max_epsilon(subset, func=func, rho=1/3)

        np.random.seed(1155107874)
        s = df / e
        gamma = real + np.random.laplace(scale=s)
        return {
            "response":gamma, 
            'epsilon':e, 
            'global sensitivity':df, 
            'max distance':dv}

    def add_adversary(self, Adv:Adversary):
        self.adv = Adv()
        if Adv == MLEAdversary or Adv == ImbalanceMLEAdversary:
            self.adv.add_data(self.numpy_data)


if __name__ == '__main__':
    test_data = pd.DataFrame({'test':[1,2,3,10]})
    data = test_data.to_numpy()
    dpm = DifferentialPrivacy()
    dpm.data = test_data # read data
    
    query_func = np.mean
    para = dpm.query(query_func, column='test', real=2)
    res = para.get('response')
    epsilon = para.get('epsilon')
    gs = para.get('global sensitivity')

    print(res, epsilon, gs) # -1.8195946629168303 0.38293926876882173 2.833333333333333

    dpm.add_adversary(MLEAdversary)
    # dpm.add_adversary(ImbalanceMLEAdversary)
    adv = dpm.adv
    adv.guess(res, query_func, epsilon, gs)
