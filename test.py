import os
import time
import numpy as np
import pandas as pd
from dp import DifferentialPrivacy
from adversary import MLEAdversary, ImbalanceMLEAdversary

dpm = DifferentialPrivacy()
filepath = os.path.join('dataset', 'dataset0.csv')
dpm.read_data(filepath)
f = np.mean

# print(dpm.data['confirm'].value_counts())
t1 = time.time()
para = dpm.query(f, 'Infected', 19/1000)
print("--- query %.4f seconds ---" % (time.time() - t1))
res = para.get('response')
epsilon = para.get('epsilon')
gs = para.get('global sensitivity')
print('Res: {:.4f}, Epsilon: {:.3f}, Global Sensitivity: {:.6f}'.format(res, epsilon, gs))


# MLE adv
t2 = time.time()
dpm.add_adversary(MLEAdversary)
adv = dpm.adv
adv.guess(res, f, epsilon, gs)
print("--- guess %.6f seconds ---" % (time.time() - t2))


# imbalance-class MLE adv
t3 = time.time()
dpm.add_adversary(ImbalanceMLEAdversary)
adv = dpm.adv
adv.guess(res, f, epsilon, gs)
print("--- guess %.6f seconds ---" % (time.time() - t3))