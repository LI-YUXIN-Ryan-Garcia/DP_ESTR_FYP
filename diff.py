import pandas as pd
import numpy as np

class Differential_privacy():
    def __init__(self, family_infect_rate, all_infect_rate, epsion):
        self.family_infect_rate = family_infect_rate
        self.all_infect_rate = all_infect_rate
        self.epsion = epsion
        self.population = pd.read_csv('data.csv')

    def infect(self, patient_fam, family):
        fam_rate = self.family_infect_rate
        all_rate = self.all_infect_rate
        rate = fam_rate if family == patient_fam else all_rate
        return 1 if np.random.rand() <= rate else 0
        
    def propagate(self):
        ppl = self.population
        pop_amount = ppl.shape[0]

        patient = np.random.randint(1,pop_amount+1)
        patient_fam = ppl.at[patient, 'family_id']
        print('patient: {}, patient famil: {}'.format(patient, patient_fam))

        infected = np.array(list(map(self.infect, [patient_fam]*pop_amount, ppl.loc[:,'family_id'])))
        print('infected people: {}'.format(infected))
        self.infected_num = np.sum(infected)
        return infected


dp = Differential_privacy(0.8, 0.1, 0.1)
infected = dp.propagate()
inf_num = np.sum(infected) 
print('infected number: {}'.format(inf_num))

loc, scale = 0., 1.
sample = np.random.laplace(loc, scale, 1000)
noise = np.random.choice(sample)
print('noise: {}'.format(noise))
print('noisy infected number: {}'.format(inf_num + noise))

