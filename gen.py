"""
This file generates datasets
"""
import os
import numpy as np
import pandas as pd

data_path = os.path.join(os.getcwd(), 'dataset')

def gen_toy_dataset():
    records = {
        'Infected': [1,0,1,0,1],
        'Ages': [30,20,25,50,35],
        'Family': [8,110, 312, 8, 8]
    }
    names = ["Infected", "Ages", "Family"]
    df = pd.DataFrame(records, columns=names)
    df.index += 1
    df.index.name = 'ID'
    print("A toy dataset")
    print(df)
    df.to_csv(os.path.join(data_path, 'toy.csv'))


def gen_sensitive_dataset():
    age = {
        "Ages_S" : [37,20,24,35,22,34,30,26,40,37,22,39,33,28,24,25,26,22,27] + [80],
        "Ages_NS" : [37,20,24,35,22,34,30,26,40,37,22,39,33,28,24,25,26,22,27] + [31]
    }
    names = ['Ages_S','Ages_NS']
    df = pd.DataFrame(age, columns=names)
    df.index += 1
    df.index.name = 'ID'
    print('Sensitive Date')
    print(df)
    df.to_csv(os.path.join(data_path, 'ages.csv'))


def gen_dataset_0():
    # family and ages
    fam_config = {1:100, 2:100, 3:200, 4:5, 5:14, 10:1}
    fam_age = {
        1 : [30], 
        2 : [22, 25], 
        3 : [7,37,42], 
        4 : [3,15,45,52], 
        5 : [2, 30, 35, 63, 70],
        10: [2, 30, 35, 63, 70, 2, 30, 35, 63, 70]}
    fam, age = [0] * 1000, [0] * 1000
    fam_id, count = 1, 0
    for k, v in fam_config.items():
        for index in range(count, count+k*v, k):
            for j in range(k):
                fam[index+j] = fam_id
            age[index:index+k] = fam_age[k]   
            fam_id += 1
        count += k*v

    # manually select confirm case
    confirm = [0] * 990 + [1] * 10
    infected = np.array([7, 124, 222, 300, 489, 537, 624, 734, 886, 929], dtype=int)
    infected = np.append(infected, np.arange(990,1000))
    infected_age = [37,20,24,35,22,34,30,26,40,37,22,39,33,28,24,25,26,22,27,80]
    for i in range(len(infected)):
        ID = infected[i]
        confirm[ID] = 1
        age[ID] = infected_age[i]

    #generate CSV file:
    names = ['Infected', 'Ages', 'Family']
    data = {
        'Infected': confirm,
        'Ages': age,
        'Family': fam
    }

    df = pd.DataFrame(data, columns=names)
    df.index += 1
    df.index.name = 'ID'
    df.to_csv(os.path.join(data_path, 'dataset0.csv'))


if __name__ == '__main__':
    gen_toy_dataset()
    gen_sensitive_dataset()
    gen_dataset_0()