import gen
import pandas as pd

population  = 100
family = 8
g = gen.Gen(population, family)
# g.gen()
g.unif()

# df = pd.read_csv('data.csv')
# print(df.loc[:,'family_id'].value_counts())

