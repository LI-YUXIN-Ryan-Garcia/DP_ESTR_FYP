import csv
import random

class Gen():
    def __init__(self, population, family):
        self.pl = population
        self.fl = family

    def gen(self):
        items = ['id', 'family_id']
        pl = self.pl
        fl = self.fl
        with open('data.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(items)
            for i in range(1,pl):
                family = random.randint(1,fl)
                writer.writerow([i,family])

        

