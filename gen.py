import csv
import random
import numpy as np

size  = 1000

# family
fam_config = {1:100, 2:100, 3:200, 4:5, 5:14, 10:1}
fam = [0] * 1000
fam_id = 1
count = 0
for k, v in fam_config.items():
    for index in range(count, count+k*v, k):
        for j in range(k):
            fam[index+j] = fam_id
        fam_id += 1
    count += k*v

# confirm cases
confirm = []
count = 990
confirm += [0] * 990 + [1] * 10

# manual selection
#infected = [7, 124, 222, 489, 537, 624, 734, 886, 929] + [300, 301, 302, 303, 304, 305]
infected = [7, 124, 222, 300, 489, 537, 624, 734, 886, 929]
for i in infected:
    confirm[i] = 1

# randomize(optional)
shuffled = list(zip(fam, confirm))
random.shuffle(shuffled)

#generate CSV file:
items = ['id', 'family', 'confirm']
with open('data.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(items)
    for i in range(1, size+1):
        writer.writerow([i, shuffled[i-1][0], shuffled[i-1][1]])

age1 = [37,20,24,35,22,34,30,26,40,37,22,39,33,28,24,25,26,22,27] + [80]
age2 = [37,20,24,35,22,34,30,26,40,37,22,39,33,28,24,25,26,22,27] + [31]
items = ['id', 'age1', 'age2']
with open('age.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(items)
    for i in range(1, len(age1)+1):
        writer.writerow([i, age1[i-1], age2[i-1]])
