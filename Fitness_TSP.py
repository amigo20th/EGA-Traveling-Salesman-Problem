import numpy as np
import pandas as pd
from collections import Counter
np.random.seed(10)
index = range(10)
#data = [(np.random.randint(0, 100), np.random.randint(0, 100)) for x in index]
data = [[ 9, 15], [64, 28], [89, 93], [29,  8], [73,  0], [40, 36], [16, 11], [54, 88], [62, 33], [72, 78]]
df = pd.DataFrame(data, index=index, columns=['x', 'y'])

def eucli_dins(p1, p2):
    return np.sqrt ((np.power(p1[0]-p2[0], 2)) + (np.power(p1[1]-p2[1], 2)))

def fitness(arr_points, n):
    list_dist = []
    for i in range(n-1):
        list_dist.append(eucli_dins(df.loc[i], df.loc[i+1]))
    return sum(list_dist)


print(df.values)


