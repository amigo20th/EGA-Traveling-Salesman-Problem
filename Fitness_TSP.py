import numpy as np
import pandas as pd

cities = 50

def eucli_dins(p1, p2):
    return np.sqrt((np.power(p1[0] - p2[0], 2)) + (np.power(p1[1] - p2[1], 2)))

np.random.seed(10)
index = range(cities)
data = np.ndarray(shape=(cities, 2), dtype=np.int16)
for i in range(cities):
    data[i][0] = np.random.randint(0, 50, 1)
    data[i][1] = np.random.randint(0, 50, 1)
df = pd.DataFrame(data, index=index, columns=['x', 'y'])
matrix_dist = np.ndarray(shape=(cities, cities), dtype=np.float)
for row in range(cities):
    for col in range(cities):
        matrix_dist[row][col] = eucli_dins(df.iloc[row], df.iloc[col])
df_dist = pd.DataFrame(data=matrix_dist, index=range(cities), columns=range(cities))

def fitness(arr_way):
    list_dist = []
    for i in range(len(arr_way)-1):
        point1 = arr_way[i]
        point2 = arr_way[i+1]
        list_dist.append(df_dist.iloc[point1, point2])#eucli_dins(df.loc[arr_way[i]], df.loc[arr_way[i + 1]]))
    return sum(list_dist)


def return_points(arr_way):
    list_points = []
    for i in arr_way:
        list_points.append(np.array(df.loc[i]))
    return np.transpose(list_points)



