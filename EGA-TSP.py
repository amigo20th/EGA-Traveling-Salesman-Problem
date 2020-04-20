import numpy as np
import Fitness_TSP
import matplotlib.pyplot as plt



def genInitPop(individuals, var):
    # This function create the first population
    list_cities = list(range(var))
    tmp_pop = np.ndarray(shape=(individuals, var), dtype=np.int16)
    for ind in range(individuals):
        np.random.shuffle(list_cities)
        tmp_pop[ind] = list_cities
    return tmp_pop


def crossover(population, individuals, cities, Pc):
    pop_tmp = np.copy(population)
    for i in range(int(individuals / 2)):
        cross_prob = np.random.random(1)
        if (cross_prob <= Pc):
            c1 = np.random.randint(0, cities)
            c2 = np.random.randint(0, cities)
            if c2 < c1:
                c1, c2 = c2, c1
            arr_tmp1 = list(population[i])
            arr_tmp2 = list(population[n - i - 1])
            subset_spring1 = arr_tmp1[c1:c2]
            subset_spring2 = arr_tmp2[c1:c2]
            way2 = np.ndarray(shape=(1, cities), dtype=np.int16)
            way2 = way2[0]
            way2[:] = -1
            way2[c1:c2] = subset_spring1
            way1 = np.ndarray(shape=(1, cities), dtype=np.int16)
            way1 = way1[0]
            way1[:] = -1
            way1[c1:c2] = subset_spring2
            for k in subset_spring1:
                arr_tmp2.remove(k)
            for k in subset_spring2:
                arr_tmp1.remove(k)
            count = 0
            for a in range(0, c1):
                way2[a] = arr_tmp2[a]
                count += 1
            for c in range(c2, n_vars):
                way2[c] = arr_tmp2[count]
                count += 1
            count1 = 0
            for a in range(0, c1):
                way1[a] = arr_tmp1[a]
                count1 += 1
            for c in range(c2, n_vars):
                way1[c] = arr_tmp1[count1]
                count1 += 1
            pop_tmp[i] = way2
            pop_tmp[n - i - 1] = way1
    return pop_tmp


def mutation(I_double, n, n_var, B2M):
    I_tmp = np.copy(I_double)
    for count in range(B2M):
        f1 = int(np.random.randint(0, n))
        c1 = int(np.random.randint(0, n_vars))
        c2 = int(np.random.randint(0, n_vars))
        I_tmp[f1][c1], I_tmp[f1][c2] = I_tmp[f1][c2], I_tmp[f1][c1]
    return I_tmp

def mul_last_city(I_double, n, n_var, Pm):
    tmp = np.random.random(1)
    I_tmp = np.copy(I_double)
    #if tmp < Pm:
    chan = np.random.randint(0, n_var-1)
    I_tmp[n-1] = I_tmp[0]
    I_tmp[n-1][n_var-1], I_tmp[n-1][chan] = I_tmp[n-1][chan], I_tmp[n-1][n_var-1]
    print("Entro en mutacion de ultimo, cambio {} por este {}".format(I_tmp[n-1][n_var-1], I_tmp[n-1][chan]))
    return I_tmp

### Variables for flexibility of the algorithm
# number of variables for one solution
# in this case, we have the number of cities
n_vars = 50
## Variables what EGA needs
# Number of generations
G = 2000
# Number of individuals
n = 50
# Length of chromosome
L = n_vars
# Population
I = np.ndarray(shape=(n, n_vars), dtype=np.int16)
# Crossing probability
Pc = 0.9
# Mutation probability
Pm = 0.028
# list of fitness
fitness = np.ndarray(shape=(2, n), dtype=float)
# Expected number of mutations for each generation
B2M = int(n * L * Pm)

### Auxiliary variables
# Double of the population
I_double = np.ndarray(shape=(2 * n, n_vars), dtype=np.int16)
# double of the list of fitness
fitness_double = np.ndarray(shape=(2, 2 * n), dtype=float)

# Initial population
I = genInitPop(n, n_vars)

for gen in range(G):
    # Double of length of the population
    I_double = np.concatenate((I, I), axis=0)

    # Apply Annular Crossover
    I_double = crossover(I_double, n, n_vars, Pc)

    # Apply Mutation
    I_double = mutation(I_double, len(I_double), n_vars, B2M)

    # Apply fitness
    count = 0
    for i in range(fitness_double.shape[1]):
        fitness_double[0][i] = count
        fitness_double[1][i] = Fitness_TSP.fitness(I_double[i])
        count += 1
    # Order by fitness
    fitness_double = fitness_double[:, fitness_double[1].argsort()]


    # Apply Elitism
    ind_eli = fitness_double[0][0:n]
    ind_eli = np.array(ind_eli, dtype=np.int16)
    count = 0
    for i in ind_eli:
        I[count] = I_double[i]
        count += 1
    print("best way {}, Best Travel Cost= {}".format(list(I[0]), fitness_double[1][0]))

    # Apply mutation in the last city
    I = mul_last_city(I, n, n_vars, Pm)

way = list(I[0])
way.append(way[0])
print("Aproaches: ")
print(list(way), fitness_double[1][0])
print()

points = Fitness_TSP.return_points(way)

plt.plot(points[0], points[1])
plt.scatter(points[0], points[1], c='red')
for inx, poi  in enumerate(I[0]):
    plt.annotate(poi, (points[0][inx]+0.3, points[1][inx]+0.3))
plt.show()

