import numpy as np
import time
import GAFrameWork
from GAFrameWork import Population
from matplotlib import pyplot as plt

TSP_CITIES_VALUES_FILE_NAME_CONST = "tsp.txt"
NUMBER_OF_CITIES = 48
NUMBER_OF_COORDS = 2
NUM_OF_GENERATIONS = 40000
NUM_OF_CHROMOSOMES = 3


class TSPChromosome(GAFrameWork.Chromosome):
    def __init__(self, tsp_cities, create_copy=True, route_distance=float("inf")):
        self.chromosome_size = tsp_cities.shape[0]
        if create_copy:
            self.tsp_cities = tsp_cities.copy()
        else:
            self.tsp_cities = tsp_cities
        self.route_distance = route_distance

    def get_chromosome_size(self):
        return self.chromosome_size

    def get_value(self):
        return self.tsp_cities

    def set_route_distance(self, rd):
        self.route_distance = rd

    def get_route_distance(self):
        return self.route_distance

    def set_value(self, v):
        self.tsp_cities = v

    def randomize_value(self):
        np.random.shuffle(self.tsp_cities)
        return self

    def __str__(self) -> str:
        return str(self.tsp_cities)


class TSPMutagen(GAFrameWork.Mutagen):
    def __init__(self, p=0.001):
        self.p = p
        self.chromosome_size = 0
        self.chromosome_indexes_range = None

    def mutate(self, chromosome: TSPChromosome, generation: int):
        severity = np.random.rand()
        modified_p = 0.1 + (self.p - 0.1) * (1 - (generation / NUM_OF_GENERATIONS) ** 2)
        if self.chromosome_size == 0:
            self.chromosome_size = chromosome.get_chromosome_size()
            self.chromosome_indexes_range = range(self.chromosome_size)
        if severity < self.p:
            # for i in range(1 + round(severity / self.p)):
            indexes_to_swap = np.random.choice(self.chromosome_indexes_range, size=2, replace=False)
            tmp = np.copy(chromosome.get_value()[indexes_to_swap[0], :])
            chromosome.get_value()[indexes_to_swap[0], :] = chromosome.get_value()[indexes_to_swap[1], :]
            chromosome.get_value()[indexes_to_swap[1], :] = tmp
            chromosome.set_route_distance(float("inf"))


class TSPFitness(GAFrameWork.FitnessObject):
    def eval_fitness(self, chromosome: TSPChromosome) -> np.float32:
        rd = chromosome.get_route_distance()
        if rd < float("inf"):
            return rd
        m = chromosome.get_value()
        dm = create_diffetence_matrix(np.delete(m, -1, axis=-1))
        p2_dm = np.power(dm, 2)
        s_dm = np.sqrt(np.sum(p2_dm, axis=1))
        # a_dm = np.abs(dm)
        s_dm = np.sum(s_dm)
        chromosome.set_route_distance(s_dm)
        return s_dm


def tsp_crossover(chromosome_1: TSPChromosome, chromosome_2: TSPChromosome, generation: int):
    allowed_child_1 = list(range(1, NUMBER_OF_CITIES + 1))
    allowed_child_2 = list(range(1, NUMBER_OF_CITIES + 1))
    c_m1 = chromosome_1.get_value()
    c_m2 = chromosome_2.get_value()
    child_1_rd = chromosome_1.get_route_distance()
    child_2_rd = chromosome_2.get_route_distance()
    child_1 = np.zeros((NUMBER_OF_CITIES, NUMBER_OF_COORDS + 1), dtype=np.int32)
    child_2 = np.zeros((NUMBER_OF_CITIES, NUMBER_OF_COORDS + 1), dtype=np.int32)

    def get_valid_city(a, b, c, d, e=None):
        if a[-1] not in b:
            selected = np.random.choice(b, replace=False)
            msk = c[:, -1] == selected
            if e is not None:
                e[0] = float("inf")
            if np.any(msk):
                r = c[msk, :].flatten()
                return r
            msk = d[:, -1] == selected
            if np.any(msk):
                return d[msk, :].flatten()
            raise ValueError("was not able to find a valid city")
        return a

    for i in range(NUMBER_OF_CITIES):
        p = np.random.uniform()
        if p < 0.5:
            r_cm2 = np.copy(c_m1[i, :])
            r_cm1 = np.copy(c_m2[i, :])
            r_cm1 = get_valid_city(r_cm1, allowed_child_1, c_m2, c_m1)
            r_cm2 = get_valid_city(r_cm2, allowed_child_2, c_m1, c_m2)
            child_1_rd = float("inf")
            child_2_rd = float("inf")
        else:
            r_cm1 = np.copy(c_m1[i, :])
            r_cm2 = np.copy(c_m2[i, :])
            r_cm2 = get_valid_city(r_cm2, allowed_child_2, c_m1, c_m2, [child_2_rd])
            r_cm1 = get_valid_city(r_cm1, allowed_child_1, c_m2, c_m1, [child_1_rd])
        allowed_child_1.remove(r_cm1[-1])
        allowed_child_2.remove(r_cm2[-1])
        child_1[i] = r_cm1
        child_2[i] = r_cm2

    return TSPChromosome(child_1, create_copy=False, route_distance=child_1_rd), \
           TSPChromosome(child_2, create_copy=False, route_distance=child_2_rd)


def create_diffetence_matrix(m):
    dm = np.diff(m, axis=0)
    dm = np.vstack([dm, m[0, :] - m[-1, :]])
    return dm


if __name__ == '__main__':
    cities_coord_matrix = np.zeros((NUMBER_OF_CITIES, NUMBER_OF_COORDS + 1), dtype=np.int32)
    with open(TSP_CITIES_VALUES_FILE_NAME_CONST, "r+") as f:
        # Reading form a file
        i = 0
        for line in f.readlines():
            coords = line.strip().split()
            for j in range(len(coords)):
                cities_coord_matrix[i][j] = int(coords[j])
            cities_coord_matrix[i][-1] = int(i + 1)
            i += 1

    randomized_chromosomes = [
        TSPChromosome(tsp_cities=cities_coord_matrix).randomize_value() for i in
        range(NUM_OF_CHROMOSOMES)]
    tsp_population = Population(randomized_chromosomes, fitness_obj=TSPFitness(),
                                crossover_func=tsp_crossover,
                                probabilities_computation_obj=GAFrameWork.MinimizationProblemComputeProbabilities(),
                                mutagen=TSPMutagen(0.01 / NUM_OF_CHROMOSOMES), cross_over_probability=0.18)
    best_fitness_history = []
    average_fitness_history = []
    #median_fitness_history = []
    start = time.perf_counter()
    for i in range(0, NUM_OF_GENERATIONS):
        best_fitness = tsp_population.get_best_fitness()
        average_fitness = tsp_population.get_average_fitness()
        #median_fitness = tsp_population.get_median_fitness()
        best_fitness_history.append(best_fitness)
        average_fitness_history.append(average_fitness)
        #median_fitness_history.append(median_fitness)
        tsp_population.evolve()
        if i % 100 == 0:
            print(
                "Finished Generetion #{0} \t best = {1} \t average = {2}".format(i + 1, best_fitness, average_fitness))
    end = time.perf_counter()
    print(f"GA took {end-start} seconds to complete")

    plt.plot(best_fitness_history)
    plt.plot(average_fitness_history)
    #plt.plot(median_fitness_history)
    plt.xlabel("Generation")
    plt.ylabel('Fitness')
    plt.show()
