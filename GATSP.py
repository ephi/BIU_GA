from time import time
import numpy as np
import GAFrameWork
from GAFrameWork import Population
import copy
from matplotlib import pyplot as plt

TSP_CITIES_VALUES_FILE_NAME_CONST = "tsp.txt"
TSP_GA_OUTPUT_FILE = "tsp_output.txt"

NUM_OF_GENERATIONS = 800000 # Probably will stop at the time threshold
NUM_OF_CHROMOSOMES = 200
P_M = 0.5
P_E = 0.15
P_CO = 0.5

RUNTIME_THRESHOLD_IN_SEC = 60 * 15 # 15 Mins
BEST_FITNESS_THRESHOLD = 33900
SHOW_GRAPHS = False
SAVE_GRAPHS = True
INCLUDE_DEBUG_PRINT = True


def generate_all_possibilities(remaining_options: list, current_possibility_arr: list,
                                       all_possibilities_vec: list):
    if len(remaining_options) == 0:
        all_possibilities_vec.append(current_possibility_arr.copy())
        return
    orig_remaining_options = remaining_options.copy()
    for i, option in enumerate(orig_remaining_options):
        current_possibility_arr.append(option)
        remaining_options.pop(i)
        generate_all_possibilities(remaining_options, current_possibility_arr, all_possibilities_vec)
        remaining_options.insert(i, option)
        current_possibility_arr.pop(len(current_possibility_arr) - 1)


def calculate_all_distances(all_cities_arr, all_distances):
    for i in range(all_distances.shape[0]):
        for j in range(all_distances.shape[1]):
            if i == j:
                all_distances[i, j] = np.inf
            else:
                all_distances[i, j] = np.linalg.norm(all_cities_arr[i] - all_cities_arr[j])


class TSPChromosome(GAFrameWork.Chromosome):
    def __init__(self, all_cities_arr: np.ndarray, all_distances: np.ndarray):
        self.chromosome_size = all_cities_arr.shape[0]
        self.cities_indexed_arr = np.zeros(self.chromosome_size, dtype=np.uint8)
        self.distance_from_first_city = np.zeros(self.chromosome_size, dtype=np.float32)
        self.all_cities_arr = all_cities_arr  # An array of x,y tuples.
        self.all_distances = all_distances
        self.full_path_length = 0
        self.temp_float_vector = np.zeros(self.chromosome_size, dtype=np.float32)

    def get_chromosome_size(self):
        return self.chromosome_size

    def get_value(self):
        return self.cities_arr

    def set_value(self, v):
        self.value = v

    def check_for_duplicate_cities(self):  # Only for debugging
        for val in self.cities_indexed_arr:
            assert (np.count_nonzero(self.cities_indexed_arr == val) == 1)

    def get_distance_between_cities(self, index_1, index_2) -> np.float32:
        index_1 = index_1 % self.chromosome_size
        index_2 = index_2 % self.chromosome_size
        return self.all_distances[self.cities_indexed_arr[index_1], self.cities_indexed_arr[index_2]]

    def recalculate_distance_from_first_city(self):
        self.full_path_length = 0
        for index in range(self.chromosome_size):
            self.full_path_length += self.get_distance_between_cities(index, index - 1)

    def randomize_value(self):#, seq_num: int):
        self.cities_indexed_arr  = np.array(range(self.chromosome_size))
        np.random.shuffle(self.cities_indexed_arr)
        self.recalculate_distance_from_first_city()
        return self

    def distance_from_neighbor_cities(self, city_index):
        return self.get_distance_between_cities(city_index, city_index - 1) + self.get_distance_between_cities(
            city_index, city_index + 1)

    def fix_ranges_to_have_same_indexes(self, master_range: np.ndarray, slave_range: np.ndarray):
        indexes_intersection = [index for index in master_range if index in slave_range]
        if (len(indexes_intersection) == 0):
            return False
        slave_indexes_to_change = [(not index in indexes_intersection) for index in slave_range]
        master_indexes_missing_in_slave_range = [index for index in master_range if not index in indexes_intersection]
        master_indexes_to_change = [(not index in indexes_intersection) for index in master_range]
        slave_indexes_missing_in_masetr_range = [index for index in slave_range if not index in indexes_intersection]
        slave_range[slave_indexes_to_change] = master_indexes_missing_in_slave_range
        master_range[master_indexes_to_change] = slave_indexes_missing_in_masetr_range
        return True

    def swap_ranges(self, self_range: np.ndarray, other_chromosome, other_range: np.ndarray, orig_self,
                    orig_other) -> np.bool8:
        assert (self_range[1] - self_range[0] == other_range[1] - other_range[0])
        range_length = self_range[1] - self_range[0] + 1
        temp_sliced_self_indexes_vector = np.zeros(range_length, dtype=np.uint8)
        temp_sliced_other_indexes_vector = np.zeros(range_length, dtype=np.uint8)

        for index_in_temp_vec, index_in_cities_vec in enumerate(range(self_range[0], self_range[1] + 1)):
            temp_sliced_self_indexes_vector[index_in_temp_vec] = orig_self.cities_indexed_arr[
                index_in_cities_vec % self.chromosome_size]

        for index_in_temp_vec, index_in_cities_vec in enumerate(range(other_range[0], other_range[1] + 1)):
            temp_sliced_other_indexes_vector[index_in_temp_vec] = orig_other.cities_indexed_arr[
                index_in_cities_vec % self.chromosome_size]

        need_to_swap = self.fix_ranges_to_have_same_indexes(temp_sliced_self_indexes_vector,
                                                            temp_sliced_other_indexes_vector)
        if need_to_swap:
            for index_in_temp_vec, index_in_cities_vec in enumerate(range(other_range[0], other_range[1] + 1)):
                other_chromosome.cities_indexed_arr[index_in_cities_vec % self.chromosome_size] = \
                    temp_sliced_self_indexes_vector[index_in_temp_vec]
            for index_in_temp_vec, index_in_cities_vec in enumerate(range(self_range[0], self_range[1] + 1)):
                self.cities_indexed_arr[index_in_cities_vec % self.chromosome_size] = temp_sliced_other_indexes_vector[
                    index_in_temp_vec]

        return need_to_swap

    def swap_two_cities(self, indexes_to_swap):
        if indexes_to_swap[0] == indexes_to_swap[1]:
            return
        if indexes_to_swap[0] > indexes_to_swap[1]:
            tmp = indexes_to_swap[1]
            indexes_to_swap[1] = indexes_to_swap[0]
            indexes_to_swap[0] = tmp

        orig_distance_from_neighbor_cities = np.array(
            [self.distance_from_neighbor_cities(city_index) for city_index in indexes_to_swap])

        tmp = self.cities_indexed_arr[indexes_to_swap[1]]
        self.cities_indexed_arr[indexes_to_swap[1]] = self.cities_indexed_arr[indexes_to_swap[0]]
        self.cities_indexed_arr[indexes_to_swap[0]] = tmp

        new_distance_from_neighbor_cities = np.array(
            [self.distance_from_neighbor_cities(city_index) for city_index in indexes_to_swap])
        difference_in_distances = new_distance_from_neighbor_cities - orig_distance_from_neighbor_cities
        self.full_path_length += (difference_in_distances[0] + difference_in_distances[1])

    def __str__(self) -> str:
        return str(self.cities_indexed_arr)


class TSPMutagen(GAFrameWork.Mutagen):
    def __init__(self, p=0.001):
        self.p = p
        self.chromosome_size = 0
        self.chromosome_indexes_range = None

    def mutate(self, chromosome: TSPChromosome, generation: int):
        severity = np.random.rand()
        if self.chromosome_size == 0:
            self.chromosome_size = chromosome.get_chromosome_size()
            self.chromosome_indexes_range = range(self.chromosome_size)
        if severity < self.p:
            # The following for loop is adding another type of mutation.
            # With probability of p^2, the swap mutation will operate twice.
            # For example, if p=0.5, than 50% (= 1-p) there is no mutation and 50% (= p) there is.
            # If there is a mutation, 50% (= 1-p) there is only one cities swap, and 50% (= p) there are two.
            # In conclusion, 1-p - no mutation, p*(1-p) - one swap mutaion, p*p - two swap mutations.
            for i in range(1 + round(severity / (self.p*2))):
                indexes_to_swap = np.random.choice(self.chromosome_indexes_range, size=2, replace=False)
                chromosome.swap_two_cities(indexes_to_swap)


class TSPFitness(GAFrameWork.FitnessObject):
    def eval_fitness(self, chromosome: TSPChromosome) -> np.float32:
        return chromosome.full_path_length

    def get_worst_fitness_value(self):
        return np.inf

    def is_minimization_fitness(self):
        return True


class TSPComputeProbabilities(GAFrameWork.ProbabilitiesComputation):
    def compute_probabilities(self, fitnesses_arr: np.ndarray) -> np.ndarray:
        converted_fitnesses_to_maximizing_problem = [1 / f for f in fitnesses_arr]
        return self.normalize_probabilities(converted_fitnesses_to_maximizing_problem)


def swap_ranges_between_two_chromosomes(chromosome_1: TSPChromosome, chromosome_2: TSPChromosome,
                                        start_index_1: np.uint8, start_index_2: np.uint8,
                                        range_length: np.uint8, orig_self, orig_other):
    range_to_swap_in_parent_1 = np.array([start_index_1, start_index_1 + range_length - 1], dtype=np.uint8)
    range_to_swap_in_parent_2 = np.array([start_index_2, start_index_2 + range_length - 1], dtype=np.uint8)
    return chromosome_1.swap_ranges(range_to_swap_in_parent_1, chromosome_2, range_to_swap_in_parent_2, orig_self,
                                    orig_other)


def tsp_crossover(chromosome_1: TSPChromosome, chromosome_2: TSPChromosome, generation: int):
    new_child_1: TSPChromosome = copy.deepcopy(chromosome_1)
    new_child_2: TSPChromosome = copy.deepcopy(chromosome_2)

    need_to_swap = False
    chromosome_size = chromosome_1.get_chromosome_size()
    range_length_to_swap = np.random.randint(2, chromosome_size / 2.0, dtype=np.uint8)
    range_start_1 = np.random.randint(chromosome_size, dtype=np.uint8)
    range_start_2 = np.random.randint(chromosome_size, dtype=np.uint8)
    range_end_1 = (range_start_1 + range_length_to_swap) % chromosome_size

    assert (range_length_to_swap >= 0)

    if range_start_1 != range_end_1:
        need_to_swap = need_to_swap or swap_ranges_between_two_chromosomes(new_child_1, new_child_2, range_start_1,
                                                                           range_start_2, range_length_to_swap,
                                                                           chromosome_1, chromosome_2)

    if need_to_swap:
        new_child_1.recalculate_distance_from_first_city()
        new_child_2.recalculate_distance_from_first_city()
    return new_child_1, new_child_2


if __name__ == '__main__':

    np.random.seed(2019)

    cities_str_lines = [""]
    with open(TSP_CITIES_VALUES_FILE_NAME_CONST, "r+") as tsp_cities_file_obj:
        # Reading form a file
        cities_str_lines = tsp_cities_file_obj.readlines()

    reduced_cities_str_lines = [city for city in cities_str_lines if len(city)>0]
    number_of_cities = len(reduced_cities_str_lines)
    all_cities_arr = np.ndarray(shape=(number_of_cities, 2), dtype=np.int32)
    for i, city_line in enumerate(reduced_cities_str_lines):
        city_seperated = city_line.split()
        if len(city_seperated) == 2:  # To skip empty lines
            all_cities_arr[i, :] = [float(str_val) for str_val in city_seperated]

    all_distances = np.ndarray((number_of_cities, number_of_cities), dtype=np.float32)
    calculate_all_distances(all_cities_arr, all_distances)
    randomized_chromosomes = [
        TSPChromosome(all_cities_arr=all_cities_arr, all_distances=all_distances).randomize_value()#i + 100) 
            for i in range(NUM_OF_CHROMOSOMES)]

    tsp_population = Population(randomized_chromosomes.copy(), fitness_obj=TSPFitness(),
                            crossover_func=tsp_crossover,
                            probabilities_computation_obj=TSPComputeProbabilities(),
                            chromosome_type=TSPChromosome,
                            mutagen=TSPMutagen(P_M),
                            elitism_percentage=P_E,
                            cross_over_probability=P_CO)
    sum_time = 0
    delta_time = 0
    if SHOW_GRAPHS or SAVE_GRAPHS:
        best_fitness_history = []
        average_fitness_history = []
    for i in range(NUM_OF_GENERATIONS):
        start_time = time()
        tsp_population.evolve()
        end_time = time()
        best_fitness = tsp_population.get_best_fitness()
        if SHOW_GRAPHS or SAVE_GRAPHS:
            best_fitness_history.append(best_fitness)
            average_fitness_history.append(tsp_population.get_average_fitness())
        delta_time = end_time - start_time
        sum_time += delta_time
        if best_fitness < BEST_FITNESS_THRESHOLD:
            break
        if sum_time > RUNTIME_THRESHOLD_IN_SEC:
            break
    if INCLUDE_DEBUG_PRINT:
        print(f"GA completed after {sum_time} seconds, best fitness is {tsp_population.get_best_fitness()}")

    if SHOW_GRAPHS or SAVE_GRAPHS:
        plt.plot(best_fitness_history, label='Best')
        plt.plot(average_fitness_history, label="Average")
        # plt.plot(median_fitness_history, label="Median")
        plt.title(f"N={NUM_OF_CHROMOSOMES}, Gen={tsp_population.generation_num}, P_M={P_M}, P_E={P_E}, P_CO={P_CO}, RT={sum_time}")
        plt.xlabel("Generation")
        plt.ylabel('Fitness')
        plt.legend()
        
        if SAVE_GRAPHS:
            plt.savefig(f"N={NUM_OF_CHROMOSOMES}, Gen={tsp_population.generation_num}, P_M={P_M}, P_E={P_E}, P_CO={P_CO}, RT={sum_time}.png", bbox_inches='tight')
        if SHOW_GRAPHS:
            plt.show()
    
    best_chromosome : TSPChromosome = tsp_population.get_best_chromosome()
    best_chromosome.cities_indexed_arr += 1

    output_file_name = TSP_GA_OUTPUT_FILE
    with open(output_file_name, "w+") as o_file:
        # Reading form a file
        o_file.write("\n".join([str(city_index) for city_index in best_chromosome.cities_indexed_arr]))
        
