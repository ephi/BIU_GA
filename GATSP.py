from math import ceil
import numpy as np

import multiprocessing as mp

import GAFrameWork
from GAFrameWork import Population
from matplotlib import pyplot as plt
import copy
from enum import IntEnum


class DebugChecksEnum(IntEnum):
    DISTANCE_CHECKS_AND_ASSERTS = 0,


TSP_CITIES_VALUES_FILE_NAME_CONST = "tsp.txt"
INCLUDE_DEBUG_PRINT = False
debug_checks_dict = {

}

NUM_OF_GENERATIONS = 6000
NUM_OF_CHROMOSOMES = 200
NUM_OF_CITIES = 48


def generate_all_sorting_possibilities(remaining_options: list, current_possibility_arr: list,
                                       all_possibilities_vec: list[list]):
    if len(remaining_options) == 0:
        # print(current_possibility_arr)
        all_possibilities_vec.append(current_possibility_arr.copy())
        return
    orig_remaining_options = remaining_options.copy()
    for i, option in enumerate(orig_remaining_options):
        current_possibility_arr.append(option)
        remaining_options.pop(i)
        generate_all_sorting_possibilities(remaining_options, current_possibility_arr, all_possibilities_vec)
        remaining_options.insert(i, option)
        current_possibility_arr.pop(len(current_possibility_arr) - 1)


class TSPChromosome(GAFrameWork.Chromosome):
    def __init__(self, all_cities_arr: np.ndarray, all_distances: np.ndarray):
        self.chromosome_size = all_cities_arr.shape[0]
        self.cities_indexed_arr = np.zeros(self.chromosome_size, dtype=np.uint8)
        self.distance_from_first_city = np.zeros(self.chromosome_size, dtype=np.float32)
        self.all_cities_arr = all_cities_arr  # An array of x,y tuples.
        self.temp_float_vector = np.zeros(self.chromosome_size, dtype=np.float32)
        self.all_distances = all_distances

    def get_chromosome_size(self):
        return self.chromosome_size

    def get_value(self):
        return self.cities_arr

    def set_value(self, v):
        self.value = v

    def calc_distance(self, city_1_index_in_ch, city_2_index_in_ch):
        # city_1 = self.all_cities_arr[self.cities_indexed_arr[city_1_index_in_ch]]
        # city_2 = self.all_cities_arr[self.cities_indexed_arr[city_2_index_in_ch]]
        # return (city_1[0] - city_2[0]) ** 2 + (city_1[1] - city_2[1]) ** 2
        return np.linalg.norm(self.all_cities_arr[self.cities_indexed_arr[city_1_index_in_ch]] - self.all_cities_arr[
            self.cities_indexed_arr[city_2_index_in_ch]])

    def recalculate_distance_from_first_city(self, indexes=None):
        if indexes is None:
            indexes = range(self.chromosome_size)

        for index in indexes:
            if index == 0:
                self.distance_from_first_city[index] = 0
            else:
                self.distance_from_first_city[index] = self.distance_from_first_city[index - 1] + \
                                                       self.all_distances[
                                                           self.cities_indexed_arr[index], self.cities_indexed_arr[
                                                               index - 1]]
                # self.calc_distance(index, index-1)

    def randomize_value(self):
        all_cities_indexes_reduced = range(self.chromosome_size)
        rand_city_places = np.random.randint(0, range(self.chromosome_size, 0, -1))

        for i, rand_city_place in enumerate(rand_city_places):
            self.cities_indexed_arr[i] = all_cities_indexes_reduced[rand_city_place - 1]
            all_cities_indexes_reduced = np.delete(all_cities_indexes_reduced, rand_city_place - 1)

        self.recalculate_distance_from_first_city()
        # chosen_cities_arr = self.all_cities_arr[self.cities_indexed_arr]

        # diff_arr = np.diff(chosen_cities_arr, axis=0)
        # squared_distances = np.sum(diff_arr * diff_arr, axis=1)
        # for i, val in enumerate(squared_distances):
        #     self.distance_from_first_city[i+1] = self.distance_from_first_city[i] + val
        # for i, (prev_city, current_city) in enumerate(zip(chosen_cities_arr[:-1], chosen_cities_arr[1:])):
        #     self.distance_from_first_city[i+1] = self.distance_from_first_city[i] + np.linalg.norm(prev_city - current_city)

        return self

    def distance_from_neighbor_cities(self, city_index):
        if city_index == 0:
            return self.distance_from_first_city[1]
        chromosome_size = self.get_chromosome_size()
        if city_index == chromosome_size - 1:
            return self.distance_from_first_city[city_index] - self.distance_from_first_city[city_index - 1]

        return self.distance_from_first_city[city_index + 1] - self.distance_from_first_city[city_index - 1]

    def update_distances_after_range_changed(self, range_changed: np.ndarray):
        if range_changed[0] == range_changed[1]:
            return
        self.temp_float_vector[:] = 0

        orig_distance_from_first_city_of_the_city_after_the_range = 0
        new_distance_from_first_city_of_the_city_after_the_range = 0
        if range_changed[1] + 1 < self.chromosome_size - 1:
            orig_distance_from_first_city_of_the_city_after_the_range = self.distance_from_first_city[
                range_changed[1] + 1]
        self.recalculate_distance_from_first_city(
            range(range_changed[0], min(range_changed[1] + 2, self.chromosome_size)))
        if range_changed[1] + 1 < self.chromosome_size - 1:
            new_distance_from_first_city_of_the_city_after_the_range = self.distance_from_first_city[
                range_changed[1] + 1]
            after_range_distance_difference = new_distance_from_first_city_of_the_city_after_the_range - orig_distance_from_first_city_of_the_city_after_the_range
            self.temp_float_vector[range_changed[1] + 2:] = after_range_distance_difference
            self.distance_from_first_city += self.temp_float_vector

    def fix_ranges_to_have_same_indexes(self, master_range: np.ndarray, slave_range: np.ndarray):
        new_slave_indexes = slave_range.copy()
        slave_indexes_to_change = [(not index in master_range) for index in slave_range]
        missing_indexes = [index for index in master_range if not index in slave_range]
        new_slave_indexes[slave_indexes_to_change] = missing_indexes
        new_master_indexes = master_range.copy()
        master_indexes_to_change = [(not index in slave_range) for index in master_range]
        missing_indexes = [index for index in slave_range if not index in master_range]
        new_master_indexes[master_indexes_to_change] = missing_indexes
        return new_master_indexes, new_slave_indexes

    def swap_ranges(self, self_range: np.ndarray, other_chromosome, other_range: np.ndarray, orig_self,
                    orig_other) -> np.bool8:
        assert (self_range[1] - self_range[0] == other_range[1] - other_range[0])
        range_length = self_range[1] - self_range[0] + 1
        temp_sliced_self_indexes_vector = np.zeros(range_length, dtype=np.uint8)
        temp_sliced_other_indexes_vector = np.zeros(range_length, dtype=np.uint8)
        temp_sliced_self_indexes_vector[:] = orig_self.cities_indexed_arr[self_range[0]: self_range[1] + 1]
        temp_sliced_other_indexes_vector[:] = orig_other.cities_indexed_arr[other_range[0]: other_range[1] + 1]

        temp_sliced_self_indexes_vector, temp_sliced_other_indexes_vector = self.fix_ranges_to_have_same_indexes(
            temp_sliced_self_indexes_vector, temp_sliced_other_indexes_vector)

        self.cities_indexed_arr[self_range[0]: self_range[1] + 1] = temp_sliced_other_indexes_vector[:]
        other_chromosome.cities_indexed_arr[other_range[0]: other_range[1] + 1] = temp_sliced_self_indexes_vector[:]

        self.update_distances_after_range_changed(self_range)
        other_chromosome.update_distances_after_range_changed(other_range)

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

        if indexes_to_swap[1] - indexes_to_swap[0] == 1:
            self.recalculate_distance_from_first_city(
                np.linspace(start=indexes_to_swap[0], stop=min(indexes_to_swap[1] + 1, self.chromosome_size - 1), num=3,
                            dtype=np.uint8))
        else:
            for index in indexes_to_swap:
                self.recalculate_distance_from_first_city(
                    np.linspace(start=index, stop=min(index + 1, self.chromosome_size - 1), num=2, dtype=np.uint8))

        new_distance_from_neighbor_cities = np.array(
            [self.distance_from_neighbor_cities(city_index) for city_index in indexes_to_swap])
        difference_in_distances = new_distance_from_neighbor_cities - orig_distance_from_neighbor_cities
        self.temp_float_vector[:] = 0
        if indexes_to_swap[1] - indexes_to_swap[0] > 2:
            self.temp_float_vector[indexes_to_swap[0] + 2:] += difference_in_distances[0]
        else:
            self.temp_float_vector[indexes_to_swap[1] + 2:] += difference_in_distances[0]
        if indexes_to_swap[1] < self.chromosome_size - 1:
            self.temp_float_vector[indexes_to_swap[1] + 2:] += difference_in_distances[1]
        self.distance_from_first_city += self.temp_float_vector

        # orig_distances = self.distance_from_first_city.copy()
        # self.recalculate_distance_from_first_city(range(self.chromosome_size))
        # diff = orig_distances - self.distance_from_first_city
        # assert(abs(max(diff) - min(diff)) < 0.1)

    def __str__(self) -> str:
        return str(self.cities_indexed_arr)


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
            chromosome.swap_two_cities(indexes_to_swap)


class TSPFitness(GAFrameWork.FitnessObject):
    def eval_fitness(self, chromosome: TSPChromosome) -> np.float32:
        return chromosome.distance_from_first_city[chromosome.get_chromosome_size() - 1]





def swap_ranges_between_two_chromosomes(chromosome_1: TSPChromosome, chromosome_2: TSPChromosome,
                                        start_index_1: np.uint8, start_index_2: np.uint8,
                                        range_length: np.uint8, orig_self, orig_other):
    range_to_swap_in_parent_1 = np.array([start_index_1, start_index_1 + range_length - 1], dtype=np.uint8)
    range_to_swap_in_parent_2 = np.array([start_index_2, start_index_2 + range_length - 1], dtype=np.uint8)
    chromosome_1.swap_ranges(range_to_swap_in_parent_1, chromosome_2, range_to_swap_in_parent_2, orig_self, orig_other)


def tsp_crossover(chromosome_1: TSPChromosome, chromosome_2: TSPChromosome, generation: int):
    new_child_1: TSPChromosome = copy.deepcopy(chromosome_1)
    new_child_2: TSPChromosome = copy.deepcopy(chromosome_2)

    chromosome_size = chromosome_1.get_chromosome_size()

    range_length_to_swap = np.random.randint(2, 2 + ceil(
        (chromosome_size / 2 - 2) * ((NUM_OF_GENERATIONS - generation) / NUM_OF_GENERATIONS)), dtype=np.uint8)

    range_start_1 = np.random.randint(chromosome_size, dtype=np.uint8)
    range_start_2 = np.random.randint(chromosome_size, dtype=np.uint8)
    range_end_1 = (range_start_1 + range_length_to_swap) % chromosome_size
    range_end_2 = (range_start_2 + range_length_to_swap) % chromosome_size
    need_to_be_splited = max(range_start_1, range_start_2) + range_length_to_swap >= chromosome_size
    # print("range_length_to_swap = {0}".format(range_length_to_swap))
    # print("range_start_1 = {0}, range_end_1 = {1}".format(range_start_1, range_end_1))
    # print("range_start_2 = {0}, range_end_2 = {1}".format(range_start_2, range_end_2))
    # print("need_to_be_splited = {0}".format(need_to_be_splited))
    while need_to_be_splited:
        later_start = max(range_start_1, range_start_2)
        current_length_from_start = chromosome_size - later_start
        swap_ranges_between_two_chromosomes(new_child_1, new_child_2, range_start_1, range_start_2,
                                            current_length_from_start, chromosome_1, chromosome_2)
        range_start_1 = (range_start_1 + current_length_from_start) % chromosome_size
        range_start_2 = (range_start_2 + current_length_from_start) % chromosome_size
        range_length_to_swap -= current_length_from_start
        # Now one of them should be zero and one should be close to end

        need_to_be_splited = max(range_start_1, range_start_2) + range_length_to_swap >= chromosome_size
        # print("range_length_to_swap = {0}".format(range_length_to_swap))
        # print("range_start_1 = {0}, range_end_1 = {1}".format(range_start_1, range_end_1))
        # print("range_start_2 = {0}, range_end_2 = {1}".format(range_start_2, range_end_2))
        # print("need_to_be_splited = {0}".format(need_to_be_splited))

    assert (range_length_to_swap >= 0)
    assert (range_length_to_swap == range_end_1 - range_start_1)
    assert (range_length_to_swap == range_end_2 - range_start_2)
    assert (range_end_2 >= range_start_2)
    assert (range_end_1 >= range_start_1)

    if range_start_1 != range_end_1:
        swap_ranges_between_two_chromosomes(new_child_1, new_child_2, range_start_1, range_start_2,
                                            range_length_to_swap, chromosome_1, chromosome_2)

    return new_child_1, new_child_2


if __name__ == '__main__':

    cities_str_lines = [""]
    with open(TSP_CITIES_VALUES_FILE_NAME_CONST, "r+") as tsp_cities_file_obj:
        # Reading form a file
        cities_str_lines = tsp_cities_file_obj.readlines()

    reduced_cities_str_lines = [cities_str_lines[i] for i in range(NUM_OF_CITIES)]
    number_of_cities = len(reduced_cities_str_lines)
    all_cities_arr = np.ndarray(shape=(number_of_cities, 2), dtype=np.int32)
    for i, city_line in enumerate(reduced_cities_str_lines):
        city_seperated = city_line.split()
        if len(city_seperated) == 2:  # To skip empty lines
            all_cities_arr[i, :] = [float(str_val) for str_val in city_seperated]

    all_distances = np.ndarray((number_of_cities, number_of_cities), dtype=np.float32)
    for i in range(all_distances.shape[0]):
        for j in range(all_distances.shape[1]):
            all_distances[i, j] = np.linalg.norm(all_cities_arr[i] - all_cities_arr[j])

    # temp_list = []
    # all_posibilities = []
    # generate_all_sorting_possibilities([i for i in range(number_of_cities)], temp_list, all_posibilities)

    # all_possible_chromosomes : list[TSPChromosome] = [TSPChromosome(all_cities_arr=all_cities_arr, all_distances=all_distances) for i in range(len(all_posibilities))]
    # min_fitness = np.inf
    # min_fitness_chromosome = None
    # fitness_obj=TSPFitness()

    # for (chromosome, possibility) in zip(all_possible_chromosomes, all_posibilities):
    #     chromosome.cities_indexed_arr = possibility
    #     chromosome.recalculate_distance_from_first_city(range(chromosome.get_chromosome_size()))
    #     chromosome_fitness = fitness_obj.eval_fitness(chromosome)
    #     if min_fitness > chromosome_fitness:
    #         min_fitness = chromosome_fitness
    #         min_fitness_chromosome = chromosome

    randomized_chromosomes = [
        TSPChromosome(all_cities_arr=all_cities_arr, all_distances=all_distances).randomize_value() for i in
        range(NUM_OF_CHROMOSOMES)]
    tsp_population = Population(randomized_chromosomes.copy(), fitness_obj=TSPFitness(),
                                crossover_func=tsp_crossover,
                                probabilities_computation_obj=GAFrameWork.MinimizationProblemComputeProbabilities(),
                                mutagen=TSPMutagen(4 / NUM_OF_CHROMOSOMES))
    best_fitness_history = []
    average_fitness_history = []
    median_fitness_history = []

    for i in range(0, NUM_OF_GENERATIONS):
        best_fitness = tsp_population.get_best_fitness()
        average_fitness = tsp_population.get_average_fitness()
        median_fitness = tsp_population.get_median_fitness()
        best_fitness_history.append(best_fitness)
        average_fitness_history.append(average_fitness)
        median_fitness_history.append(median_fitness)
        tsp_population.evolve()
        print("Finished Generetion #{0} \t best = {1} \t average = {2}".format(i + 1, best_fitness, average_fitness))

    # print(best_fitness_history)
    # print(average_fitness_history)

    # print("Best chromosome by brute force: " + str(min_fitness_chromosome.cities_indexed_arr) + " --> " + str(min_fitness))
    plt.plot(best_fitness_history)
    plt.plot(average_fitness_history)
    plt.plot(median_fitness_history)
    plt.xlabel("Generation")
    plt.ylabel('Fitness')
    plt.show()
