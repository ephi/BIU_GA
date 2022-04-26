from math import ceil
from time import time
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

def generate_all_sorting_possibilities(remaining_options : list, current_possibility_arr : list, all_possibilities_vec : list[list]):
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

def calculate_all_distances(all_cities_arr, all_distances):
    # number_of_cities = len(all_cities_arr)
    # all_distances = np.ndarray((number_of_cities, number_of_cities), dtype=np.float32)
    for i in range(all_distances.shape[0]):
        for j in range(all_distances.shape[1]):
            if i == j:
                all_distances[i, j] = np.inf
            else:
                all_distances[i, j] = np.linalg.norm(all_cities_arr[i] - all_cities_arr[j])
    # return all_distances

class TSPChromosome(GAFrameWork.Chromosome):
    def __init__(self, all_cities_arr : np.ndarray, all_distances : np.ndarray):
        self.chromosome_size = all_cities_arr.shape[0]
        self.cities_indexed_arr = np.zeros(self.chromosome_size, dtype=np.uint8)
        self.distance_from_first_city = np.zeros(self.chromosome_size, dtype=np.float32)
        self.all_cities_arr = all_cities_arr # An array of x,y tuples. 
        self.all_distances = all_distances
        self.full_path_length = 0
        self.temp_float_vector = np.zeros(self.chromosome_size, dtype=np.float32)

    def get_chromosome_size(self):
        return self.chromosome_size

    def get_value(self):
        return self.cities_arr

    def set_value(self, v):
        self.value = v

    def check_for_duplicate_cities(self): # Only for debugging
        for val in self.cities_indexed_arr:
            assert(np.count_nonzero(self.cities_indexed_arr == val) ==1)

    def get_distance_between_cities(self, index_1, index_2) -> np.float32:
        index_1 = index_1 % self.chromosome_size
        index_2 = index_2 % self.chromosome_size
        return self.all_distances[self.cities_indexed_arr[index_1], self.cities_indexed_arr[index_2]]

    def recalculate_distance_from_first_city(self):
        self.full_path_length = 0
        for index in range(self.chromosome_size):
            self.full_path_length += self.get_distance_between_cities(index, index-1)

    def randomize_value(self, seq_num : int):
        self.cities_indexed_arr[0] = seq_num % len(self.all_cities_arr)
        # if seq_num / len(self.all_cities_arr) < 2:

        #     for i in range(self.chromosome_size - 1):
        #         min_dis_index = np.argmin(self.all_distances[self.cities_indexed_arr[i], :])
        #         self.cities_indexed_arr[i+1] = min_dis_index
        #         self.all_distances[self.cities_indexed_arr[i], :] = np.inf
        #         self.all_distances[:, self.cities_indexed_arr[i]] = np.inf
                
        #     calculate_all_distances(self.all_cities_arr, self.all_distances)
        #     if seq_num >= self.chromosome_size:
        #         self.cities_indexed_arr = self.cities_indexed_arr[::-1]
        # else:
        all_cities_indexes_reduced = np.array(range(self.chromosome_size))
        all_cities_indexes_reduced = np.delete(all_cities_indexes_reduced, self.cities_indexed_arr[0])
        rand_city_places = np.random.randint(0, range(self.chromosome_size - 1, 0, -1))
        for i, rand_city_place in enumerate(rand_city_places):
            self.cities_indexed_arr[i+1] = all_cities_indexes_reduced[rand_city_place - 1]
            all_cities_indexes_reduced = np.delete(all_cities_indexes_reduced, rand_city_place - 1)

        self.recalculate_distance_from_first_city()

        return self

    def distance_from_neighbor_cities(self, city_index):
        return self.get_distance_between_cities(city_index, city_index - 1) + self.get_distance_between_cities(city_index, city_index + 1)

    def fix_ranges_to_have_same_indexes(self, master_range : np.ndarray, slave_range : np.ndarray):
        indexes_intersection = [index for index in master_range if index in slave_range]
        if(len(indexes_intersection) == 0):
            return False
        slave_indexes_to_change = [(not index in indexes_intersection) for index in slave_range]
        master_indexes_missing_in_slave_range = [index for index in master_range if not index in indexes_intersection]
        master_indexes_to_change = [(not index in indexes_intersection) for index in master_range]
        slave_indexes_missing_in_masetr_range = [index for index in slave_range if not index in indexes_intersection]
        slave_range[slave_indexes_to_change] = master_indexes_missing_in_slave_range
        master_range[master_indexes_to_change] = slave_indexes_missing_in_masetr_range
        return True

    def swap_ranges(self, self_range : np.ndarray, other_chromosome, other_range : np.ndarray, orig_self, orig_other) -> np.bool8:
        assert(self_range[1] - self_range[0] == other_range[1] - other_range[0])
        range_length = self_range[1] - self_range[0] + 1
        temp_sliced_self_indexes_vector = np.zeros(range_length, dtype=np.uint8)
        temp_sliced_other_indexes_vector = np.zeros(range_length, dtype=np.uint8)

        for index_in_temp_vec, index_in_cities_vec in enumerate(range(self_range[0], self_range[1] + 1)):
            temp_sliced_self_indexes_vector[index_in_temp_vec] = orig_self.cities_indexed_arr[index_in_cities_vec % self.chromosome_size]

        for index_in_temp_vec, index_in_cities_vec in enumerate(range(other_range[0], other_range[1] + 1)):
            temp_sliced_other_indexes_vector[index_in_temp_vec] = orig_other.cities_indexed_arr[index_in_cities_vec % self.chromosome_size]

        need_to_swap = self.fix_ranges_to_have_same_indexes(temp_sliced_self_indexes_vector, temp_sliced_other_indexes_vector)
        if need_to_swap:
            for index_in_temp_vec, index_in_cities_vec in enumerate(range(other_range[0], other_range[1] + 1)):
                other_chromosome.cities_indexed_arr[index_in_cities_vec % self.chromosome_size] = temp_sliced_self_indexes_vector[index_in_temp_vec]
            for index_in_temp_vec, index_in_cities_vec in enumerate(range(self_range[0], self_range[1] + 1)):
                self.cities_indexed_arr[index_in_cities_vec % self.chromosome_size] = temp_sliced_other_indexes_vector[index_in_temp_vec]

        return need_to_swap

    def swap_two_cities(self, indexes_to_swap):
        if indexes_to_swap[0] == indexes_to_swap[1]:
            return
        if indexes_to_swap[0] > indexes_to_swap[1]:
            tmp = indexes_to_swap[1]
            indexes_to_swap[1] = indexes_to_swap[0]
            indexes_to_swap[0] = tmp
        
        orig_distance_from_neighbor_cities = np.array([self.distance_from_neighbor_cities(city_index) for city_index in indexes_to_swap])

        tmp = self.cities_indexed_arr[indexes_to_swap[1]]
        self.cities_indexed_arr[indexes_to_swap[1]] = self.cities_indexed_arr[indexes_to_swap[0]]
        self.cities_indexed_arr[indexes_to_swap[0]] = tmp

        new_distance_from_neighbor_cities = np.array([self.distance_from_neighbor_cities(city_index) for city_index in indexes_to_swap])
        difference_in_distances = new_distance_from_neighbor_cities - orig_distance_from_neighbor_cities
        self.full_path_length += (difference_in_distances[0] + difference_in_distances[1])

    def __str__(self) -> str:
        return str(self.cities_indexed_arr)

class TSPMutagen(GAFrameWork.Mutagen):
    def __init__(self, p=0.001):
        self.p = p
        self.chromosome_size = 0
        self.chromosome_indexes_range = None

    def mutate(self, chromosome : TSPChromosome, generation : int):
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
    def eval_fitness(self, chromosome : TSPChromosome) -> np.float32:
        return chromosome.full_path_length

    def get_worst_fitness_value(self):
        return np.inf

    def is_minimization_fitness(self):
        return True

class TSPComputeProbabilities(GAFrameWork.ProbabilitiesComputation):
    def compute_probabilities(self, fitnesses_arr : np.ndarray) -> np.ndarray:
        converted_fitnesses_to_maximizing_problem = [1 / f for f in fitnesses_arr]
        return self.normalize_probabilities(converted_fitnesses_to_maximizing_problem)

def swap_ranges_between_two_chromosomes(chromosome_1 : TSPChromosome, chromosome_2 : TSPChromosome, 
                                        start_index_1 : np.uint8, start_index_2 : np.uint8, 
                                        range_length : np.uint8, orig_self, orig_other) :
    range_to_swap_in_parent_1 = np.array([start_index_1, start_index_1 + range_length - 1], dtype=np.uint8)
    range_to_swap_in_parent_2 = np.array([start_index_2, start_index_2 + range_length - 1], dtype=np.uint8)
    return chromosome_1.swap_ranges(range_to_swap_in_parent_1, chromosome_2, range_to_swap_in_parent_2, orig_self, orig_other)

def tsp_crossover(chromosome_1 : TSPChromosome, chromosome_2 : TSPChromosome, generation : int):
    new_child_1 : TSPChromosome = copy.deepcopy(chromosome_1)
    new_child_2 : TSPChromosome = copy.deepcopy(chromosome_2) 
    need_to_swap = False

    chromosome_size = chromosome_1.get_chromosome_size()

    # for index, (chromosome_1_gene, chromosome_2_gene) in enumerate(zip(chromosome_1.cities_indexed_arr, chromosome_2.cities_indexed_arr)):
    #     if index > 0 and chromosome_1_gene != chromosome_2_gene:
    #         need_to_swap = True
    #         second_index_to_swap_with_in_chromosome_1 = np.where(chromosome_1.cities_indexed_arr == chromosome_2_gene)[0]
    #         new_child_1.swap_two_cities([index, second_index_to_swap_with_in_chromosome_1])

    #         second_index_to_swap_with_in_chromosome_2 = np.where(chromosome_2.cities_indexed_arr == chromosome_1_gene)[0]
    #         new_child_2.swap_two_cities([index, second_index_to_swap_with_in_chromosome_2])
    #         break

    # if need_to_swap:
    #     new_child_1.recalculate_distance_from_first_city()
    #     new_child_2.recalculate_distance_from_first_city()
    # # # else:
    # #     print()
    # return new_child_1, new_child_2

    range_length_to_swap = np.random.randint(2, chromosome_size/2, dtype=np.uint8)
    # range_length_to_swap = chromosome_size/2

    range_start_1 = np.random.randint(chromosome_size, dtype=np.uint8)
    range_start_2 = np.random.randint(chromosome_size, dtype=np.uint8)
    range_end_1 = (range_start_1 + range_length_to_swap) % chromosome_size
    range_end_2 = (range_start_2 + range_length_to_swap) % chromosome_size
    need_to_be_splited = max(range_start_1, range_start_2) + range_length_to_swap >= chromosome_size
    # print("range_length_to_swap = {0}".format(range_length_to_swap))
    # print("range_start_1 = {0}, range_end_1 = {1}".format(range_start_1, range_end_1))
    # print("range_start_2 = {0}, range_end_2 = {1}".format(range_start_2, range_end_2))
    # print("need_to_be_splited = {0}".format(need_to_be_splited))
    # while need_to_be_splited:
    #     later_start = max(range_start_1, range_start_2)
    #     current_length_from_start = chromosome_size - later_start
    #     need_to_swap = need_to_swap or swap_ranges_between_two_chromosomes(new_child_1, new_child_2, range_start_1, range_start_2, current_length_from_start, chromosome_1, chromosome_2)
    #     range_start_1 = (range_start_1 + current_length_from_start) % chromosome_size
    #     range_start_2 = (range_start_2 + current_length_from_start) % chromosome_size
    #     range_length_to_swap -= current_length_from_start
    #     # Now one of them should be zero and one should be close to end
        
    #     need_to_be_splited = max(range_start_1, range_start_2) + range_length_to_swap >= chromosome_size
    #     # print("range_length_to_swap = {0}".format(range_length_to_swap))
    #     # print("range_start_1 = {0}, range_end_1 = {1}".format(range_start_1, range_end_1))
    #     # print("range_start_2 = {0}, range_end_2 = {1}".format(range_start_2, range_end_2))
    #     # print("need_to_be_splited = {0}".format(need_to_be_splited))

    assert(range_length_to_swap >= 0)
    # assert(range_end_2 >= range_start_2)
    # assert(range_end_1 >= range_start_1)
    # assert(range_length_to_swap == range_end_1 - range_start_1)
    # assert(range_length_to_swap == range_end_2 - range_start_2)

    if range_start_1 != range_end_1:
        need_to_swap = need_to_swap or swap_ranges_between_two_chromosomes(new_child_1, new_child_2, range_start_1, range_start_2, range_length_to_swap, chromosome_1, chromosome_2)

    if need_to_swap:
        new_child_1.recalculate_distance_from_first_city()
        new_child_2.recalculate_distance_from_first_city()
    return new_child_1, new_child_2

if __name__ == '__main__':
     
    cities_str_lines = [""]
    with open(TSP_CITIES_VALUES_FILE_NAME_CONST, "r+") as tsp_cities_file_obj:
        # Reading form a file
        cities_str_lines = tsp_cities_file_obj.readlines()
    
    reduced_cities_str_lines = [ cities_str_lines[i] for i in range(NUM_OF_CITIES)]
    number_of_cities = len(reduced_cities_str_lines)
    all_cities_arr = np.ndarray(shape= (number_of_cities, 2), dtype=np.int32)
    for i, city_line in enumerate(reduced_cities_str_lines):
        city_seperated = city_line.split()
        if len(city_seperated) == 2: # To skip empty lines
            all_cities_arr[i, :] = [float(str_val) for str_val in city_seperated]

    all_distances = np.ndarray((number_of_cities, number_of_cities), dtype=np.float32)
    calculate_all_distances(all_cities_arr, all_distances)

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

    randomized_chromosomes = [TSPChromosome(all_cities_arr=all_cities_arr, all_distances=all_distances).randomize_value(i + 100) for i in range(NUM_OF_CHROMOSOMES)]
    tsp_population = Population(randomized_chromosomes.copy(), fitness_obj=TSPFitness(),
                                   crossover_func=tsp_crossover,
                                   probabilities_computation_obj=TSPComputeProbabilities(),
                                   chromosome_type=TSPChromosome,
                                   mutagen=TSPMutagen(40 / NUM_OF_CHROMOSOMES),
                                   elitism_percentage=(2 / NUM_OF_CHROMOSOMES))
    best_fitness_history = []
    best_chromosoe_history : list[TSPChromosome] = [] 
    average_fitness_history = []
    median_fitness_history = []
    sum_time = 0
    delta_time = 0
    
    for i in range(0, NUM_OF_GENERATIONS):
        start_time = time()
        best_fitness = tsp_population.get_best_fitness()
        average_fitness = tsp_population.get_average_fitness()
        median_fitness = tsp_population.get_median_fitness()
        best_fitness_history.append(best_fitness)
        average_fitness_history.append(average_fitness)
        median_fitness_history.append(median_fitness)
        best_chromosoe_history.append(tsp_population.get_best_chromosome())
        tsp_population.evolve()
        end_time = time()
        delta_time = end_time - start_time
        sum_time += delta_time
        print("Generetion #{0}\tbest = {1}\taverage = {2}\ttime = {3}\t avg = {4}".format(i+1, best_fitness, average_fitness, delta_time, sum_time/(i+1)))

   # print(best_fitness_history)
   # print(average_fitness_history)

    # print("Best chromosome by brute force: " + str(min_fitness_chromosome.cities_indexed_arr) + " --> " + str(min_fitness))
    plt.plot(best_fitness_history)
    plt.plot(average_fitness_history)
    plt.plot(median_fitness_history)
    output_lines = [str(best_chromosoe_history[i]) + " : " + str(best_fitness_history[i]) + "\tavg = " + str(average_fitness_history[i]) + "\n" for i in range(len(best_fitness_history))]
    with open("output.txt", "w+") as o_file:
        # Reading form a file
        o_file.writelines(output_lines)
    plt.xlabel("Generation")
    plt.ylabel('Fitness')
    plt.show()

