from math import floor

try:
    import PythonGALib

    to_bin_code = PythonGALib.inverse_gray
    to_gray_code = PythonGALib.to_gray
except Exception as e:
    import graycode

    to_bin_code = graycode.gray_code_to_tc
    to_gray_code = graycode.tc_to_gray_code
import numpy as np

GRAYCODE_CHROMOSOME = True


def binary_single_point_cross_over(ca, cb, generation):
    if type(ca) is not BitwiseChromosome or type(cb) is not BitwiseChromosome:
        raise ValueError("chromosome a / chromosome b are not bitwise chromosomes")
    if ca.get_bitsize() != cb.get_bitsize():
        raise ValueError("chromosome a and chromosome b do not have the same bitsize")
    bitsize = ca.get_bitsize()
    bitsize_cut = (1 << np.random.randint(bitsize + 1)) - 1
    bitmsk = (1 << bitsize) & ~bitsize_cut

    child_a = ca.get_value() & bitsize_cut | cb.get_value() & bitmsk
    child_b = cb.get_value() & bitsize_cut | ca.get_value() & bitmsk
    return child_a, child_b


def uniform_binary_cross_over(ca, cb, generation):
    if not isinstance(ca, BitwiseChromosome) or not isinstance(cb, BitwiseChromosome):
        raise ValueError("chromosome a / chromosome b are not bitwise chromosomes")
    if ca.get_bitsize() != cb.get_bitsize():
        raise ValueError("chromosome a and chromosome b do not have the same bitsize")

    bitsize = ca.get_bitsize()
    child_a = ca.apply_bitsize_cast_on_value(0)
    child_b = cb.apply_bitsize_cast_on_value(0)
    msk = cb.apply_bitsize_cast_on_value(1)
    for i in range(0, bitsize):
        k = ca.apply_bitsize_cast_on_value(i)
        p = np.random.uniform(0, 1)
        # take from child a the i'th bit
        ca_bit = (ca.get_value() & msk) >> k
        # take from child b the i'th bit
        cb_bit = (cb.get_value() & msk) >> k
        if p < 0.5:
            # bits are inverted
            tmp = ca_bit
            ca_bit = cb_bit
            cb_bit = tmp
        child_a |= ca_bit << k
        child_b |= cb_bit << k
        msk = msk << ca.apply_bitsize_cast_on_value(1)

    return ca.create_chromosome(child_a), cb.create_chromosome(child_b)


class ProbabilitiesComputation:
    def normalize_probabilities(self, fitnesses_arr: np.ndarray) -> np.ndarray:
        fitnesses_sum = np.sum(fitnesses_arr)
        return fitnesses_arr / fitnesses_sum

    def compute_probabilities(self, fitnesses_arr: np.ndarray) -> np.ndarray:
        return self.normalize_probabilities(fitnesses_arr)


class MinimizationProblemComputeProbabilities(ProbabilitiesComputation):
    def compute_probabilities(self, fitnesses_arr: np.ndarray) -> np.ndarray:
        epsilon = np.exp(-10)
        converted_fitnesses_to_maximizing_problem = [1 / (f + epsilon) for f in fitnesses_arr]
        return self.normalize_probabilities(converted_fitnesses_to_maximizing_problem)


class Mutagen:
    def __init__(self, p=0.001):
        self.p = p

    def mutate(self, c, generation):
        pass


class BitwiseMutagen(Mutagen):
    def __init__(self, p=0.001):
        super().__init__(p)

    def mutate(self, c, generation):
        if not isinstance(c, BitwiseChromosome):
            raise ValueError("chromosome is not not bitwise")
        bitsize = c.get_bitsize()
        v = c.get_value()
        for i in range(0, bitsize):
            r_p = np.random.uniform(0, 1)
            if r_p < self.p:
                k = c.apply_bitsize_cast_on_value(i)
                v = v ^ (c.apply_bitsize_cast_on_value(1) << k)
        c.set_value(v)
        return c


class FitnessObject:
    def __init__(self):
        pass

    def eval_fitness(self, chromosome):
        pass

    def is_minimization_fitness(self):
        return True


class Population:
    def __init__(self, chromosomes,
                 fitness_obj: FitnessObject,
                 crossover_func,
                 mutagen: Mutagen,
                 probabilities_computation_obj: ProbabilitiesComputation,
                 elitism_percentage=0.01, cross_over_probability=0.7):

        self.chromosomes = chromosomes
        self.size = len(self.chromosomes)
        if self.size < 1:
            raise ValueError("Population is of size 0, not allowed.")
        if self.size % 2 == 1:
            raise ValueError("Population is odd, not allowed.")
        self.chromosomes_fitness = None
        self.fitness_sum = 0
        self.crossover_func = crossover_func
        self.mutagen = mutagen
        self.elitism_percentage = elitism_percentage
        self.fitness_obj = fitness_obj
        self.probabilities_computation_obj = probabilities_computation_obj
        self.generation_num = 0
        self.cross_over_probability = cross_over_probability

    def eval_fitness(self):
        self.chromosomes_fitness = [self.fitness_obj.eval_fitness(chromosome) for chromosome in self.chromosomes]
        self.fitness_sum = sum(self.chromosomes_fitness)

    def get_median_fitness(self):
        if self.chromosomes_fitness is None:
            self.eval_fitness()
        return np.median(self.chromosomes_fitness)

    def get_average_fitness(self):
        if self.chromosomes_fitness is None:
            self.eval_fitness()
        return self.fitness_sum / self.size

    def get_best_fitness(self):
        if self.chromosomes_fitness is None:
            self.eval_fitness()
        return min(self.chromosomes_fitness)

    def get_random_chromosome(self):
        c = np.random.choice(self.chromosomes)
        return c

    def get_best_chromosome(self):
        if self.chromosomes_fitness is None:
            self.eval_fitness()
        if self.fitness_obj.is_minimization_fitness():
            optimum_index = np.argmin(self.chromosomes_fitness)
        else:
            optimum_index = np.argmax(self.chromosomes_fitness)
        return self.chromosomes[optimum_index]

    def evolve(self):
        global GRAYCODE_CHROMOSOME

        # evaluate fitness
        if self.chromosomes_fitness is None:
            self.eval_fitness()

        new_generation = []
        # apply elitism
        elitism_size = max(floor(self.size * self.elitism_percentage), 1)
        if elitism_size % 2 != 0:
            elitism_size += 1
        if self.fitness_obj.is_minimization_fitness():
            # weakest has the highest fitness/strongest has the lowest fitness
            strongest_indexes = np.argpartition(self.chromosomes_fitness, elitism_size)[:elitism_size]
            weakest_indexes = np.argpartition(self.chromosomes_fitness, -elitism_size)[-elitism_size:]
        else:
            # weakest has the lowest fitness/strongest has the highest fitness
            strongest_indexes = np.argpartition(self.chromosomes_fitness, -elitism_size)[-elitism_size:]
            weakest_indexes = np.argpartition(self.chromosomes_fitness, elitism_size)[:elitism_size]
        # apply graycode conversion before eltisim if needed so all the next
        # generation will have be at graycode form, so they will get converted back to binray
        # properly
        if GRAYCODE_CHROMOSOME:
            if isinstance(self.chromosomes[0], BitwiseChromosome):
                try:
                    for c in self.chromosomes:
                        c.to_gray_code()
                except Exception as e:
                    print("GAFrameWork: unable to convert chromosome into graycode. DISABLING GRAYCODE CONVERSION")
                    GRAYCODE_CHROMOSOME = False

        for index in strongest_indexes:
            new_generation.append(self.chromosomes[index])
        weakest_chromosomes_fitness = []
        new_poplation_size = len(weakest_indexes)
        for index in weakest_indexes:
            weakest_chromosomes_fitness.append((self.chromosomes[index], self.chromosomes_fitness[index]))
        # prev_generation_str = ""
        # for i, (chromosome, chromosome_fitness) in enumerate(zip(self.chromosomes, self.chromosomes_fitness)):
        #     prev_generation_str += "#" + str(i) + " :\t" + str(chromosome) + " --> " + str(chromosome_fitness) + "\n"
        # print(prev_generation_str)
        for cf in weakest_chromosomes_fitness:
            self.chromosomes.remove(cf[0])
            self.chromosomes_fitness.remove(cf[1])

        # compute probability for each
        c_probs = self.probabilities_computation_obj.compute_probabilities(self.chromosomes_fitness)

        # while new population size < current population size
        while new_poplation_size < self.size:
            # select 2 for cross-over using fitness proportional selection
            chromosome_a, chromosome_b = np.random.choice(self.chromosomes, p=c_probs, size=2, replace=False)
            if np.random.uniform() < self.cross_over_probability:
                # apply cross over between the two chromosomes
                child_a, child_b = self.crossover_func(chromosome_a, chromosome_b, self.generation_num)
            else:
                child_a = chromosome_a
                child_b = chromosome_b
            # apply the mutation on both children
            self.mutagen.mutate(child_a, self.generation_num)
            self.mutagen.mutate(child_b, self.generation_num)

            # add the children to the new generation
            new_generation.append(child_a)
            new_generation.append(child_b)

            new_poplation_size += 2
        if GRAYCODE_CHROMOSOME:
            if isinstance(self.chromosomes[0], BitwiseChromosome):
                try:
                    for c in new_generation:
                        c.to_binary_code()
                except Exception as e:
                    print("GAFrameWork: unable to convert chromosome into binray code, DISABLING GRAYCODE CONVERSION")
                    GRAYCODE_CHROMOSOME = False
        self.chromosomes = new_generation
        self.chromosomes_fitness = None
        self.generation_num += 1
        # return Population(chromosomes=new_generation, 
        #     fitness_obj=self.fitness_obj, 
        #     crossover_func=self.crossover_func,
        #     mutagen=self.mutagen, 
        #     probabilities_computation_obj=self.probabilities_computation_obj,
        #     elitism_percentage=self.elitism_percentage)


class Chromosome:
    def __init__(self):
        self.value = None

    def get_value(self):
        return self.value

    def set_value(self, v):
        self.value = v

    def randomize_value(self):
        pass

    def create_chromosome(self, value):
        pass


class BitwiseChromosome(Chromosome):
    def __init__(self, bit_size):
        super().__init__()
        self.bit_size = bit_size
        if self.bit_size <= 8:
            self.bit_cast_func = np.uint8
        elif self.bit_size <= 16:
            self.bit_cast_func = np.uint16
        elif self.bit_size <= 32:
            self.bit_cast_func = np.uint32
        elif self.bit_size <= 64:
            self.bit_cast_func = np.uint64

    def set_value(self, v):
        self.value = self.bit_cast_func(v)

    def get_bitsize(self):
        return self.bit_size

    def randomize_value(self):
        self.set_value(np.random.randint(1 << self.bit_size, dtype=np.uint64))
        return self

    def to_gray_code(self):
        graycoded = to_gray_code(self.value.item())
        self.set_value(graycoded)

    def to_binary_code(self):
        self.set_value(to_bin_code(self.value.item()))

    def apply_bitsize_cast_on_value(self, value):
        return self.bit_cast_func(value)


if __name__ == '__main__':
    pass
