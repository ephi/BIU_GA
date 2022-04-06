from math import ceil
import graycode
import numpy as np

GRAYCODE_CHROMOSOME = True


def binary_single_point_cross_over(ca, cb):
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


def uniform_binary_cross_over(ca, cb):
    if not isinstance(ca, BitwiseChromosome) or not isinstance(cb, BitwiseChromosome):
        raise ValueError("chromosome a / chromosome b are not bitwise chromosomes")
    if ca.get_bitsize() != cb.get_bitsize():
        raise ValueError("chromosome a and chromosome b do not have the same bitsize")

    bitsize = ca.get_bitsize()
    child_a = 0
    child_b = 0
    msk = 1
    for i in range(0, bitsize):
        p = np.random.uniform(0, 1)
        # take from child a the i'th bit
        vl = ca.get_value()
        if p < 0.5:
            # take from child b the i'th bit
            vl = cb.get_value()
        v = (vl & msk) >> i
        child_a |= v << i
        child_b |= (1 - v) << i
        msk = msk << 1

    return ca.create_chromosome(child_a), cb.create_chromosome(child_b)


class Mutagen:
    def __init__(self, p=0.001):
        self.p = p

    def mutate(self, c):
        pass


class BitwiseMutagen(Mutagen):
    def __init__(self, p=0.001):
        super().__init__(p)

    def mutate(self, c):
        if not isinstance(c, BitwiseChromosome):
            raise ValueError("chromosome is not not bitwise")
        bitsize = c.get_bitsize()
        v = c.get_value()
        for i in range(0, bitsize):
            r_p = np.random.uniform(0, 1)
            if r_p < self.p:
                v = v ^ (1 << i)
        c.set_value(v)
        return c


class Population:
    def __init__(self, chromosomes, fitness_func, crossover_func, mutagen, elitism_percentage=0.01):
        self.chromosomes = chromosomes
        self.size = len(self.chromosomes)
        self.chromosomes_fitness = None
        self.fitness_sum = 0
        self.crossover_func = crossover_func
        self.mutagen = mutagen
        self.elitism_percentage = elitism_percentage
        self.fitness_func = fitness_func

    def eval_fitness(self):
        self.chromosomes_fitness = [self.fitness_func(chromosome) for chromosome in self.chromosomes]
        self.fitness_sum = sum(self.chromosomes_fitness)

    def get_average_fitness(self):
        if self.chromosomes_fitness is None:
            self.eval_fitness()
        return self.fitness_sum / self.size

    def get_best_fitness(self):
        if self.chromosomes_fitness is None:
            self.eval_fitness()
        return max(self.chromosomes_fitness)

    def evolve(self):

        # evaluate fitness
        if self.chromosomes_fitness is None:
            self.eval_fitness()

        new_poplation_size = 0
        new_generation = []
        # apply elitism
        elitism_size = ceil(self.size * self.elitism_percentage)
        if elitism_size % 2 != 0:
            elitism_size += 1
        strongest_indexes = np.argpartition(self.chromosomes_fitness, -elitism_size)[-elitism_size:]
        for index in strongest_indexes:
            new_generation.append(self.chromosomes[index])
        weakest_indexes = np.argpartition(self.chromosomes_fitness, elitism_size)[:elitism_size]
        weakest_chromosomes_fitness = []
        self.size -= len(weakest_indexes)
        for index in weakest_indexes:
            weakest_chromosomes_fitness.append((self.chromosomes[index], self.chromosomes_fitness[index]))
        for cf in weakest_chromosomes_fitness:
            self.chromosomes.remove(cf[0])
            self.chromosomes_fitness.remove(cf[1])
        self.fitness_sum = sum(self.chromosomes_fitness)
        # compute probability for each
        c_probs = [f / self.fitness_sum for f in self.chromosomes_fitness]
        if GRAYCODE_CHROMOSOME:
            try:
                for c in self.chromosomes:
                    c.to_gray_code()
            except Exception:
                print("GAFrameWork: unable to convert chromosome into graycode")

        # while new population size < current population size
        while new_poplation_size < self.size:
            # select 2 for cross-over using fitness proportional selection
            chromosome_a, chromosome_b = np.random.choice(self.chromosomes, p=c_probs, size=2)
            # apply cross over between the two chromosomes
            child_a, child_b = self.crossover_func(chromosome_a, chromosome_b)
            # apply the mutation on both children
            self.mutagen.mutate(child_a)
            self.mutagen.mutate(child_b)

            # add the children to the new generation
            new_generation.append(child_a)
            new_generation.append(child_b)

            new_poplation_size += 2
        if GRAYCODE_CHROMOSOME:
            try:
                for c in new_generation:
                    c.to_binary_code()
            except Exception:
                print("GAFrameWork: unable to convert chromosome into binray code")

        return Population(new_generation, self.fitness_func, self.crossover_func, self.mutagen, self.elitism_percentage)


class Chromosome:
    def __init__(self):
        self.value = None

    def get_value(self):
        return self.value

    def set_value(self, v):
        self.value = v

    def randomize_value(self):
        pass

    def to_gray_code(self):
        pass

    def to_binary_code(self):
        pass

    # create addNumbers static method
    @staticmethod
    def create_chromosome(value):
        pass


class BitwiseChromosome(Chromosome):
    def __init__(self, bit_size):
        super().__init__()
        self.bit_size = bit_size

    def set_value(self, v):
        if self.bit_size <= 8:
            v = np.uint8(v)
        elif self.bit_size <= 16:
            v = np.uint16(v)
        elif self.bit_size <= 32:
            v = np.uint32(v)
        elif self.bit_size <= 64:
            v = np.uint64(v)
        self.value = v

    def get_bitsize(self):
        return self.bit_size

    def randomize_value(self):
        self.set_value(np.random.randint(1 << self.bit_size))
        return self

    def to_gray_code(self):
        self.value = graycode.tc_to_gray_code(self.value)

    def to_binary_code(self):
        self.value = graycode.gray_code_to_tc(self.value)


if __name__ == '__main__':
    pass
