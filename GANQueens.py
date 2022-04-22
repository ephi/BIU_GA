import numpy as np
import time
import GAFrameWork
from GAFrameWork import Population
from matplotlib import pyplot as plt


class NQueenFitness(GAFrameWork.FitnessObject):
    def __init__(self):
        pass

    def eval_fitness(self, chromosome):
        #    0 1 2 3 4 5 6 7
        # 0  X
        # 1          X
        # 2                X
        # 3  X
        # 4  X
        # 5              X
        # 6          X
        # 7                X
        N = chromosome.get_n_queen_N()
        queens = []
        msk_size = np.uint8(chromosome.get_bitmsk())
        msk = chromosome.apply_bitsize_cast_on_value((chromosome.apply_bitsize_cast_on_value(1) << msk_size) - 1)
        for i in range(N):
            queens.append((msk & chromosome.get_value()) >> chromosome.apply_bitsize_cast_on_value(msk_size * i))
            msk = msk << msk_size
        threats = 0
        for i in range(0, N):
            for j in range(i + 1, N):
                # same col threat
                if queens[i] == queens[j]:
                    threats += 1
                # right diag threat
                if queens[i] + (j - i) == queens[j]:
                    threats += 1
                # left diag threat
                if queens[i] - (j - i) == queens[j]:
                    threats += 1
        return threats


class NQueenChromosome(GAFrameWork.BitwiseChromosome):
    def __init__(self, N):
        if N < 1:
            raise ValueError("Minimal N is 1")
        if N > 15:
            raise ValueError("Maximal N is 15")
        self.N = N
        # 3 bits per a queen to locate it in a row
        # 1st queen at row0, 2nd queen at row 1, ... ,8th queen at row 7
        # 8 queens total
        self.bit_msk = np.uint32(np.ceil(np.log2(N)))
        super().__init__(self.bit_msk * N)

    def create_chromosome(self, value):
        n_queen_chromosome = NQueenChromosome(self.N)
        n_queen_chromosome.set_value(value)
        return n_queen_chromosome

    def get_bitmsk(self):
        return self.bit_msk

    def get_n_queen_N(self):
        return self.N

    def randomize_value(self):
        if (self.N - 1) & self.N == 0:
            self.set_value(np.random.randint(1 << self.bit_size, dtype=np.uint64))
        else:
            value = np.random.randint(self.N)
            for i in range(1, self.N):
                value <<= self.bit_msk
                value += np.random.randint(self.N)
            self.set_value(value)
        return self

    def print(self):
        if self.N != 8:
            return "Print is supported only when N=8"
        print_str = "\t"
        for i in range(self.N):
            print_str += f"{i} "
        print_str = print_str.rstrip() + "\n"
        queens = []
        msk_size = np.uint8(np.ceil(np.log2(self.N)))
        msk = self.apply_bitsize_cast_on_value((self.apply_bitsize_cast_on_value(1) << msk_size) - 1)
        for i in range(self.N):
            queens.append((msk & self.get_value()) >> self.apply_bitsize_cast_on_value(msk_size * i))
            msk = msk << msk_size
        for i, q in enumerate(queens):
            print_str += f"{i}\t"
            for k in range(q):
                print_str += "  "
            print_str += "X\n"
        return print_str


# treat the N(=8) Queen as CSP
# in: chromosome c that represents a random solution for N queen problem
# go over queens, for each queen at row i, if it threats any queen below it then move until position is found
# such that no queen below it is threaten.
# output: None if no solution is reachable from c or a new chromosome that is the solution reachable from c
def csp_n_queen_c(c, N=8):
    queens = []
    msk_size = np.uint8(np.ceil(np.log2(N)))
    msk = c.apply_bitsize_cast_on_value((c.apply_bitsize_cast_on_value(1) << msk_size) - 1)
    for i in range(N):
        queens.append((msk & c.get_value()) >> c.apply_bitsize_cast_on_value(msk_size * i))
        msk = msk << msk_size
    i = N - 2
    new_c_val = c.apply_bitsize_cast_on_value(queens[N - 1]) << c.apply_bitsize_cast_on_value((N - 1) * msk_size)
    while i >= 0:
        pos = queens[i]
        while True:
            safe_state = True
            for j in range(i + 1, N):
                # same col threat
                if pos == queens[j] or pos + (j - i) == queens[j] or pos - (j - i) == queens[j]:
                    pos = int((pos + 1) % N)
                    safe_state = False
                    break
            if safe_state:
                break
            if pos == queens[i]:
                return None
        queens[i] = pos
        pos = c.apply_bitsize_cast_on_value(pos) << c.apply_bitsize_cast_on_value((i * msk_size))
        new_c_val += pos
        i -= 1
    new_c = NQueenChromosome(N)
    new_c.set_value(new_c_val)
    return new_c


def csp_n_queen(N=8):
    start = time.perf_counter()
    while True:
        # randomize a chromosome of n queen problem
        c = NQueenChromosome(N).randomize_value()
        # try to solve CSP problem from this chromosome
        c = csp_n_queen_c(c, N)
        if c is not None:  # CSP was solvable from chromosome starting position
            return c
        end = time.perf_counter()
        if end - start > 2:
            print("csp_n_queen: Timeout")
            return None


def chromosome_stats(c, fitness_obj):
    c_fitness = fitness_obj.eval_fitness(c)
    print(f"Chromosome, threats={c_fitness}")
    print(c.print())


if __name__ == '__main__':
    N = 50
    N_queen = 8  # if N_queen is not power of 2 then invalid chromosomes might be created
    chromosomes = [NQueenChromosome(N_queen).randomize_value() for i in range(N)]
    fitness_obj = NQueenFitness()
    nqueen_population = Population(chromosomes, fitness_obj=fitness_obj,
                                   crossover_func=GAFrameWork.uniform_binary_cross_over,
                                   probabilities_computation_obj=GAFrameWork.MinimizationProblemComputeProbabilities()
                                   , mutagen=GAFrameWork.BitwiseMutagen(0.00001))
    best_fitness_history = []
    average_fitness_history = []
    # median_fitness_history = []
    start = time.perf_counter()
    for i in range(0, 50):
        best_fitness_history.append(nqueen_population.get_best_fitness())
        average_fitness_history.append(nqueen_population.get_average_fitness())
        # median_fitness_history.append(nqueen_population.get_median_fitness())
        nqueen_population.evolve()
    end = time.perf_counter()
    print(f"GA completed in {end - start} seconds")
    c = nqueen_population.get_best_chromosome()
    chromosome_stats(c, fitness_obj)
    plt.plot(best_fitness_history, label='Best')
    plt.plot(average_fitness_history, label="Average")
    # plt.plot(median_fitness_history, label="Median")
    plt.xlabel("Generation")
    plt.ylabel('Fitness')
    plt.legend()
    plt.show()
    # Brute force solution

    start = time.perf_counter()
    c = csp_n_queen(N_queen)
    end = time.perf_counter()
    print(f"CSP completed in {end - start} seconds")
    if c is not None:
        chromosome_stats(c, fitness_obj)
    else:
        print("No CSP solution was found in a reasonable time")


