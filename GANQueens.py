import numpy as np

import GAFrameWork
from GAFrameWork import Population
from matplotlib import pyplot as plt


class NQueenFitness(GAFrameWork.FitnessObject):
    def __init__(self, N):
        self.N = N
        self.mx_threat = np.uint32(((self.N - 1) * self.N) / 2)

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
        queens = []
        msk_size = np.uint8(np.ceil(np.log2(self.N)))
        msk = chromosome.apply_bitsize_cast_on_value((chromosome.apply_bitsize_cast_on_value(1) << msk_size) - 1)
        for i in range(self.N):
            queens.append((msk & chromosome.get_value()) >> chromosome.apply_bitsize_cast_on_value(msk_size * i))
            msk = msk << msk_size
        threats = 0
        for i in range(0, self.N):
            for j in range(i + 1, self.N):
                # same col threat
                if queens[i] == queens[j]:
                    threats += 1
                # right diag threat
                if queens[i] + (j - i) == queens[j]:
                    threats += 1
                # left diag threat
                if queens[i] - (j - i) == queens[j]:
                    threats += 1
        return self.mx_threat - threats


class NQueenChromosome(GAFrameWork.BitwiseChromosome):
    def __init__(self, N):
        self.N = N
        # 3 bits per a queen to locate it in a row
        # 1st queen at row0, 2nd queen at row 1, ... ,8th queen at row 7
        # 8 queens total
        super().__init__(np.uint32(np.ceil(np.log2(N))) * N)

    def create_chromosome(self, value):
        n_queen_chromosome = NQueenChromosome(self.N)
        n_queen_chromosome.set_value(value)
        return n_queen_chromosome


if __name__ == '__main__':
    N = 200
    N_queen = 9
    chromosomes = [NQueenChromosome(N_queen).randomize_value() for i in range(N)]
    nqueen_population = Population(chromosomes, fitness_obj=NQueenFitness(N_queen),
                                   crossover_func=GAFrameWork.uniform_binary_cross_over
                                   , mutagen=GAFrameWork.BitwiseMutagen(0.000001))
    best_fitness_history = []
    average_fitness_history = []
    for i in range(0, 370):
        best_fitness_history.append(nqueen_population.get_best_fitness())
        average_fitness_history.append(nqueen_population.get_average_fitness())
        nqueen_population = nqueen_population.evolve()
    plt.plot(best_fitness_history)
    plt.plot(average_fitness_history)
    plt.xlabel("Generation")
    plt.ylabel('Fitness')
    plt.show()
