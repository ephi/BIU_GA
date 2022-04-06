import GAFrameWork
from GAFrameWork import Population
from matplotlib import pyplot as plt


def eightqueen_fitness(chromosome):
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
    msk = 0x7
    for i in range(8):
        queens.append((msk & chromosome.get_value()) >> 3 * i)
        msk = msk << 3
    threats = 0
    for i in range(0, 8):
        for j in range(i + 1, 8):
            # same col threat
            if queens[i] == queens[j]:
                threats += 1
            # right diag threat
            if queens[i] + (j - i) == queens[j]:
                threats += 1
            # left diag threat
            if queens[i] - (j - i) == queens[j]:
                threats += 1
    return -threats


class EightQueenChromosome(GAFrameWork.BitwiseChromosome):
    def __init__(self):
        # 3 bits per a queen to locate it in a row
        # 1st queen at row0, 2nd queen at row 1, ... ,8th queen at row 7
        # 8 queens total
        super().__init__(3 * 8)

    # create addNumbers static method
    @staticmethod
    def create_chromosome(value):
        eight_queen_chromosome = EightQueenChromosome()
        eight_queen_chromosome.set_value(value)
        return eight_queen_chromosome


if __name__ == '__main__':
    N = 10
    chromosomes = [EightQueenChromosome().randomize_value() for i in range(N)]
    nqueen_population = Population(chromosomes, fitness_func=eightqueen_fitness,
                                   crossover_func=GAFrameWork.uniform_binary_cross_over
                                   , mutagen=GAFrameWork.BitwiseMutagen(0.0001))
    best_fitness_history = []
    average_fitness_history = []
    for i in range(0, 10):
        best_fitness_history.append(nqueen_population.get_best_fitness())
        average_fitness_history.append(nqueen_population.get_average_fitness())
        nqueen_population = nqueen_population.evolve()
    plt.plot(best_fitness_history)
    plt.plot(average_fitness_history)
    plt.ylabel('some numbers')
    plt.show()
