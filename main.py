from experiment import Experiment
from cell import Cell
import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
from scipy.stats import poisson
rng = default_rng(12345)

DNA_SIZE = int(1e4)
MR = 4e-6
N_CYCLES = 10
N_GENERATIONS = 8
BOTTLENECK_SIZE = 1
N_EXPERIMENTS = 10


def main():
    dna_sequence = rng.choice(Cell.N_BASES, DNA_SIZE)
    base_cell = Cell(dna_sequence, MR)

    experiments = [Experiment([Cell.from_cell(base_cell)]) for i in range(N_EXPERIMENTS)]
    for n, experiment in enumerate(experiments):
        print(f'experiment: {n}')
        experiment.do_full_experiment(N_CYCLES, N_GENERATIONS, BOTTLENECK_SIZE)
        print()
    final_cells = [experiment.cells[0] for experiment in experiments]
    num_mutations = np.zeros(N_EXPERIMENTS)
    for i in range(N_EXPERIMENTS):
        num_mutations[i] = (final_cells[i].dna_sequence != base_cell.dna_sequence).sum()
    plt.hist(num_mutations, bins=0.5 + np.arange(-1, 30))
    plt.show()

    pass


if __name__ == '__main__':
    main()
