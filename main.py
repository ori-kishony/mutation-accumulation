from experiment import Experiment
from cell import Cell, FitnessCell
import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
from scipy.stats import poisson
rng = default_rng(12345)

DNA_SIZE = int(1e3)
MR = 4e-3
N_CYCLES = 10
N_GENERATIONS = 6
BOTTLENECK_SIZE = 20
N_EXPERIMENTS = 10


def main():
    dna_sequence = rng.choice(Cell.N_BASES, DNA_SIZE).astype(np.int8)
    base_cell = FitnessCell(dna_sequence, MR)

    experiments = [Experiment([FitnessCell.from_cell(base_cell)]) for i in range(N_EXPERIMENTS)]
    for n, experiment in enumerate(experiments):
        print(f'experiment: {n}')
        experiment.do_full_experiment(N_CYCLES, N_GENERATIONS, BOTTLENECK_SIZE)
        print()
    final_cells = [experiment.cells[0] for experiment in experiments]
    num_mutations = np.zeros(N_EXPERIMENTS)
    for i in range(N_EXPERIMENTS):
        num_mutations[i] = (final_cells[i].dna_sequence != base_cell.dna_sequence).sum()
    plt.hist(num_mutations, bins=0.5 + np.arange(-1, 50, 2))
    plt.show()

    pass


if __name__ == '__main__':
    main()
