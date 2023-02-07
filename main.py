from experiment import Experiment
from cell import Cell, FitnessCell
import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
from itertools import product
from scipy.stats import poisson
rng = default_rng(12345)

DNA_SIZE = int(1e3)
MR = 4e-3
N_CYCLES = 10
N_GENERATIONS = 6
BOTTLENECK_SIZE = 20
N_EXPERIMENTS = 10


def main():
    bad_dna_sequence = rng.choice(Cell.N_BASES, DNA_SIZE).astype(np.int8)
    good_mutations_dict = {}
    bad_mutations_dict = {}
    total_mutations_dict = {}
    for i, dna_sequence in enumerate([bad_dna_sequence, FitnessCell(bad_dna_sequence, 0).ref_dna_sequence]):
        dna_sequence = dna_sequence.copy()
        base_cell = FitnessCell(dna_sequence, MR)
        n_generations_list = [3, 5]
        bottle_neck_size_list = [1, 16]
        for n_gen, bottle_neck in product(n_generations_list, bottle_neck_size_list):
            experiments = [Experiment([FitnessCell.from_cell(base_cell) for i in range(bottle_neck)])
                           for i in range(N_EXPERIMENTS)]
            for n, experiment in enumerate(experiments):
                print(f'experiment: {n}')
                experiment.do_full_experiment(N_CYCLES, n_gen, bottle_neck)
                print()
            final_cells = [cell for experiment in experiments for cell in experiment.cells]
            good_mutations = np.mean([((cell.dna_sequence == cell.ref_dna_sequence)
                               & (cell.dna_sequence != base_cell.dna_sequence)).sum() for cell in final_cells])
            bad_mutations = np.mean([((cell.dna_sequence != cell.ref_dna_sequence)
                              & (cell.dna_sequence != base_cell.dna_sequence)).sum() for cell in final_cells])
            total_mutations = good_mutations + bad_mutations
            good_mutations_dict[(i, n_gen, bottle_neck)] = good_mutations
            bad_mutations_dict[(i, n_gen, bottle_neck)] = bad_mutations
            total_mutations_dict[(i, n_gen, bottle_neck)] = total_mutations
            print(f'Number generations, bottle neck:{n_gen}, {bottle_neck} gives {good_mutations} good mutations and {bad_mutations} bad mutations')

    # plots
    good_total_ratio_dict = {k: good_mutations_dict[k] / total_mutations_dict[k] for k in good_mutations_dict}
    good_total_ratio_array = np.zeros([2, len(n_generations_list), len(bottle_neck_size_list)])
    approx_mr_array = good_total_ratio_array.copy()
    for i in range(2):
        for j, n_gen in enumerate(n_generations_list):
            for k, b_size in enumerate(bottle_neck_size_list):
                good_total_ratio_array[i, j, k] = good_total_ratio_dict[(i, n_gen, b_size)]
                approx_mr = total_mutations_dict[(i, n_gen, b_size)] / (N_CYCLES * n_gen)
                approx_mr_array[i, j, k] = approx_mr
    for i in range(2):
        plt.figure()
        plt.contourf(n_generations_list, np.log2(bottle_neck_size_list), approx_mr_array[i])
        plt.title(f'Mutation Rate Approximation: true rate is: {MR}')
        ax = plt.gca()
        ax.set_xlabel("Number of Generations")
        ax.set_ylabel("Log Bottle Neck Size")
        plt.colorbar()

        plt.figure()
        plt.contourf(n_generations_list, np.log2(bottle_neck_size_list), good_total_ratio_array[i])
        plt.title("Benefitial Mutations Relative to Total Mutations")
        ax = plt.gca()
        ax.set_xlabel("Number of Generations")
        ax.set_ylabel("Log Bottle Neck Size")
        plt.colorbar()
    plt.show()



if __name__ == '__main__':
    main()
