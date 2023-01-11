from __future__ import annotations
from typing import Callable, Optional
import numpy as np
from numpy.random import default_rng

rng = default_rng(12345)


class Cell(object):
    N_BASES = 4

    def __init__(self, dna_sequence: np.ndarray, mutation_rate: float, n_bases=N_BASES):
        self.dna_sequence = dna_sequence
        self.mutation_rate = mutation_rate
        self.n_bases = n_bases

    def mutate(self):
        p_mutation_per_change = self.mutation_rate / (self.n_bases - 1)
        p = [1 - self.mutation_rate,
             p_mutation_per_change,
             p_mutation_per_change,
             p_mutation_per_change]
        mutations = rng.choice(self.n_bases, self.dna_sequence.shape, p=p)
        self.dna_sequence = (self.dna_sequence + mutations) % self.n_bases

    @classmethod
    def from_cell(cls, cell: Cell):
        return Cell(cell.dna_sequence.copy(), cell.mutation_rate)

    def divide(self, n_offspring: int = 2):
        cells = [Cell.from_cell(self) for n in range(n_offspring)]
        for cell in cells:
            cell.mutate()
        return cells


class FitnessCell(Cell):
    def __init__(self, dna_sequence: np.ndarray, mutation_rate: float, n_bases=Cell.N_BASES,
                 ref_dna_sequence: Optional[np.ndarray] = None):
        super().__init__(dna_sequence, mutation_rate, n_bases=n_bases)
        if ref_dna_sequence is None:
            ref_dna_sequence = np.arange(len(dna_sequence)) % n_bases
        self.ref_dna_sequence = ref_dna_sequence

    @classmethod
    def genetic_fitness(cls, cell: FitnessCell):
        return cls.genetic_similarity_fitness(cell.dna_sequence, cell.ref_dna_sequence)

    @staticmethod
    def genetic_similarity_fitness(dna_sequence1: np.ndarray, dna_sequence2: np.ndarray,
                                   base_fitness: float = 2.0, fitness_boost: float = 1.0):
        similarity = (dna_sequence1 == dna_sequence2).sum() / len(dna_sequence1)
        fitness = base_fitness + similarity * fitness_boost
        return fitness

    def divide(self, fitness_function: Optional[Callable] = None):
        if fitness_function is None:
            fitness = self.genetic_fitness(self)
        else:
            fitness = fitness_function(self)
        n_offspring = rng.choice(np.arange(int(fitness), int(fitness) + 2), p=[1 - (fitness % 1), fitness % 1])
        return super().divide(n_offspring=n_offspring)

    @classmethod
    def from_cell(cls, cell: FitnessCell):
        return FitnessCell(cell.dna_sequence.copy(), cell.mutation_rate, n_bases=cell.n_bases,
                           ref_dna_sequence=cell.ref_dna_sequence)


def main():
    DNA_SIZE = int(1e6)
    mutation_rate = 3e-6
    dna_sequence = rng.choice(Cell.N_BASES, DNA_SIZE)
    cell = FitnessCell(dna_sequence, mutation_rate)
    children = cell.divide()
    for n, child in enumerate(children):
        n_mutations = (child.dna_sequence != cell.dna_sequence).sum()
        print(f'The number of mutations for child {n} is: {n_mutations}')
    pass


if __name__ == '__main__':
    main()
