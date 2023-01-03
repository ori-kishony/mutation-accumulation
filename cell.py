import numpy as np
import copy
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
        self.dna_sequence = (self.dna_sequence + mutations) % self.n_bases\

    @classmethod
    def from_cell(cls, cell):
        return Cell(cell.dna_sequence.copy(), cell.mutation_rate)


    def divide(self, n_offspring: int = 2):
        cells = [Cell.from_cell(self) for n in range(n_offspring)]
        for cell in cells:
            cell.mutate()
        return cells

def main():
    DNA_SIZE = int(1e6)
    mutation_rate = 1e-5
    dna_sequence = rng.choice(Cell.N_BASES, DNA_SIZE)
    cell = Cell(dna_sequence, mutation_rate)
    children = cell.divide()
    for n, child in enumerate(children):
        n_mutations = (child.dna_sequence != cell.dna_sequence).sum()
        print(f'The number of mutations for child {n} is: {n_mutations}')
    pass

if __name__ == '__main__':
    main()