from cell import Cell
from numpy.random import default_rng

rng = default_rng(12345)

class Experiment(object):
    DEFAULT_BOTTLENECK_SIZE = 1
    def __init__(self, cells: list[Cell]):
        self.cells = cells

    def do_generation(self):
        new_generation = (c.divide() for c in self.cells)
        self.cells = (c for children in new_generation for c in children)

    def bottleneck(self, bottleneck_size: int = DEFAULT_BOTTLENECK_SIZE):
        self.cells = rng.choice(self.cells, size=bottleneck_size, replace=False)

    def grow(self, n_generations):
        for g in range(n_generations):
            self.do_generation()
        self.cells = list(self.cells)

    def grow_bottleneck_cycle(self, n_generations, bottleneck_size: int = DEFAULT_BOTTLENECK_SIZE):
        self.grow(n_generations)
        self.bottleneck(bottleneck_size)

    def do_full_experiment(self, n_cycles, n_generations, bottleneck_size: int = DEFAULT_BOTTLENECK_SIZE):
        for cycle in range(n_cycles):
            print(f'cycle: {cycle}')
            self.grow_bottleneck_cycle(n_generations, bottleneck_size)


def main():
    DNA_SIZE = int(1e5)
    mutation_rate = 1e-6
    n_cycles = 10
    n_generations = 8
    bottleneck_size = 1
    dna_sequence = rng.choice(Cell.N_BASES, DNA_SIZE)
    cell = Cell(dna_sequence, mutation_rate)
    cells = [cell]
    experiment = Experiment(cells)
    experiment.do_full_experiment(n_cycles, n_generations, bottleneck_size)
    cell_final = experiment.cells[0]

    n_mutations = (cell_final.dna_sequence != cell.dna_sequence).sum()
    print(f'The number of mutations for final cell is: {n_mutations}')

if __name__ == '__main__':
    main()