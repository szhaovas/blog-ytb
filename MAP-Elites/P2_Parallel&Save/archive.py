import numpy as np
from random import choice
import matplotlib.pyplot as plt

import pickle

def discretize(bd, grid):
    return grid[min(np.digitize(bd, grid), len(grid)-1)]

class Archive:
    '''
    dims: [(dim1_low, dim1_high, dim1_step) * len(dims)]

    priority_buffer_alpha: how strongly to prefer sampling high fitness
        individuals (range is [0,1] where 0 means fair sampling)
    '''
    def __init__(self, dims, priority_buffer_alpha=0.6):
        self.dims = [np.arange(d[0], d[1], d[2]) for d in dims]
        self.archive = {}

    def add(self, individual):
        discretized_bds = \
            list(map(lambda bd, grid: discretize(bd, grid), individual.bds, self.dims))
        individual.discretized_bds = discretized_bds
        if (not individual in self.archive) or \
            individual.fitness > self.archive[individual].fitness:
            self.archive[individual] = individual

    def sample(self, batch_size):
        return np.random.choice(list(self.archive.values()), batch_size)

    def find_best(self):
        max_fitness = float('-inf')
        best_idv = None
        for idv in self.archive.values():
            if idv.fitness > max_fitness:
                max_fitness = idv.fitness
                best_idv = idv

        return best_idv

    @staticmethod
    def visualize(archive2D):
        dim1 = archive2D.dims[0].tolist()
        dim1 += [dim1[-1]+dim1[-1]-dim1[-2]]
        dim2 = archive2D.dims[1].tolist()
        dim2 += [dim2[-1]+dim2[-1]-dim2[-2]]
        fitness_mat = np.zeros((len(dim1), len(dim2)))
        for idv in archive2D.archive.values():
            index1 = np.where(idv.discretized_bds[0] == archive2D.dims[0])[0]
            index2 = np.where(idv.discretized_bds[1] == archive2D.dims[1])[0]
            fitness_mat[index1, index2] = idv.fitness

        z_min = np.min(fitness_mat)
        z_max = max(np.max(fitness_mat), -z_min)
        fig, ax = plt.subplots()
        c = ax.pcolormesh(dim1, dim2, fitness_mat, cmap='RdBu', vmin=z_min, vmax=z_max)
        ax.axis([np.min(dim1), np.max(dim1), np.min(dim2), np.max(dim2)])
        fig.colorbar(c, ax=ax)

        plt.show()

    @classmethod
    def from_pickle(cls, filename):
        try:
            return pickle.load(open(filename, 'rb'))
        except FileNotFoundError:
            print('Cannot find filename, exiting...')
            quit()
        except Exception as err:
            print(err)
            quit()

    def save(self, filename):
        pickle.dump(self, open(filename, 'wb'))
