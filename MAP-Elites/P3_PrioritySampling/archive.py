import numpy as np
from random import choice
import matplotlib.pyplot as plt

import pickle

from sampler import Sampler

def discretize(bd, grid):
    return grid[min(np.digitize(bd, grid), len(grid)-1)]

class Archive:
    sampler_alpha = 0.7
    '''
    dims: [(dim1_low, dim1_high, dim1_step) * len(dims)]
    '''
    def __init__(self, dims):
        def archive_layer(dims):
            if len(dims) > 1:
                layer = {key:archive_layer(dims[1:]) for key in np.arange(dims[0][0], dims[0][1], dims[0][2])}
            else:
                layer = {key:None for key in np.arange(dims[0][0], dims[0][1], dims[0][2])}
            return layer

        self.archive = archive_layer(dims)

        #####################################
        num_grids = 1
        layer = self.archive
        while isinstance(layer, dict):
            keys = list(layer.keys())
            num_grids *= len(keys)
            layer = layer[keys[0]]
        self.sampler = Sampler(num_grids, self.sampler_alpha)
        #####################################

    def add(self, individual):
        layer = self.archive
        for bd in individual.bds[:-1]:
            key = discretize(bd, list(layer.keys()))
            layer = layer[key]

        last_key = discretize(individual.bds[-1], list(layer.keys()))
        if layer[last_key] is None or individual.fitness > layer[last_key].fitness:
            layer[last_key] = individual
            #####################################
            self.sampler.add(individual)
            #####################################

    '''
    might return None
    '''
    def sample(self):
        #####################################
        return self.sampler.sample()
        #####################################

    def find_best(self):
        #####################################
        return self.sampler.individuals[0]
        #####################################

    @staticmethod
    def visualize(archive2D):
        dim1 = list(archive2D.archive.keys())
        dim1 += [dim1[-1]+dim1[-1]-dim1[-2]]
        dim2 = list(list(archive2D.archive.values())[0].keys())
        dim2 += [dim2[-1]+dim2[-1]-dim2[-2]]
        fitness_mat = np.zeros((len(dim1), len(dim2)))
        for dim1_count, layer in enumerate(archive2D.archive.values()):
            for dim2_count, idv in enumerate(layer.values()):
                if idv is not None:
                    fitness_mat[dim1_count, dim2_count] = idv.fitness

        z_min = np.min(fitness_mat)
        z_max = max(np.max(fitness_mat), -z_min)
        fig, ax = plt.subplots()
        c = ax.pcolormesh(dim1, dim2, fitness_mat, cmap='RdBu', vmin=z_min, vmax=z_max)
        ax.axis([np.min(dim1), np.max(dim1), np.min(dim2), np.max(dim2)])
        fig.colorbar(c, ax=ax)

        plt.show()

    #######################################
    @classmethod
    def from_pickle(cls, filename):
        try:
            return pickle.load(open(filename, 'rb'))
        except FileNotFoundError:
            print('Cannot find filename, exiting...')
            quit()
        except _ as err:
            print(err)
            quit()

    def save(self, filename):
        pickle.dump(self, open(filename, 'wb'))
    #######################################