import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from archive import discretize

'''
x1: 1*D np.array
x2: N*D np.array
returns: 1*N np.array
'''
def Matern52(x1, x2, rho):
    assert x1.shape[1] == x2.shape[1]

    sqrt5 = np.sqrt(5)
    dist = np.linalg.norm(x1 - x2, axis=1)
    return (1 + sqrt5*dist/rho + (5*dist**2)/(3*rho**2)) * np.exp(-sqrt5*dist/rho)

class MBOAArchive:
    '''
    prior_archive: archive object prior to adaptation
    sigma2:        feedback noise in adaptation environment
    rho:           parameter for Matern52 kernel; determines "wiggliness"
    kappa:         how much to value uncertainty when returning X to be tested
    render:        if true plots visualization of archive each time after tell()
    '''
    def __init__(self, prior_archive, sigma2=0.001, rho=0.4, kappa=0.05, render=False):
        self.archive = {}
        for idv in prior_archive.archive.values():
            if render:
                self.archive[tuple(idv.genome)] = (idv.fitness, 1, idv.discretized_bds)
            else:
                self.archive[tuple(idv.genome)] = (idv.fitness, 1)

        self.sigma2 = sigma2
        self.rho = rho
        self.kappa = kappa

        self.tested_X = np.array([])
        self.tested_fitness = np.array([])

        self.render = render
        if render:
            self.dim1 = prior_archive.dims[0].tolist()
            self.dim1 += [self.dim1[-1]+ \
                            self.dim1[-1]- \
                            self.dim1[-2]]
            self.dim1 = np.array(self.dim1)
            self.dim2 = prior_archive.dims[1].tolist()
            self.dim2 += [self.dim2[-1]+ \
                            self.dim2[-1]- \
                            self.dim2[-2]]
            self.dim2 = np.array(self.dim2)

            self.fitness_mat = np.zeros((self.dim1.size, self.dim2.size))
            for idv in prior_archive.archive.values():
                index1 = np.where(idv.discretized_bds[0] == prior_archive.dims[0])[0][0]
                index2 = np.where(idv.discretized_bds[1] == prior_archive.dims[1])[0][0]
                self.fitness_mat[index1, index2] = idv.fitness

            self.z_min = 0
            self.z_max = max(np.max(self.fitness_mat), -self.z_min)

    def ask(self, batch_size):
        results = [None] * batch_size
        maxvals = [float('-inf')] * batch_size
        for x, state in self.archive.items():
            mu, var = state[0], state[1]
            val = mu + self.kappa*var
            for i, mv in enumerate(maxvals):
                if val > mv:
                    results[i] = x
                    maxvals[i] = val
                    break

        results = np.array(list(map(list, results)))
        return results

    '''
    X: N*D np.array
    P: N, np.array
    '''
    def tell(self, X, P):
        assert X.shape[0] == P.size
        self.tested_X = np.vstack([self.tested_X, X]) if self.tested_X.size else X
        self.tested_fitness = np.concatenate((self.tested_fitness, P))

        num_tests = self.tested_X.shape[0]
        K = np.zeros((num_tests, num_tests))
        for r in range(num_tests):
            K[r,:] = Matern52(self.tested_X[r,:].reshape(1, -1), self.tested_X, self.rho)
        K += self.sigma2*np.eye(num_tests)
        K_inv = np.linalg.inv(K)
        error = self.tested_fitness - np.array([self.archive[tuple(x)][0] for x in self.tested_X])

        for x, state in self.archive.items():
            mu, var = state[0], state[1]
            k = Matern52(np.array(x).reshape(1, -1), self.tested_X, self.rho)
            new_mu = mu + k.T @ K_inv @ error
            new_var = var - k.T @ K_inv @ k
            if self.render:
                discretized_bds = state[2]
                self.archive[x] = (new_mu, new_var, discretized_bds)
                index1 = np.where(discretized_bds[0] == self.dim1)[0][0]
                index2 = np.where(discretized_bds[1] == self.dim2)[0][0]
                self.fitness_mat[index1, index2] = new_mu
            else:
                self.archive[x] = (new_mu, new_var)

        if self.render:
            plt.cla()
            fig, ax = plt.subplots()
            c = ax.pcolormesh(self.dim1, self.dim2, self.fitness_mat, \
                                cmap='RdBu', vmin=self.z_min, vmax=self.z_max)
            ax.axis([np.min(self.dim1), np.max(self.dim1), \
                        np.min(self.dim2), np.max(self.dim2)])
            fig.colorbar(c, ax=ax)
            # re-enables this if archive is 2D
            # for genome in X:
            #     discretized_bds = self.archive[tuple(genome)][2]
            #     ax.add_patch(Circle((discretized_bds[1],discretized_bds[0]),0.005,color='yellow'))
            plt.draw()
            plt.pause(0.001)

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
