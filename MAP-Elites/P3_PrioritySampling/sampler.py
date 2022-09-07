import numpy as np

class Sampler:
    def __init__(self, capacity, alpha):
        self.individuals = np.array([None] * capacity)
        self.counter = 0
        exponents = np.array([i**(-alpha) for i in range(1, capacity+1)])
        self.pdb = exponents / sum(exponents)

    '''
    TODO: downweigh individual if improvement is flat
    '''
    def sample(self):
        return np.random.choice(a=self.individuals, p=self.pdb)

    def add(self, individual):
        idx = np.where(self.individuals == individual)[0]
        if len(idx) != 0:
            self.individuals[idx[0]] = individual
        else:
            self.individuals[self.counter] = individual
            self.counter += 1
        self.individuals = np.array(sorted(self.individuals,
            key=lambda x: float('-inf') if x is None else x.fitness,
            reverse=True))
