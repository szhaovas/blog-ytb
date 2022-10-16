import copy
import numpy as np
from math import pi

class Individual:
    max_possible_period = pi / 10
    def __init__(self, genome):
        self.bds = None
        self.discretized_bds = None
        self.genome = genome
        self.genome[self.genome < 0] = 0
        self.genome[self.genome > 1] = 1
        self.fitness = float('-inf')

    @classmethod
    def random_init(cls, num_joints):
        return cls(np.random.uniform(0, 1, num_joints*3))

    def __select_action__(self, t):
        n_joints = int(self.genome.size / 3)
        ampl = copy.deepcopy(self.genome[ : n_joints])
        period = copy.deepcopy(self.genome[n_joints : 2*n_joints])
        phase = copy.deepcopy(self.genome[2*n_joints : ])

        # amplitude scale: [0, 1], no need to scale

        # period scale: [0, max_possible_period], scale by max_possible_period
        period *= self.max_possible_period

        # phase shift scale: [-2*pi/max_possible_period, 2*pi/max_possible_period]
        #   center on 0.5 and scale by 2*pi/max_possible_period
        phase -= 0.5
        phase *= 2*pi/self.max_possible_period

        return ampl * np.sin(period * (t-phase))

    @staticmethod
    def eval(idv, env, random_seed):
        state = env.reset()
        env.seed(random_seed)
        done = False
        while not done:
            action = idv.__select_action__(env.T)
            new_state, _, done, _ = env.step(action)
            state = new_state
        return env.desc, env.tot_reward

    @staticmethod
    def mutate(this, sigma1):
        return Individual(np.random.normal(this.genome, sigma1))

    # '''
    # Discovering the elite hypervolume by leveraging interspecies correlation (Vassiliades & Mouret, 2018)
    # TODO: fix covariance scale
    # '''
    # @staticmethod
    # def crossover(this, other, sigma1, sigma2):
    #     new_genome = this.genome + \
    #             sigma1 * np.random.normal(0, 1, this.genome.size) + \
    #             sigma2 * (other.genome - this.genome) * np.random.normal(0, 1)
    #     return Individual(new_genome)

    def __hash__(self):
        return hash(tuple(self.discretized_bds))

    def __eq__(self, other):
        return np.all(self.discretized_bds == other.discretized_bds)
