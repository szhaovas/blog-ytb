import numpy as np
from math import pi

class Individual:
    T_divby = 10
    max_possible_freq = pi
    def __init__(self, genome):
        self.bds = None
        self.discretized_bds = None
        self.genome = genome
        self.fitness = float('-inf')

    @classmethod
    def random_init(cls, num_joints):
        ampl = np.random.uniform(0, 1, num_joints)
        freq = np.random.uniform(0, cls.max_possible_freq, num_joints)
        phase = np.random.uniform(0, 2*cls.max_possible_freq/freq, num_joints)
        genome = np.concatenate((ampl, freq, phase))
        return cls(genome)

    def __select_action__(self, t):
        n_joints = int(self.genome.size / 3)
        ampl = self.genome[ : n_joints]
        freq = self.genome[n_joints : 2*n_joints]
        phase = self.genome[2*n_joints : ]
        return ampl * np.sin(freq * (t-phase))

    @staticmethod
    def eval(idv, env):
        state = env.reset()
        done = False
        while not done:
            action = idv.__select_action__(env.T / idv.T_divby)
            new_state, _, done, _ = env.step(action)
            state = new_state
        return env.desc, env.tot_reward

    @staticmethod
    def mutate(this, sigma1):
        return Individual(np.random.normal(this.genome, sigma1))

    def __eq__(self, other):
        return np.all(self.discretized_bds == other.discretized_bds)

    def __hash__(self):
        string = f"{['{:0.2f}'.format(i) for i in self.discretized_bds]}"
        return hash(string)
