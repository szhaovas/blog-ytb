import gym
import QDgym
import numpy as np
from archive import Archive
from individual import Individual

class Agent:
    env_id = 'QDHalfCheetahBulletEnv-v0'
    num_joints = 6
    sigma1 = 0.05
    random_seed = 0
    def __init__(self, archive):
        self.archive = archive
        self.env = gym.make(self.env_id)
        self.env.seed(self.random_seed)

    def warmup(self, nsteps):
        for i in range(nsteps):
            rand_idv = Individual.random_init(self.num_joints)
            bds, fitness = Individual.eval(rand_idv, self.env)

            print(f'Warmup Iteration: {i}; Fitness: {fitness}')

            rand_idv.bds, rand_idv.fitness = bds, fitness
            self.archive.add(rand_idv)

    def evolve(self, nsteps):
        for i in range(nsteps):
            rand_idv = self.archive.sample()
            while rand_idv is None:
                rand_idv = self.archive.sample()
            new_idv = Individual.mutate(rand_idv, self.sigma1)

            bds, fitness = Individual.eval(new_idv, self.env)

            print(f'Evolve Iteration: {i}; Fitness: {fitness}')

            new_idv.bds, new_idv.fitness = bds, fitness
            self.archive.add(new_idv)

    def render_best(self):
        best_idv = self.archive.find_best()
        env = gym.make(self.env_id, render=True)
        env.seed(self.random_seed)
        env.reset()
        Individual.eval(best_idv, env)
        env.close()
        Archive.visualize(self.archive)
        print(f'Best Fitness: {best_idv.fitness}')

if __name__ == '__main__':
    archive = Archive([(0, 1, 0.1)] * 2)
    agent = Agent(archive)
    agent.warmup(10)
    agent.evolve(10)
    agent.render_best()
