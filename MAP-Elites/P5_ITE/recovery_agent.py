import gym
import QDgym
import numpy as np
from archive import Archive
from mboa import MBOAArchive
from individual import Individual

import time
import multiprocessing

class RecoveryAgent:
    env_id = 'QDAntDamagedBulletEnv-v0'
    num_joints = 8
    random_seed = 0
    def __init__(self, prior_archive, num_cores, alpha=0.9):
        self.archive = MBOAArchive(prior_archive, render=True)
        self.alpha = alpha

        self.num_cores = num_cores
        self.envs = [gym.make(self.env_id) for _ in range(num_cores)]
        [env.seed(self.random_seed) for env in self.envs]
        self.pool = multiprocessing.Pool(num_cores)

        self.best_genome = None
        self.best_fitness = float('-inf')
        self.best_fitness_prior = prior_archive.find_best().fitness

    def recover(self, max_iter=1000):
        counter = 0
        while self.best_fitness < self.alpha*self.best_fitness_prior and counter < max_iter:
            genomes = self.archive.ask(self.num_cores)
            to_eval = [Individual(g) for g in genomes]

            _, fitness = \
                zip(*self.pool.starmap(Individual.eval, [*zip(to_eval, self.envs, [self.random_seed]*self.num_cores)]))

            print(f'Recover Iteration: {counter}')
            print(f'Fitness: {fitness}')
            self.archive.tell(genomes, np.array(fitness).flatten())
            iter_best_idx = np.argmax(fitness)
            if fitness[iter_best_idx] > self.best_fitness:
                self.best_genome = genomes[iter_best_idx, :]
                self.best_fitness = fitness[iter_best_idx]

            counter += self.num_cores

    def render_best(self):
        best_idv = Individual(self.best_genome)
        env = gym.make(self.env_id, render=True)
        env.reset()
        env.seed(self.random_seed)
        Individual.eval(best_idv, env, self.random_seed)
        env.close()

if __name__ == '__main__':
    archive = Archive.from_pickle('QDAntBulletEnv-v0_10000.p')
    num_cores = multiprocessing.cpu_count()
    agent = RecoveryAgent(archive, num_cores)
    agent.recover(1000)
    agent.render_best()
