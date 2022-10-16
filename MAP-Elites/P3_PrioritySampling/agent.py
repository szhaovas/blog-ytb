import gym
import QDgym
import numpy as np
from archive import Archive
from individual import Individual

import time
import multiprocessing

def input_flush(prompt):
    for char in prompt:
        time.sleep(0.1)
        print(char, end='', flush=True)
    return input()

class Agent:
    env_id = 'QDAntBulletEnv-v0'
    num_joints = 8
    mutate_sigma1 = 0.005
    random_seed = 0
    def __init__(self, archive, num_cores):
        self.archive = archive

        self.num_cores = num_cores
        self.envs = [gym.make(self.env_id) for _ in range(num_cores)]
        [env.seed(self.random_seed) for env in self.envs]
        self.pool = multiprocessing.Pool(num_cores)

    def warmup(self, nsteps):
        for i in range(nsteps // self.num_cores):
            to_eval = \
                [Individual.random_init(self.num_joints) for _ in range(self.num_cores)]
            bds, fitness = \
                zip(*self.pool.starmap(Individual.eval, [*zip(to_eval, self.envs, [self.random_seed]*self.num_cores)]))

            print(f'Warmup Iteration: {i*self.num_cores}; Fitness: {fitness}')

            for idv, b, f in zip(to_eval, bds, fitness):
                idv.bds, idv.fitness = b, f
                self.archive.add(idv)

    def evolve(self, nsteps):
        for i in range(nsteps // self.num_cores):
            try:
                print(f'Evolve Iteration: {i*self.num_cores}')

                parents = self.archive.importance_sample(self.num_cores)
                to_eval = [Individual.mutate(idv, self.mutate_sigma1) \
                            for idv in parents]

                bds, fitness = \
                    zip(*self.pool.starmap(Individual.eval, [*zip(to_eval, self.envs, [self.random_seed]*self.num_cores)]))

                print(f'Fitness: {fitness}')

                for idv, b, f, p in zip(to_eval, bds, fitness, parents):
                    idv.bds, idv.fitness = b, f
                    self.archive.add(idv, p)
            except KeyboardInterrupt:
                filename = f'{self.env_id}_{i*self.num_cores}.p'
                self.archive.save(filename)
                qt = input_flush('Quit? (y)')
                if qt == 'y':
                    quit()
                else:
                    continue
        filename = f'{self.env_id}_{nsteps}.p'
        self.archive.save(filename)

    def render_best(self):
        best_idv = self.archive.find_best()
        env = gym.make(self.env_id, render=True)
        env.reset()
        env.seed(self.random_seed)
        Individual.eval(best_idv, env, self.random_seed)
        env.close()
        Archive.visualize(self.archive)
        print(f'Best fitness: {best_idv.fitness}')
        print(f'Cells discovered: {len([*self.archive.archive.keys()])}')

if __name__ == '__main__':
    '''use this if train from scratch
    '''
    archive = Archive([(0, 1, 0.05)] * 4)
    num_cores = multiprocessing.cpu_count()
    agent = Agent(archive, num_cores)
    agent.warmup(50)
    agent.evolve(1000)
    agent.render_best()

    '''use this if resume from previously trained archive
    '''
    # archive = Archive.from_pickle('QDAntBulletEnv-v0_1000.p')
    # num_cores = multiprocessing.cpu_count()
    # agent = Agent(archive, num_cores)
    # agent.evolve(1000)
    # agent.render_best()
