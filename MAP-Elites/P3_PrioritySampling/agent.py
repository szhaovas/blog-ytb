import gym
import QDgym
import numpy as np
from archive import Archive
from individual import Individual

import multiprocessing

class Agent:
    env_id = 'QDAntBulletEnv-v0'
    num_joints = 8
    sigma1 = 0.05
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
                zip(*self.pool.starmap(Individual.eval, [*zip(to_eval, self.envs)]))

            print(f'Warmup Iteration: {i*self.num_cores}; Fitness: {fitness}')

            for idv, b, f in zip(to_eval, bds, fitness):
                idv.bds, idv.fitness = b, f
                self.archive.add(idv)

    def evolve(self, nsteps):
        for i in range(nsteps // self.num_cores):
            try:
                to_eval = []
                for _ in range(self.num_cores):
                    rand_idv = self.archive.sample()
                    while rand_idv is None:
                        rand_idv = self.archive.sample()

                    new_idv = Individual.mutate(rand_idv, self.sigma1)
                    to_eval.append(new_idv)

                bds, fitness = \
                    zip(*self.pool.starmap(Individual.eval, [*zip(to_eval, self.envs)]))

                print(f'Evolve Iteration: {i*self.num_cores}; Fitness: {fitness}')

                for idv, b, f in zip(to_eval, bds, fitness):
                    idv.bds, idv.fitness = b, f
                    self.archive.add(idv)
            except KeyboardInterrupt:
                filename = f'{self.env_id}_{i*self.num_cores}.p'
                self.archive.save(filename)
                qt = input('Quit? (y)')
                if qt == 'y':
                    quit()
                else:
                    continue
        filename = f'{self.env_id}_{nsteps}.p'
        self.archive.save(filename)

    def render_best(self):
        best_idv = self.archive.find_best()
        env = gym.make(self.env_id, render=True)
        env.seed(self.random_seed)
        env.reset()
        Individual.eval(best_idv, env)
        env.close()
        try:
            Archive.visualize(self.archive)
        except:
            pass
        print(f'Best Fitness: {best_idv.fitness}')

if __name__ == '__main__':
    '''use this if train from scratch
    '''
    archive = Archive([(0, 1, 0.1)] * 4)
    ####################################
    num_cores = multiprocessing.cpu_count()
    agent = Agent(archive, num_cores)
    ####################################
    agent.warmup(10)
    agent.evolve(10)
    agent.render_best()

    '''use this if resume from previously trained archive
    '''
    # archive = Archive.from_pickle('QDAntBulletEnv-v0_500.p')
    # num_cores = multiprocessing.cpu_count()
    # agent = Agent(archive, num_cores)
    # agent.evolve(10)
    # agent.render_best()
