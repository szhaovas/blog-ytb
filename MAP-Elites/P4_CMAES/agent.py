import gym
import QDgym
import numpy as np
from archive import Archive
from individual import Individual

import time
import multiprocessing

import cma

def input_flush(prompt):
    for char in prompt:
        time.sleep(0.1)
        print(char, end='', flush=True)
    return input()

class Agent:
    env_id = 'QDAntBulletEnv-v0'
    num_joints = 8
    cmaes_sigma1 = 0.2
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
        starting_mean = self.archive.importance_sample(1)[0].genome
        self.es = cma.CMAEvolutionStrategy(
            starting_mean,
            self.cmaes_sigma1,
            {'popsize': self.num_cores-1}
        )

        counter = 0
        self.restarts = 0
        while counter < nsteps:
            print(f'Evolve Iteration: {counter}')
            # if cmaes instance reaches terminating condition, restart from
            #   new cell
            if self.es.stop():
                starting_mean = self.archive.importance_sample(1)[0].genome
                self.es = cma.CMAEvolutionStrategy(
                    starting_mean,
                    self.cmaes_sigma1,
                    {'popsize': self.num_cores-1}
                )
                self.restarts += 1

            parent = Individual(self.es.mean)
            new_genomes = self.es.ask()
            to_eval = [parent] + [Individual(g) for g in new_genomes]
            bds, fitness = \
                zip(*self.pool.starmap(Individual.eval, [*zip(to_eval, self.envs, [self.random_seed]*self.num_cores)]))

            print(f'Fitness: {fitness}')
            parent.bds, parent.fitness = bds[0], fitness[0]
            self.archive.add(parent)

            improvement_scores = []
            for idv, b, f in zip(to_eval[1:], bds[1:], fitness[1:]):
                idv.bds, idv.fitness = b, f
                score = self.archive.add(idv, parent)
                # pycma minimizes objective by default
                improvement_scores.append(-score)

            print(f'Scores: {improvement_scores}')

            self.es.tell(new_genomes, improvement_scores)
            counter += self.num_cores
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
        print(f'Restarts: {self.restarts}')

        print('Histogram:')
        individuals = [i.fitness for i in [*self.archive.archive.values()]]
        bins, itvls = np.histogram(individuals)
        hist_dict = {}
        for i in range(len(bins)):
            itvl = f'{itvls[i]}~{itvls[i+1]}'
            hist_dict[itvl] = bins[i]
        print('\n'.join(f'{key}: {value}' for key, value in hist_dict.items()))

if __name__ == '__main__':
    '''use this if train from scratch
    '''
    # archive = Archive([(0, 1, 0.05)] * 4)
    # num_cores = 50
    # agent = Agent(archive, num_cores)
    # agent.warmup(50)
    # agent.evolve(1000)
    # agent.render_best()

    '''use this if resume from previously trained archive
    '''
    archive = Archive.from_pickle('QDAntBulletEnv-v0_1000.p')
    num_cores = 50
    agent = Agent(archive, num_cores)
    agent.evolve(1000)
    agent.render_best()
