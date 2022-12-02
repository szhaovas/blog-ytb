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
    cmaes_sigma1 = 0.1
    gradient_estimate_sigma = 0.1
    learning_rate = 1
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

    def eval_and_gradient(self, genome):
        # need 1 env for evaluating genome
        # λ_es needs to be even for mirror sampling
        if (self.num_cores - 1) % 2 == 0:
            half_batch = (self.num_cores - 1) // 2
        else:
            half_batch = (self.num_cores - 2) // 2
        noise = np.random.standard_normal((half_batch, genome.size))
        noisy_genomes = np.repeat([genome], 2*half_batch, axis=0)
        noisy_genomes[:half_batch, :] += self.gradient_estimate_sigma * noise
        noisy_genomes[half_batch:, :] -= self.gradient_estimate_sigma * noise
        to_eval = [Individual(genome)] + [Individual(g) for g in noisy_genomes]
        bds, fitness = \
            zip(*self.pool.starmap(Individual.eval, [*zip(to_eval, self.envs[:len(to_eval)], [self.random_seed]*len(to_eval))]))
        bds, fitness = np.array(bds), np.array(fitness)

        bds_grad = np.zeros((bds.shape[1], genome.size))
        fitness_grad = None
        for j in range(bds.shape[1] + 1):
            if j == 0:
                ranking_indices = np.argsort(fitness[1:])
            else:
                ranking_indices = np.argsort(bds[1:,j-1])

            ranks = np.empty(2*half_batch, dtype=np.int32)
            ranks[ranking_indices] = np.arange(2*half_batch)
            ranks = (ranks / (2*half_batch - 1)) - 0.5

            gradient = np.sum(
                noise * (ranks[:half_batch] - ranks[half_batch:])[:, None],
                axis=0)
            gradient /= half_batch * self.gradient_estimate_sigma

            if j == 0:
                fitness_grad = gradient
            else:
                bds_grad[j-1, :] = gradient

        return fitness[0], fitness_grad, bds[0], bds_grad

    def evolve(self, nsteps):
        genome_centroid = self.archive.importance_sample(1)[0].genome
        self.es = cma.CMAEvolutionStrategy(
            # FIXME
            np.zeros(5),
            self.cmaes_sigma1,
            {'popsize': self.num_cores}
        )

        counter = 0
        self.restarts = 0
        while counter < nsteps:
            print(f'Evolve Iteration: {counter}')
            # if cmaes instance reaches terminating condition, restart from
            #   new cell
            if self.es.stop():
                genome_centroid = self.archive.importance_sample(1)[0].genome
                self.es = cma.CMAEvolutionStrategy(
                    # FIXME
                    np.zeros(5),
                    self.cmaes_sigma1,
                    {'popsize': self.num_cores}
                )
                self.restarts += 1

            fitness, fitness_grad, bds, bds_grad = self.eval_and_gradient(genome_centroid)
            individual_centroid = Individual(genome_centroid)
            individual_centroid.bds, individual_centroid.fitness = bds, fitness
            self.archive.add(individual_centroid)
            print(f'Centroid Fitness: {fitness}')

            fitness_grad /= (np.linalg.norm(fitness_grad) + 1e-8)
            bds_grad /= (np.linalg.norm(bds_grad, axis=1) + 1e-8)[:, None]

            # (nbds+1) * genome.size
            all_grads = np.concatenate((fitness_grad[None], bds_grad))
            # λ * (nbds+1)
            coeffs = np.array(self.es.ask())
            coeffs[:,0] = abs(coeffs[:,0])
            # λ * genome.size
            weighted_grads = coeffs @ all_grads

            cmaes_genomes = np.repeat([genome_centroid], self.num_cores, axis=0)
            cmaes_genomes += weighted_grads
            to_eval = [Individual(g) for g in cmaes_genomes]

            bds, fitness = \
                zip(*self.pool.starmap(Individual.eval, [*zip(to_eval, self.envs, [self.random_seed]*self.num_cores)]))

            print(f'CMAES Fitness: {fitness}')

            improvement_scores = []
            for idv, b, f in zip(to_eval, bds, fitness):
                idv.bds, idv.fitness = b, f
                score = self.archive.add(idv, individual_centroid)
                # pycma minimizes objective by default
                improvement_scores.append(-score)

            print(f'Scores: {improvement_scores}')
            print(f'Average Score: {np.mean(improvement_scores)}')

            self.es.tell(tuple([*coeffs]), improvement_scores)
            counter += self.num_cores

            ranking_indices = np.argsort(improvement_scores)
            sorted_genomes = cmaes_genomes[ranking_indices]
            weights = (np.log(self.num_cores + 0.5) -
                       np.log(np.arange(1, self.num_cores + 1)))
            total_weights = np.sum(weights)
            weights = weights / total_weights
            genome_centroid_cmaes = np.sum(sorted_genomes * np.expand_dims(weights, axis=1), axis=0)
            step = self.learning_rate * (genome_centroid_cmaes - genome_centroid)
            genome_centroid += step

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
    # agent.evolve(1000)
    agent.render_best()
