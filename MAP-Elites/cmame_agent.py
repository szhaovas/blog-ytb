import gym
import QDgym
import numpy as np
from individual import Individual

import multiprocessing

from ribs.archives import GridArchive
from ribs.emitters import ImprovementEmitter
from ribs.optimizers import Optimizer
from ribs.visualize import grid_archive_heatmap

import matplotlib.pyplot as plt

class CMAESAgent:
    env_id = 'QDAntBulletEnv-v0'
    num_joints = 8
    random_seed = 0
    def __init__(self):
        self.env = gym.make(self.env_id)
        self.archive = GridArchive(
            [20] * 4,
            [(0, 1)] * 4
        )
        self.emitters = [
            ImprovementEmitter(
                self.archive,
                np.array([0.0] * self.num_joints * 3),
                1.0,
                batch_size=5
            ) for _ in range(5)  # Create 5 separate emitters.
        ]
        self.optimizer = Optimizer(self.archive, self.emitters)

    def evolve(self, nsteps):
        for itr in range(nsteps):
            to_eval = self.optimizer.ask()
            to_eval_idv = [Individual(g) for g in to_eval]

            bds, fitness = [], []
            for i in to_eval_idv:
                b, f = Individual.eval(i, self.env)
                bds.append(b)
                fitness.append(f)

            print(f'Evolve Iteration: {itr}; Fitness: {fitness}', flush=True)

            for i,b,f in zip(to_eval_idv, bds, fitness):
                i.bds, i.fitness = b, f
            self.optimizer.tell(fitness, bds)

    def render_best(self):
        df = self.archive.as_pandas()
        high_perf_sols = df.sort_values("objective", ascending=False)
        best_idv = Individual(high_perf_sols[0])

        env = gym.make(self.env_id, render=True)
        env.seed(self.random_seed)
        env.reset()
        Individual.eval(best_idv, env)
        env.close()

        plt.figure(figsize=(8, 6))
        grid_archive_heatmap(self.archive)
        print(f'Best Fitness: {best_idv.fitness}')

if __name__ == '__main__':
    agent = CMAESAgent()
    agent.evolve(100)
    agent.render_best()
