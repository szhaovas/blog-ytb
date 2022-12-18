import gym
import QDgym
import numpy as np
from archive import Archive
from individual import NNIndividual
from replay_buffer import ReplayBuffer
from networks import PolicyNetwork, DQCriticNetwork
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MSE
from tensorflow.keras.models import load_model
from itertools import chain
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import pickle

import time
import multiprocessing
# https://pythonspeed.com/articles/python-multiprocessing/
from multiprocessing import get_context

import cma

def input_flush(prompt):
    for char in prompt:
        time.sleep(0.1)
        print(char, end='', flush=True)
    return input()

class Agent:
    env_id = 'QDAntBulletEnv-v0'
    Ashape = (28, 64, 64, 8)
    Cshape = (28, 128, 128, 1)
    buffer_size = 100000
    Adam_learning_rate = 0.001
    # gradient_batch_size = 65536
    gradient_batch_size = 1024
    TD3_train_niters = 100
    TD3_train_batchsize = 256
    TD3_train_smoothing_std = 0.4
    TD3_train_smoothing_clip = 0.5
    TD3_discount = 0.99
    TD3_target_update_freq = 2
    TD3_target_update_rate = 0.03

    cmaes_sigma1 = 0.1
    centroid_learning_rate = 1
    random_seed = 0
    def __init__(self, archive, num_cores):
        self.archive = archive

        self.num_cores = num_cores
        self.envs = [gym.make(self.env_id) for _ in range(num_cores)]
        [env.seed(self.random_seed) for env in self.envs]

        self.state_dims = self.Ashape[0]
        self.action_dims = self.Ashape[-1]
        self.bd_dims = len(archive.dims)
        self.min_action = self.envs[0].action_space.low[0]
        self.max_action = self.envs[0].action_space.high[0]
        self.__reinitialize_TD3__()

    def warmup(self, nsteps):
        for i in range(nsteps // self.num_cores):
            to_eval = \
                [NNIndividual.random_init(self.Ashape) for _ in range(self.num_cores)]
            with get_context('spawn').Pool() as pool:
                bds, fitness, transitions = \
                    zip(*pool.starmap(NNIndividual.eval, [*zip(to_eval, self.envs, [self.random_seed]*self.num_cores)]))

            print(f'Warmup Iteration: {i*self.num_cores}; Fitness: {fitness}')

            for episode in transitions:
                for state, action, reward, next_state, done, bd in episode:
                    self.buffer.store_transition(state, action, reward, next_state, done, bd)

            for idv, b, f in zip(to_eval, bds, fitness):
                idv.bds, idv.fitness = b, f
                self.archive.add(idv)

    def __reinitialize_TD3__(self):
        self.buffer = ReplayBuffer(self.buffer_size, self.state_dims, self.action_dims, self.bd_dims)
        # DON'T confuse with actor policies in archive
        # These are only for gradient estimation
        self.gradient_actors = [PolicyNetwork(self.Ashape) for _ in range(1+self.bd_dims)]
        [p.compile(optimizer=Adam(learning_rate=self.Adam_learning_rate)) for p in self.gradient_actors]
        self.target_gradient_actors = [PolicyNetwork(self.Ashape) for _ in range(1+self.bd_dims)]
        for a, ta in zip(self.gradient_actors, self.target_gradient_actors):
            ta.set_weights(a.get_weights())

        # DDQN: take the min of the critics to avoid overestimation
        self.gradient_critics = [DQCriticNetwork(self.Cshape, self.action_dims) for _ in range(1+self.bd_dims)]
        [p.compile(optimizer=Adam(learning_rate=self.Adam_learning_rate)) for p in self.gradient_critics]
        self.target_gradient_critics = [DQCriticNetwork(self.Cshape, self.action_dims) for _ in range(1+self.bd_dims)]
        for c1, tc1 in zip(self.gradient_critics, self.target_gradient_critics):
            tc1.set_weights(c1.get_weights())

        self.gradient_critics_alt = [DQCriticNetwork(self.Cshape, self.action_dims) for _ in range(1+self.bd_dims)]
        [p.compile(optimizer=Adam(learning_rate=self.Adam_learning_rate)) for p in self.gradient_critics_alt]
        self.target_gradient_critics_alt = [DQCriticNetwork(self.Cshape, self.action_dims) for _ in range(1+self.bd_dims)]
        for c2, tc2 in zip(self.gradient_critics_alt, self.target_gradient_critics_alt):
            tc2.set_weights(c2.get_weights())

    def TD3_gradient(self, genome):
        actor = PolicyNetwork(self.Ashape, False, genome)
        states, _, _, _, _, _ = \
            self.buffer.sample_buffer(self.gradient_batch_size)
        states = tf.convert_to_tensor(states, dtype=tf.float32)

        jacobian = []
        for c in self.gradient_critics:
            with tf.GradientTape() as tape:
                actor_loss = tf.reduce_mean(-c(states, actor(states)))
            gradient = tape.gradient(actor_loss, actor.trainable_variables)
            jacobian.append(np.array([*chain.from_iterable([i.numpy().flatten() for i in gradient])]))
        return np.array(jacobian)

    def train_TD3(self):
        for niters in range(self.TD3_train_niters):
            states, actions, rewards, next_states, done, bds = \
                self.buffer.sample_buffer(self.TD3_train_batchsize)

            states = tf.convert_to_tensor(states, dtype=tf.float32)
            actions = tf.convert_to_tensor(actions, dtype=tf.float32)
            rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
            next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)

            for i, (a, ta, c1, tc1, c2, tc2) in enumerate(zip(
                self.gradient_actors, self.target_gradient_actors,
                self.gradient_critics, self.target_gradient_critics,
                self.gradient_critics_alt, self.target_gradient_critics_alt
            )):
                # train critic
                with tf.GradientTape(persistent=True) as tape:
                    noise = tf.clip_by_value(
                        tf.random.normal(
                            shape=(self.TD3_train_batchsize, self.action_dims),
                            stddev=self.TD3_train_smoothing_std
                        ),
                        clip_value_min=-self.TD3_train_smoothing_clip,
                        clip_value_max=self.TD3_train_smoothing_clip
                    )
                    next_actions = tf.clip_by_value(
                        ta(next_states) + noise,
                        clip_value_min=self.min_action,
                        clip_value_max=self.max_action
                    )

                    target_q1, target_q2 = tc1(next_states, next_actions), tc2(next_states, next_actions)
                    target_q = tf.math.minimum(target_q1, target_q2)

                    r = rewards if i == 0 else bds[:, i-1]

                    target_q = (r[:, None] + (1.0 - done[:, None]) *
                                self.TD3_discount * target_q)

                    current_q1, current_q2 = c1(states, actions), c2(states, actions)

                    c1_loss, c2_loss = MSE(current_q1, target_q), MSE(current_q2, target_q)

                c1_gradient = tape.gradient(c1_loss, c1.trainable_variables)
                c1.optimizer.apply_gradients(zip(c1_gradient, c1.trainable_variables))
                c2_gradient = tape.gradient(c2_loss, c2.trainable_variables)
                c2.optimizer.apply_gradients(zip(c2_gradient, c2.trainable_variables))
                del tape

                # train actor every TD3_target_update_freq iters
                if niters % self.TD3_target_update_freq == 0:
                    with tf.GradientTape() as tape:
                        actor_loss = tf.reduce_mean(-c1(states, a(states)))

                    a_gradient = tape.gradient(actor_loss, a.trainable_variables)
                    a.optimizer.apply_gradients(zip(a_gradient, a.trainable_variables))

                    # Update the frozen target models.
                    weights_biases = []
                    targets = a.get_weights()
                    for i, wb in enumerate(ta.get_weights()):
                        weights_biases.append(self.TD3_target_update_rate*targets[i] + (1-self.TD3_target_update_rate)*wb)
                    ta.set_weights(weights_biases)

                    weights_biases = []
                    targets = c1.get_weights()
                    for i, wb in enumerate(tc1.get_weights()):
                        weights_biases.append(self.TD3_target_update_rate*targets[i] + (1-self.TD3_target_update_rate)*wb)
                    tc1.set_weights(weights_biases)

                    weights_biases = []
                    targets = c2.get_weights()
                    for i, wb in enumerate(tc2.get_weights()):
                        weights_biases.append(self.TD3_target_update_rate*targets[i] + (1-self.TD3_target_update_rate)*wb)
                    tc2.set_weights(weights_biases)

    def evolve(self, nsteps):
        # genome_centroid = self.archive.importance_sample(1)[0].genome
        # genome_centroid = self.gradient_actors[0].flatten_wb()
        genome_centroid = self.archive.find_best().genome
        self.es = cma.CMAEvolutionStrategy(
            np.zeros(1+self.bd_dims),
            self.cmaes_sigma1,
            {'popsize': self.num_cores-1}
        )

        counter = 0
        self.restarts = 0
        while counter < nsteps:
            print(f'Evolve Iteration: {counter}')
            # (nbds+1) * genome.size
            jacobian = self.TD3_gradient(genome_centroid)
            jacobian /= (np.linalg.norm(jacobian, axis=1) + 1e-8)[:, None]

            # λ * (nbds+1)
            coeffs = np.array(self.es.ask())
            coeffs[:,0] = abs(coeffs[:,0])
            # λ * genome.size
            weighted_grads = coeffs @ jacobian

            cmaes_genomes = np.repeat([genome_centroid], self.num_cores-1, axis=0)
            cmaes_genomes += weighted_grads
            to_eval = [NNIndividual(self.Ashape, genome_centroid)] + \
                        [NNIndividual(self.Ashape, g) for g in cmaes_genomes]

            with get_context('spawn').Pool() as pool:
                bds, fitness, transitions = \
                    zip(*pool.starmap(NNIndividual.eval, [*zip(to_eval, self.envs, [self.random_seed]*self.num_cores)]))

            print(f'Centroid Fitness: {fitness[0]}')
            print(f'CMAES Fitness: {fitness[1:]}')

            to_eval[0].bds, to_eval[0].fitness = bds[0], fitness[0]
            self.archive.add(to_eval[0])

            for episode in transitions:
                for state, action, reward, next_state, done, bd in episode:
                    self.buffer.store_transition(state, action, reward, next_state, done, bd)

            improvement_scores = []
            for idv, b, f in zip(to_eval[1:], bds[1:], fitness[1:]):
                idv.bds, idv.fitness = b, f
                score = self.archive.add(idv, to_eval[0])
                # pycma minimizes objective by default
                improvement_scores.append(-score)

            print(f'Scores: {improvement_scores}')
            print(f'Average Score: {np.mean(improvement_scores)}')

            self.es.tell(tuple([*coeffs]), improvement_scores)
            counter += self.num_cores

            ranking_indices = np.argsort(improvement_scores)
            sorted_genomes = cmaes_genomes[ranking_indices]
            weights = (np.log(self.num_cores - 1 + 0.5) -
                       np.log(np.arange(1, self.num_cores)))
            total_weights = np.sum(weights)
            weights = weights / total_weights
            genome_centroid_cmaes = np.sum(sorted_genomes * np.expand_dims(weights, axis=1), axis=0)
            step = self.centroid_learning_rate * (genome_centroid_cmaes - genome_centroid)
            genome_centroid += step

            self.train_TD3()

            # if cmaes instance reaches terminating condition, restart from
            #   new cell
            if np.all(improvement_scores == 0) or self.es.stop():
                # genome_centroid = self.archive.importance_sample(1)[0].genome
                # genome_centroid = self.gradient_actors[0].flatten_wb()
                genome_centroid = self.archive.find_best().genome
                self.es = cma.CMAEvolutionStrategy(
                    np.zeros(1+self.bd_dims),
                    self.cmaes_sigma1,
                    {'popsize': self.num_cores-1}
                )
                self.restarts += 1

        self.savename = f"TD3_{self.env_id}_{nsteps}"

    # def render(self, idv, env):
    #     env.reset()
    #     env.seed(self.random_seed)
    #     NNIndividual.eval(idv, env, self.random_seed)

    def render_best(self):
        best_idv = self.archive.find_best()
        env = gym.make(self.env_id, render=True)
        env.reset()
        env.seed(self.random_seed)
        NNIndividual.eval(best_idv, env, self.random_seed)
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

        # Each actor should produce behaviors increasing the fitness/BD it is set to optimize
        to_eval = [NNIndividual(self.Ashape, a.flatten_wb()) for a in self.gradient_actors]
        with get_context('spawn').Pool() as pool:
            bds, fitness, _ = \
                zip(*pool.starmap(NNIndividual.eval, [*zip(to_eval, self.envs[:1+self.bd_dims], [self.random_seed]*(1+self.bd_dims))]))

        for i in range(1+self.bd_dims):
            if i == 0:
                print(f'Actor0 Fitness {fitness[0]}')
            else:
                print(f'Actor{i} BD{i-1} {bds[i][i-1]}')

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['envs'], \
            state['gradient_actors'], state['target_gradient_actors'], \
            state['gradient_critics'], state['target_gradient_critics'], \
            state['gradient_critics_alt'], state['target_gradient_critics_alt']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.envs = [gym.make(self.env_id) for _ in range(self.num_cores)]
        [env.seed(self.random_seed) for env in self.envs]

        dir = os.path.join(self.savename, 'tf_models')

        self.gradient_actors = [PolicyNetwork(self.Ashape) for _ in range(1+self.bd_dims)]
        [p.compile(optimizer=Adam(learning_rate=self.Adam_learning_rate)) for p in self.gradient_actors]
        self.target_gradient_actors = [PolicyNetwork(self.Ashape) for _ in range(1+self.bd_dims)]
        for i, (a, ta) in enumerate(zip(self.gradient_actors, self.target_gradient_actors)):
            a.load_weights(os.path.join(dir, f'a_{i}'))
            ta.load_weights(os.path.join(dir, f'ta_{i}'))

        # DDQN: take the min of the critics to avoid overestimation
        self.gradient_critics = [DQCriticNetwork(self.Cshape, self.action_dims) for _ in range(1+self.bd_dims)]
        [p.compile(optimizer=Adam(learning_rate=self.Adam_learning_rate)) for p in self.gradient_critics]
        self.target_gradient_critics = [DQCriticNetwork(self.Cshape, self.action_dims) for _ in range(1+self.bd_dims)]
        for i, (c1, tc1) in enumerate(zip(self.gradient_critics, self.target_gradient_critics)):
            c1.load_weights(os.path.join(dir, f'c1_{i}'))
            tc1.load_weights(os.path.join(dir, f'tc1_{i}'))

        self.gradient_critics_alt = [DQCriticNetwork(self.Cshape, self.action_dims) for _ in range(1+self.bd_dims)]
        [p.compile(optimizer=Adam(learning_rate=self.Adam_learning_rate)) for p in self.gradient_critics_alt]
        self.target_gradient_critics_alt = [DQCriticNetwork(self.Cshape, self.action_dims) for _ in range(1+self.bd_dims)]
        for i, (c2, tc2) in enumerate(zip(self.gradient_critics_alt, self.target_gradient_critics_alt)):
            c2.load_weights(os.path.join(dir, f'c2_{i}'))
            tc2.load_weights(os.path.join(dir, f'tc2_{i}'))

    @classmethod
    def from_pickle(cls, env_id, nsteps):
        savename = f"TD3_{env_id}_{nsteps}"
        filename = os.path.join(savename, f'{savename}.p')
        try:
            return pickle.load(open(filename, 'rb'))
        except FileNotFoundError:
            print('Cannot find savefile, exiting...')
            quit()
        except Exception as err:
            print(err)
            quit()

    def save(self):
        dir = os.path.join(self.savename, 'tf_models')
        if not os.path.exists(dir):
            os.makedirs(dir, exist_ok=True)

        for i, (a, ta, c1, tc1, c2, tc2) in enumerate(zip(
            self.gradient_actors, self.target_gradient_actors,
            self.gradient_critics, self.target_gradient_critics,
            self.gradient_critics_alt, self.target_gradient_critics_alt
        )):
            a.save_weights(os.path.join(dir, f'a_{i}'), overwrite=True)
            ta.save_weights(os.path.join(dir, f'ta_{i}'), overwrite=True)
            c1.save_weights(os.path.join(dir, f'c1_{i}'), overwrite=True)
            tc1.save_weights(os.path.join(dir, f'tc1_{i}'), overwrite=True)
            c2.save_weights(os.path.join(dir, f'c2_{i}'), overwrite=True)
            tc2.save_weights(os.path.join(dir, f'tc2_{i}'), overwrite=True)

        pickle.dump(self, open(os.path.join(self.savename, f'{self.savename}.p'), 'wb'))

if __name__ == '__main__':
    '''use this if train from scratch
    '''
    # archive = Archive([(0, 1, 0.05)] * 4)
    # num_cores = 50
    # agent = Agent(archive, num_cores)
    # agent.warmup(50)
    # agent.evolve(1000)
    # agent.render_best()
    # agent.save()

    '''use this if resume from previously trained archive
    '''
    agent = Agent.from_pickle('QDAntBulletEnv-v0', 500)
    # agent.evolve(1000)
    agent.render_best()
    # agent.save()
