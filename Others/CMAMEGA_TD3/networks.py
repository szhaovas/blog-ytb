import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense
from itertools import chain

class PolicyNetwork(keras.Model):
    def __init__(self, shape, random_init=True, wb=None):
        '''
        shape: (input_dims, layer1_dims, layer2_dims, output_dims)
        wb:    1d array containing weights and biases of all layers
        '''
        super(PolicyNetwork, self).__init__()
        self.hidden1 = Dense(shape[1], activation='tanh', kernel_initializer='glorot_normal')
        self.hidden2 = Dense(shape[2], activation='tanh', kernel_initializer='glorot_normal')
        self.out = Dense(shape[3], activation='tanh')
        # FIXME: There should be a better way to build and initialize
        self.call(np.zeros((1, shape[0])))

        if not random_init:
            self.__set_wb__(wb)

    def call(self, state):
        hidden1_value = self.hidden1(state)
        hidden2_value = self.hidden2(hidden1_value)
        return self.out(hidden2_value)

    def flatten_wb(self):
        return np.array([*chain.from_iterable([i.numpy().flatten() for i in self.trainable_variables])])

    def __set_wb__(self, wb):
        layer_start_idx = 0
        for l in self.layers:
            w, b = l.get_weights()
            new_w = wb[layer_start_idx:layer_start_idx+w.size].reshape(w.shape)
            new_b = wb[layer_start_idx+w.size:layer_start_idx+w.size+b.size].reshape(b.shape)
            l.set_weights([new_w, new_b])
            layer_start_idx += w.size+b.size

'''
Learns to predict V(s), used in PPO
'''
class VCriticNetwork(keras.Model):
    def __init__(self, shape):
        super(VCriticNetwork, self).__init__()
        self.hidden1 = Dense(shape[1], activation='relu')
        self.hidden2 = Dense(shape[2], activation='relu')
        self.out = Dense(1, activation=None)
        # FIXME: There should be a better way to build and initialize
        self.call(np.zeros((1, shape[0])))

    def call(self, state):
        hidden1_value = self.hidden1(state, axis=1)
        hidden2_value = self.hidden2(hidden1_value)
        return self.out(hidden2_value)

'''
Learns to predict Q(s,a), used in TD3
'''
class DQCriticNetwork(keras.Model):
    def __init__(self, shape, action_dims):
        super(DQCriticNetwork, self).__init__()
        self.hidden1 = Dense(shape[1], activation='relu')
        self.hidden2 = Dense(shape[2], activation='relu')
        self.out = Dense(1, activation=None)
        # FIXME: There should be a better way to build and initialize
        self.call(np.zeros((1, shape[0])), np.zeros((1, action_dims)))

    def call(self, state, action):
        hidden1_value = self.hidden1(tf.concat([state, action], axis=1))
        hidden2_value = self.hidden2(hidden1_value)
        return self.out(hidden2_value)
