import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import math
import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf
import time
import pickle
import pdb
import tensorflow_probability as tfp
import csv
import cv2

class transform:
    def __init__(self):
        super(transform, self).__init__()
        self.std_ = np.array([82.45586931, 89.61514707, 87.01358807, 23.3564316,  21.99383088, 18.40371951,
            18.69889549, 24.73797033, 17.80170602, 33.7088314 ])
        self.m_ = np.array([ -1.0084381,   -0.94979133,  -1.67127022, -11.48051483, -11.67116862,
            2.96867813,  -0.05636066, -13.11423594,   1.21702482, -11.46219248])
        self.state_std = self.std_[7:]
        self.state_m = self.m_[7:]
        self.m = self.m_[:7]
        self.std = self.std_[:7]

    def state_transform(self, state):
        '''
        state -> [batch_size, num_ensemble, dim_x]
        '''
        batch_size = state.shape[0]
        num_ensemble = state.shape[1]
        dim_x = state.shape[2]
        state = tf.reshape(state, [batch_size * num_ensemble, dim_x])
        state = (state - self.state_m)/self.state_std
        state = tf.reshape(state, [batch_size, num_ensemble, dim_x])
        return state

    def state_inv_transform(self, state):
        '''
        state -> [batch_size, num_ensemble, dim_x]
        '''
        batch_size = state.shape[0]
        num_ensemble = state.shape[1]
        dim_x = state.shape[2]
        state = tf.reshape(state, [batch_size * num_ensemble, dim_x])
        state = (state * self.state_std) + self.state_m 
        state = tf.reshape(state, [batch_size, num_ensemble, dim_x])
        return state

    def obs_transform(self, state):
        '''
        obs -> [batch_size, num_ensemble, dim_z]
        '''
        batch_size = state.shape[0]
        num_ensemble = state.shape[1]
        dim_z = state.shape[2]
        state = tf.reshape(state, [batch_size * num_ensemble, dim_z])
        state = (state - self.m)/self.std
        state = tf.reshape(state, [batch_size, num_ensemble, dim_z])
        return state

    def obs_inv_transform(self, state):
        '''
        obs -> [batch_size, num_ensemble, dim_z]
        '''
        batch_size = state.shape[0]
        num_ensemble = state.shape[1]
        dim_z = state.shape[2]
        state = tf.reshape(state, [batch_size * num_ensemble, dim_z])
        state = (state * self.std) + self.m 
        state = tf.reshape(state, [batch_size, num_ensemble, dim_z])
        return state

class DataLoader:
    def __init__(self):
        self.transform_ = transform()

    def train_data(self, path, batch_size):
        dataset = pickle.load(open(path, 'rb'))
        N = len(dataset)
        select = random.sample(range(0, N), batch_size)
        states_gt_save = []
        states_pre_save = []
        observation_save = []
        for idx in select:
            states_pre_save.append(dataset[idx][0][7:])
            states_gt_save.append(dataset[idx][1][7:])
            observation_save.append(dataset[idx][2])
        states_pre_save = np.array(states_pre_save)
        states_gt_save = np.array(states_gt_save)
        observation_save = np.array(observation_save)

        # to tensor
        states_pre_save = tf.convert_to_tensor(states_pre_save, dtype=tf.float32)
        states_pre_save = tf.reshape(states_pre_save, [batch_size, 1, 3])
        states_gt_save = tf.convert_to_tensor(states_gt_save, dtype=tf.float32)
        states_gt_save = tf.reshape(states_gt_save, [batch_size, 1, 3])
        observation_save = tf.convert_to_tensor(observation_save, dtype=tf.float32)
        observation_save = tf.reshape(observation_save, [batch_size, 1, 7])

        states_pre_save = self.transform_.state_transform(states_pre_save)
        states_gt_save = self.transform_.state_transform(states_gt_save)
        observation_save = self.transform_.obs_transform(observation_save)
        return states_pre_save, states_gt_save, observation_save

    def test_data(self, path):
        dataset = pickle.load(open(path, 'rb'))
        N = len(dataset)
        states_gt_save = []
        states_pre_save = []
        observation_save = []
        for idx in range (N):
            states_pre_save.append(dataset[idx][0][7:])
            states_gt_save.append(dataset[idx][1][7:])
            observation_save.append(dataset[idx][2])
        states_pre_save = np.array(states_pre_save)
        states_gt_save = np.array(states_gt_save)
        observation_save = np.array(observation_save)

        # to tensor
        states_pre_save = tf.convert_to_tensor(states_pre_save, dtype=tf.float32)
        states_gt_save = tf.convert_to_tensor(states_gt_save, dtype=tf.float32)
        observation_save = tf.convert_to_tensor(observation_save, dtype=tf.float32)
        states_pre_save = tf.reshape(states_pre_save, [N, 1, 3])
        states_gt_save = tf.reshape(states_gt_save, [N, 1, 3])
        observation_save = tf.reshape(observation_save, [N, 1, 7])
        states_pre_save = self.transform_.state_transform(states_pre_save)
        states_gt_save = self.transform_.state_transform(states_gt_save)
        observation_save = self.transform_.obs_transform(observation_save)
        states_pre_save = tf.reshape(states_pre_save, [N, 1, 1, 3])
        states_gt_save = tf.reshape(states_gt_save, [N, 1, 1, 3])
        observation_save = tf.reshape(observation_save, [N, 1, 1, 7])

        return states_pre_save, states_gt_save, observation_save


    def format_state(self, state, batch_size, num_ensemble, dim_x):
        dim_x = dim_x
        diag = np.ones((dim_x)).astype(np.float32) * 0.01
        diag = diag.astype(np.float32)
        mean = np.zeros((dim_x))
        mean = tf.convert_to_tensor(mean, dtype=tf.float32)
        mean = tf.stack([mean] * batch_size)
        nd = tfp.distributions.MultivariateNormalDiag(loc=mean, scale_diag=diag)
        Q = tf.reshape(nd.sample(num_ensemble), [batch_size, num_ensemble, dim_x])
        for n in range (batch_size):
            if n == 0:
                ensemble = tf.reshape(tf.stack([state[n]] * num_ensemble), [1, num_ensemble, dim_x])
            else:
                tmp = tf.reshape(tf.stack([state[n]] * num_ensemble), [1, num_ensemble, dim_x])
                ensemble = tf.concat([ensemble, tmp], 0)
        ensemble = ensemble + Q
        state_input = (ensemble, state)
        return state_input

    def format_init_state(self, state, batch_size, num_ensemble, dim_x):
        dim_x = dim_x
        diag = np.ones((dim_x)).astype(np.float32) * 0.01
        diag = diag.astype(np.float32)
        mean = np.zeros((dim_x))
        mean = tf.convert_to_tensor(mean, dtype=tf.float32)
        mean = tf.stack([mean] * batch_size)
        nd = tfp.distributions.MultivariateNormalDiag(loc=mean, scale_diag=diag)
        Q = tf.reshape(nd.sample(num_ensemble), [batch_size, num_ensemble, dim_x])
        for n in range (batch_size):
            if n == 0:
                ensemble = tf.reshape(tf.stack([state[n]] * num_ensemble), [1, num_ensemble, dim_x])
            else:
                tmp = tf.reshape(tf.stack([state[n]] * num_ensemble), [1, num_ensemble, dim_x])
                ensemble = tf.concat([ensemble, tmp], 0)
        ensemble = ensemble + Q
        state_input = (ensemble, state)
        return state_input


# DataLoader = DataLoader()

# states_pre_save, states_gt_save, observation_save = DataLoader.train_data('./dataset/train_set.pkl', 8)
# print(states_pre_save.shape)
# print(states_gt_save)
# print(observation_save.shape)

# states_pre_save, states_gt_save, observation_save = DataLoader.test_data('./dataset/test_set_1.pkl')
# print(states_pre_save.shape)
# print(states_gt_save)
# print(observation_save.shape)


########################### calculate parameters  ######################
# parameters = {}

# parameters['state_m'] = np.mean(states_gt_save, axis=0)
# parameters['state_std'] = np.std(states_gt_save, axis=0)

# print(parameters['state_m'])
# print(parameters['state_std'])

# with open('parameters.pkl', 'wb') as f:
#     pickle.dump(parameters, f)
#########################################################################






