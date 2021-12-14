import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as Rot
import random
import tensorflow as tf
import time
import pickle
import pdb
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import tensorflow_probability as tfp

config = ConfigProto()
config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)

'''
data loader for training the toy example
'''
def data_loader_function(data_path):
    name = ['constant', 'exp']
    num_sensors = 100

    observations = []
    states_true = []
    perturb_state = []
    scale = []
    for i in range (num_sensors):
        scale.append(random.uniform(0.8, 1.2))

    s = 1/25.
    with open(data_path, 'rb') as f:
        traj = pickle.load(f)
    for i in range (len(traj['xTrue'])):
        observation = []
        state = []
        state_prime = []
        for j in range (num_sensors):
            observe = [traj['sensors'][i][0][j]*s, traj['sensors'][i][1][j]*s]
            observation.append(observe)
            angles = traj['xTrue'][i][2]
            xTrue = [traj['xTrue'][i][0]*s, traj['xTrue'][i][1]*s, np.cos(angles), np.sin(angles), traj['xTrue'][i][3]]
            xTrue_p = [traj['xTrue'][i][0]*s*scale[j], traj['xTrue'][i][1]*s*scale[j], np.cos(angles), np.sin(angles), traj['xTrue'][i][3]]
            state.append(xTrue)
            state_prime.append(xTrue_p)
        observations.append(observation)
        states_true.append(state)
        perturb_state.append(state_prime)
    observations = np.array(observations)
    observations = tf.reshape(observations, [len(traj['xTrue']), num_sensors, 1, 2])
    states_true = np.array(states_true)
    states_true = tf.reshape(states_true, [len(traj['xTrue']), num_sensors, 1, 5])
    perturb_state = np.array(perturb_state)
    perturb_state = tf.reshape(perturb_state, [len(traj['xTrue']), num_sensors, 1, 5])

    return observations, states_true, perturb_state

'''
load data for training
'''
global name
name = ['constant', 'exp']
global index
index = 1
observations, states_true, perturb_state = data_loader_function('./dataset/100_demos_'+name[index]+'.pkl')
print(perturb_state.shape)
print(states_true.shape)
print(perturb_state[10][0])

