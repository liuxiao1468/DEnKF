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
import tensorflow_probability as tfp
import csv
import cv2
import re

class DataLoader:
    def __init__(self):
        self.dataset_path = '/Users/xiao.lu/project/KITTI_dataset/' 

    def preprocessing(self, data):
        img_2 = cv2.imread(self.dataset_path+data[3][1])
        img_1 = cv2.imread(self.dataset_path+data[3][0])
        img_2 = cv2.resize(img_2, (150, 50), interpolation=cv2.INTER_LINEAR)
        img_1 = cv2.resize(img_1, (150, 50), interpolation=cv2.INTER_LINEAR)
        img_2_ = img_2.astype(np.float32)/255.
        img_1_ = img_1.astype(np.float32)/255.
        ###########
        diff = img_2_ - img_1_
        diff = diff*0.5 + 0.5
        # diff = (diff * 255).astype(np.uint8)
        ###########
        img = np.concatenate((img_2_, diff), axis=-1)
        return img

    def load_training_data(self, batch_size):
        dim_x = 5
        dim_z = 2
        dataset = pickle.load(open('KITTI_VO_dataset.pkl', 'rb'))
        select = random.sample(range(0, len(dataset)), batch_size)
        states_gt_save = []
        states_pre_save = []
        observation_save = []
        observation_img = []

        for idx in select:
            states_gt_save.append(dataset[idx][1])
            states_pre_save.append(dataset[idx][0])
            observation_save.append(dataset[idx][2])
            img = self.preprocessing(dataset[idx])
            observation_img.append(img)
        states_pre_save = np.array(states_pre_save)
        states_gt_save = np.array(states_gt_save)
        observation_save = np.array(observation_save)
        observation_img = np.array(observation_img)

        # to tensor
        states_pre_save = tf.convert_to_tensor(states_pre_save, dtype=tf.float32)
        states_pre_save = tf.reshape(states_pre_save, [batch_size, 1, dim_x])

        states_gt_save = tf.convert_to_tensor(states_gt_save, dtype=tf.float32)
        states_gt_save = tf.reshape(states_gt_save, [batch_size, 1, dim_x])

        observation_save = tf.convert_to_tensor(observation_save, dtype=tf.float32)
        observation_save = tf.reshape(observation_save, [batch_size, 1, dim_z])

        observation_img = tf.convert_to_tensor(observation_img, dtype=tf.float32)

        return states_pre_save, states_gt_save, observation_save, observation_img


    def load_testing_data(self):
        dim_x = 5
        dim_z = 2
        dataset = pickle.load(open('KITTI_VO_test.pkl', 'rb'))
        N = len(dataset)
        states_gt_save = []
        states_pre_save = []
        observation_save = []
        observation_img = []

        for idx in range(N):
            states_gt_save.append(dataset[idx][1])
            states_pre_save.append(dataset[idx][0])
            observation_save.append(dataset[idx][2])
            img = self.preprocessing(dataset[idx])
            observation_img.append(img)
        states_pre_save = np.array(states_pre_save)
        states_gt_save = np.array(states_gt_save)
        observation_save = np.array(observation_save)
        observation_img = np.array(observation_img)

        # to tensor
        states_pre_save = tf.convert_to_tensor(states_pre_save, dtype=tf.float32)
        states_pre_save = tf.reshape(states_pre_save, [N, 1, 1, dim_x])

        states_gt_save = tf.convert_to_tensor(states_gt_save, dtype=tf.float32)
        states_gt_save = tf.reshape(states_gt_save, [N, 1, 1, dim_x])

        observation_save = tf.convert_to_tensor(observation_save, dtype=tf.float32)
        observation_save = tf.reshape(observation_save, [N, 1, 1, dim_z])

        observation_img = tf.convert_to_tensor(observation_img, dtype=tf.float32)
        observation_img = tf.expand_dims(observation_img, axis=1)

        return states_pre_save, states_gt_save, observation_save, observation_img


    def format_state(self, state, batch_size, num_ensemble, dim_x):
        dim_x = dim_x
        # diag = np.ones((dim_x)).astype(np.float32) * 0.1
        diag = np.array([0.1, 0.1, 0.1, 0.5, 0.01]).astype(np.float32)
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
        # diag = np.ones((dim_x)).astype(np.float32) * 0.1
        diag = np.array([0.1, 0.1, 0.1, 0.5, 0.01]).astype(np.float32)
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


# DataLoader_func = DataLoader()
# states_pre_save, states_gt_save, observation_save, observation_img = DataLoader_func.load_training_data(
#     8)
# print(states_pre_save.shape)
# print('---')
# print(states_gt_save[0])
# print('---')
# print(observation_save.shape)
# print('---')
# print(observation_img.shape)
# print('=============')

# state = DataLoader_func.format_state(states_gt_save, 8, 32, 5)
# print(state[0][1][0])

# states_pre_save, states_gt_save, observation_save, observation_img = DataLoader_func.load_testing_data()
# print(states_pre_save.shape)
# print('---')
# print(states_gt_save.shape)
# print('---')
# print(observation_save.shape)
# print('---')
# print(observation_img.shape)
# print('=============')













