import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as Rot
import random
import tensorflow as tf
import time
import pickle as pkl
import pdb
import tensorflow_probability as tfp
import csv
import cv2
import re

class DataLoader:
    def __init__(self):
        self.camera_list = ['checkstand', 'checkstandleft', 'checkstandright', 'coffeebar', 
        'coolerleft', 'coolerright', 'entranceleft', 'entranceright', 'sideleft', 'sideright',
        'smoothiebar']
        (self.width, self.height) = (1920,1080) # the camera frame
        (self.w, self.h) = (2550, 1650) # the floor plan
        # load the parameter
        # parameter = pkl.load(open('./dataset/parameter.pkl', 'rb'))

        # # pre state
        # self.m1 = parameter['m1']
        # self.std1 = parameter['std1']

        # # gt state
        # self.m2 = parameter['m2']
        # self.std2 = parameter['std2']

        # # bbx observation
        # self.m3 = parameter['m3']
        # self.std3 = parameter['std3']

        # # pose observation
        # self.m4 = parameter['m4']
        # self.std4 = parameter['std4']

        # pre state
        self.m1 = 0
        self.std1 = 1

        # gt state
        self.m2 = 0
        self.std2 = 1

        # bbx observation
        self.m3 = 0
        self.std3 = 1

        # pose observation
        self.m4 = 0
        self.std4 = 1

        # # dataset path
        # self.path_1 = './dataset/track_dataset_01.pkl'
        # self.path_2 = './dataset/track_dataset_02.pkl'
        # self.path_3 = './dataset/track_dataset_03.pkl'

    def str_2_array(self,x):
        x = str(x)
        x = x[1:-1]
        x = x.strip()
        x = re.sub('\s+',',',x)
        x1 = float(x.split(',')[0])
        y1 = float(x.split(',')[1])
        point = np.array([x1, y1])
        return point

    def load_training_data(self, path_1, path_2, path_3, batch_size, cam_name):
        dim_x = 2
        dim_z1 = 4
        dim_z2 = 51
        kf_data = pkl.load(open(path_1, 'rb'))
        tmp = pkl.load(open(path_2, 'rb'))
        kf_data = kf_data + tmp
        tmp = pkl.load(open(path_3, 'rb'))
        kf_data = kf_data + tmp
        N = len(kf_data)

        train_data =  kf_data
        kf_data = []
        for i in range (N):
            if train_data[i][0] == cam_name:
                kf_data.append(train_data[i])
        N = len(kf_data)
        # print('new dataset length: ',N)
        select = random.sample(range(0, N), batch_size)
        states_gt_save = []
        states_pre_save = []
        observation_save_1 = []
        observation_save_2 = []
        for idx in select:
            fp = self.str_2_array(kf_data[idx][4])
            fp[0] = fp[0]/self.w
            fp[1] = fp[1]/self.h
            states_gt_save.append((fp-self.m2)/self.std2)
            fp = self.str_2_array(kf_data[idx][3])
            fp[0] = fp[0]/self.w
            fp[1] = fp[1]/self.h
            states_pre_save.append((fp-self.m1)/self.std1)
            bbx = kf_data[idx][5]
            bbx[0] = bbx[0]/self.width
            bbx[1] = bbx[1]/self.height
            bbx[2] = bbx[2]/self.width
            bbx[3] = bbx[3]/self.height
            observation_save_1.append((bbx-self.m3)/self.std3)
            pose = kf_data[idx][6]
            pose[:,0] = pose[:,0]/self.width
            pose[:,1] = pose[:,1]/self.height
            pose_conf = kf_data[idx][7]
            pose = np.concatenate((pose, pose_conf), axis=1)
            pose = pose.flatten('F')
            observation_save_2.append((pose-self.m4)/self.std4)
        states_pre_save = np.array(states_pre_save)
        states_gt_save = np.array(states_gt_save)
        observation_save_1 = np.array(observation_save_1)
        observation_save_2 = np.array(observation_save_2)
        # to tensor
        states_pre_save = tf.convert_to_tensor(states_pre_save, dtype=tf.float32)
        states_pre_save = tf.reshape(states_pre_save, [batch_size, 1, dim_x])

        states_gt_save = tf.convert_to_tensor(states_gt_save, dtype=tf.float32)
        states_gt_save = tf.reshape(states_gt_save, [batch_size, 1, dim_x])

        observation_save_1 = tf.convert_to_tensor(observation_save_1, dtype=tf.float32)
        observation_save_1 = tf.reshape(observation_save_1, [batch_size, 1, dim_z1])

        observation_save_2 = tf.convert_to_tensor(observation_save_2, dtype=tf.float32)
        observation_save_2 = tf.reshape(observation_save_2, [batch_size, 1, dim_z2])

        return states_pre_save, states_gt_save, observation_save_1, observation_save_2


    def load_testing_data(self, path_2, tracker_id):
        dim_x = 2
        dim_z1 = 4
        dim_z2 = 51
        test_data = []
        kf_data = pkl.load(open(path_2, 'rb'))
        states_gt_save = []
        states_pre_save = []
        observation_save_1 = []
        observation_save_2 = []
        for i in range (len(kf_data)):
            if kf_data[i][1] == tracker_id:
                test_data.append(kf_data[i])
        for idx in range (len(test_data)):
            fp = self.str_2_array(test_data[idx][4])
            fp[0] = fp[0]/self.w
            fp[1] = fp[1]/self.h
            states_gt_save.append((fp-self.m2)/self.std2)
            fp = self.str_2_array(test_data[idx][3])
            fp[0] = fp[0]/self.w
            fp[1] = fp[1]/self.h
            states_pre_save.append((fp-self.m1)/self.std1)
            bbx = test_data[idx][5]
            bbx[0] = bbx[0]/self.width
            bbx[1] = bbx[1]/self.height
            bbx[2] = bbx[2]/self.width
            bbx[3] = bbx[3]/self.height
            observation_save_1.append((bbx-self.m3)/self.std3)
            pose = test_data[idx][6]
            pose[:,0] = pose[:,0]/self.width
            pose[:,1] = pose[:,1]/self.height
            pose_conf = test_data[idx][7]
            pose = np.concatenate((pose, pose_conf), axis=1)
            pose = pose.flatten('F')
            observation_save_2.append((pose-self.m4)/self.std4)
        states_pre_save = np.array(states_pre_save)
        states_gt_save = np.array(states_gt_save)
        observation_save_1 = np.array(observation_save_1)
        observation_save_2 = np.array(observation_save_2)

        N = len(test_data)

        states_pre_save = tf.convert_to_tensor(states_pre_save, dtype=tf.float32)
        states_pre_save = tf.reshape(states_pre_save, [N, 1, dim_x])
        states_pre_save = tf.expand_dims(states_pre_save, axis=1)

        states_gt_save = tf.convert_to_tensor(states_gt_save, dtype=tf.float32)
        states_gt_save = tf.reshape(states_gt_save, [N, 1, dim_x])
        states_gt_save = tf.expand_dims(states_gt_save, axis=1)

        observation_save_1 = tf.convert_to_tensor(observation_save_1, dtype=tf.float32)
        observation_save_1 = tf.reshape(observation_save_1, [N, 1, dim_z1])
        observation_save_1 = tf.expand_dims(observation_save_1, axis=1)


        observation_save_2 = tf.convert_to_tensor(observation_save_2, dtype=tf.float32)
        observation_save_2 = tf.reshape(observation_save_2, [N, 1, dim_z2])
        observation_save_2 = tf.expand_dims(observation_save_2, axis=1)

        return states_pre_save, states_gt_save, observation_save_1, observation_save_2


    def format_state(self, state, batch_size, num_ensemble, dim_x):
        dim_x = dim_x
        diag = np.ones((dim_x)).astype(np.float32) * 0.1
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
        diag = np.ones((dim_x)).astype(np.float32) * 0.1
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

# path = './dataset/track_dataset.pkl'


# # dataset path
# path_1 = './dataset/track_dataset_01.pkl'
# path_2 = './dataset/track_dataset_02.pkl'
# path_3 = './dataset/track_dataset_03.pkl'
# DataLoader_func = DataLoader()
# states_pre_save, states_gt_save, observation_save_1, observation_save_2 = DataLoader_func.load_training_data(
#     path_1, path_2, path_3, 8, "coolerleft")
# print(states_pre_save.shape)
# print('---')
# print(states_gt_save[0])
# print('---')
# print(observation_save_2.shape)

# print('=============')

# states_pre_save, states_gt_save, observation_save_1, observation_save_2 = DataLoader_func.load_testing_data(path_2, "coolerleft-4")
# print(states_pre_save.shape)
# print('---')
# print(states_gt_save.shape)
# print('---')
# # print(observation_save_2)













