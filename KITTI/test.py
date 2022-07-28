import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Times New Roman"
plt.rcParams['savefig.dpi'] = 500
import numpy as np
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf 
import pickle
import math
import cv2
from scipy.spatial.transform import Rotation as Rot
from numpy.random import random


arr = np.linspace(1, 20, 20)
arr = np.reshape(arr, (5, 4))
# aa = arr[:,2]
# print(aa)


# ct = np.cos(-(aa-np.pi/2))
# st = np.sin(-(aa-np.pi/2))
# print(ct)

# print('-----')
arr = tf.convert_to_tensor(arr, dtype=tf.float32)
# arr = tf.reshape(arr, [5, 1, 4])
# num_ensemble= 4
# for i in range (5):
#     if i == 0:
#         ensemble = tf.reshape(tf.stack([arr[i]] * num_ensemble), [1, num_ensemble, 4])
#     else:
#         tmp = tf.reshape(tf.stack([arr[i]] * num_ensemble), [1, num_ensemble, 4])
#         ensemble = tf.concat([ensemble, tmp], 0)
# print(ensemble.shape)




# a = tf.reshape(ensemble[:,:,2], [5,10,1])
# # b = tf.reshape(ensemble[:,:,1:4], [5,10,3])
# # print(a)

# ct = tf.cos(-(a-np.pi/2))
# st = tf.sin(-(a-np.pi/2))
# print(ct)

# # c = tf.concat([a, b, a, a], -1)
# # print(c.shape)


# diag = np.array([0.1, 0.1, 0.1, 0.5, 0.01]).astype(np.float32)
# diag = diag.astype(np.float32)
# print(diag)
print('-----')
bs = 5
num_particles = 30
dim_x = 4

arr = np.linspace(1, bs*num_particles, bs*num_particles)
arr = np.reshape(arr, (bs, num_particles))
arr = tf.convert_to_tensor(arr, dtype=tf.float32)

particles = np.linspace(1, bs*dim_x*num_particles, bs*dim_x*num_particles)
particles = tf.convert_to_tensor(particles, dtype=tf.float32)
particles = tf.reshape(particles, [bs, num_particles, dim_x])

weights = arr

print(particles.shape)
print(weights.shape)
# print(weights)

def resample(particles, weights):
    a = tf.reduce_sum(weights, axis=1)
    a = tf.stack([a]*num_particles)
    a = tf.transpose(a, perm=[1,0])
    a = weights/a

    idx = []

    for i in range (bs):
        weight = a[i]
        N = len(weight)
        # make N subdivisions, and chose a random position within each one
        positions = (random(N) + range(N)) / N
        indexes = np.zeros(N, 'i')
        cumulative_sum = np.cumsum(weight)
        i, j = 0, 0
        while i < N:
            if positions[i] < cumulative_sum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1
        idx.append(indexes)
    idx = np.array(idx)

    for i in range (bs):
        if i == 0:
            new_particle = tf.reshape(tf.gather(particles[i], idx[i]), [1, num_particles, dim_x] )
        else:
            tmp = tf.reshape(tf.gather(particles[i], idx[i]), [1, num_particles, dim_x])
            new_particle = tf.concat([new_particle, tmp], 0)

    weights = tf.expand_dims(a,1)
    print(weights.shape)
    tmp = tf.matmul(weights, new_particle)
    print(tmp)
    
    return new_particle, weights, tmp


new_particle, weights, tmp = resample(particles, weights)
# # print(a)
# print(particles[0])
# b = tf.gather(particles[0], a[0])
# print('-----')
# print(b)
# print(weights[0])
# w = weights[0]
# w = tf.expand_dims(w,0)
# tmp = tf.matmul(w, b)
# print(tmp)
