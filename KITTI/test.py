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


# arr = np.linspace(1, 20, 20)
# arr = np.reshape(arr, (5, 4))
# aa = arr[:,2]
# print(aa)


# ct = np.cos(-(aa-np.pi/2))
# st = np.sin(-(aa-np.pi/2))
# print(ct)

# print('-----')
# arr = tf.convert_to_tensor(arr, dtype=tf.float32)
# arr = tf.reshape(arr, [5, 1, 4])
# num_ensemble= 10
# for i in range (5):
#     if i == 0:
#         ensemble = tf.reshape(tf.stack([arr[i]] * num_ensemble), [1, num_ensemble, 4])
#     else:
#         tmp = tf.reshape(tf.stack([arr[i]] * num_ensemble), [1, num_ensemble, 4])
#         ensemble = tf.concat([ensemble, tmp], 0)

# print(ensemble.shape)
# # print(arr)

# a = tf.reshape(ensemble[:,:,2], [5,10,1])
# # b = tf.reshape(ensemble[:,:,1:4], [5,10,3])
# # print(a)

# ct = tf.cos(-(a-np.pi/2))
# st = tf.sin(-(a-np.pi/2))
# print(ct)

# # c = tf.concat([a, b, a, a], -1)
# # print(c.shape)

print('-----')
diag = np.array([0.1, 0.1, 0.1, 0.5, 0.01]).astype(np.float32)
diag = diag.astype(np.float32)
print(diag)