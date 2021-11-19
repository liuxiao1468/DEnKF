import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import glob
import random
import csv
import shutil
import json
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import math
import matplotlib.pyplot as plt


condition_list = ['walk', 'walk_dual','OA', 'OA_dual']

subjects = ['MN02', 'MN03','MN04','MN05','MN06','MN07',
'MN08','MN09','MN10','MN11']

joints = ['LAnkle', 'LFemur', 'LFoot', 'LHip', 'LKnee', 'LTibia',
'RAnkle', 'RFemur', 'RFoot', 'RHip', 'RKnee', 'RTibia',
'Lumbar_bending', 'Lumbar_flexion', 'Lumbar_rotation', 
'Pelvis_list', 'Pelvis_rotation', 'Pelvis_tilt'] 

parts = ['kinematics', 'kinetics', 'power']

# Opening JSON file
f = open('/home/xiaoliu/bio_data/dataset.json',)
data = json.load(f)

# ##################### extract one data as example #################

# follow the convention data['condition']['subject']['joint']
# you can do whatever cross-reference as you need
data_01 = data['walk']['MN02']['LFoot']['kinematics']
data_02 = data['walk']['MN03']['LFoot']['kinematics']
data_03 = data['walk']['MN04']['LFoot']['kinematics']
data_04 = data['walk']['MN05']['LFoot']['kinematics']

# every data is saved as a 2D list
# print('col: ', len(data_01)) 
# print('row: ', len(data_01[0]))

# preprocess data into 2 cycles one sequence
def preprocess_sequence(data):
	seq_len = math.floor(len(data)/2)
	data_temp = []
	idx = 0
	for i in range (seq_len):
		seq = data[idx][:-1] + data[idx+1]
		data_temp.append(seq)
		idx = idx+2
	data = data_temp
	return data

# training_sequence = []
# for i in range (4):
# 	seq = preprocess_sequence(data['walk']['MN0'+str(i+2)]['LFemur']['kinematics'])
# 	training_sequence.append(seq)


##################### data presentation #################
# look at one subject in 4 different conditions
# look at the data in 3 spaces
condition_list = ['walk', 'walk_dual','OA', 'OA_dual']
parts = ['kinematics', 'kinetics', 'power']
select_joints = ['LAnkle','LFemur','LFoot','LKnee','Lumbar_flexion','Pelvis_tilt'] 
# select_joints = ['LAnkle'] 

joint_data = []

for i in range (len(select_joints)):
	fig = plt.figure(figsize=(3, 4))
	for k in range (len(parts)):
		for j in range (len(condition_list)):
			get_data = data[condition_list[j]]['MN05'][select_joints[i]][parts[k]]
			get_data = preprocess_sequence(get_data)

			plt.subplot(3, 4, (4*k+1)+j)
			x = list(range(1, len(get_data[0])+1))
			for m in range (len(get_data)):
				plt.plot(x, get_data[m])
			plt.grid()
			plt.xlabel('time')
			plt.ylabel(parts[k]+' - '+condition_list[j])
	fig.suptitle(select_joints[i]+' data for subject MN02')
	plt.show()




condition_list = ['walk']
parts = ['kinematics', 'kinetics', 'power']
select_joints = ['LAnkle','LKnee','LHip']
fig = plt.figure(figsize=(3, 3))

for i in range (len(select_joints)):
    for k in range (len(parts)):
        for j in range (len(condition_list)):
            get_data = data[condition_list[j]]['MN05'][select_joints[i]][parts[k]]
            get_data = preprocess_sequence(get_data)

            plt.subplot(3, 3, 3*i+k+1)
            x = list(range(1, len(get_data[0])+1))
            for m in range (len(get_data)):
                plt.plot(x, get_data[m])
            plt.grid()
            plt.xlabel('time')
            plt.ylabel(parts[k]+' - '+select_joints[i])
    # fig.suptitle(select_joints[i]+' data for subject MN02')
plt.show()




# data_01 = training_sequence[0]
# data_02 = training_sequence[1]
# data_03 = training_sequence[2]
# data_04 = training_sequence[3]

# ##################### to plot the example data #################

# fig = plt.figure(figsize=(2, 4))

# plt.subplot(2, 4, 1)
# x = list(range(1, len(data_01[0])+1 ))
# for i in range (len(data_01)):
# 	plt.plot(x, data_01[i])
# plt.grid()
# plt.xlabel('step')
# plt.ylabel('LFemur position - MN02')

# plt.subplot(2, 4, 2)
# x = list(range(1, len(data_02[0])+1 ))
# for i in range (len(data_02)):
# 	plt.plot(x, data_02[i])
# plt.grid()
# plt.xlabel('step')
# plt.ylabel('LFemur position - MN03')


# plt.subplot(2, 4, 3)
# x = list(range(1, len(data_03[0])+1 ))
# for i in range (len(data_03)):
# 	plt.plot(x, data_03[i])
# plt.grid()
# plt.xlabel('step')
# plt.ylabel('LFemur position - MN04')


# plt.subplot(2, 4, 4)
# x = list(range(1, len(data_04[0])+1 ))
# for i in range (len(data_04)):
# 	plt.plot(x, data_04[i])
# plt.grid()
# plt.xlabel('step')
# plt.ylabel('LFemur position - MN05')

# # plt.show()



# # # # ##################### to plot the pred data #################



