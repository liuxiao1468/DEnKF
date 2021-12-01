import pickle
import matplotlib.pyplot as plt
import numpy as np
import pdb
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf 
import pickle
import math

import json

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

def get_joint_data(person_id):
    condition_list = ['walk']
    parts = ['kinematics', 'kinetics', 'power']
    select_joints = ['LAnkle','LKnee','LHip']
    for j in range (len(parts)):
        for i in range (len(select_joints)):
            get_data = data[condition_list[0]][person_id][select_joints[i]][parts[j]]
            get_data = preprocess_sequence(get_data)
            get_data = np.array(get_data)
            if (i == 0 and j ==0):
                train = get_data
            else:
                train = np.vstack((train, get_data))
    return train

def reformat_train_data(raw_train):
    gt = []
    observations = []
    num_points = raw_train.shape[1]
    seg = int(raw_train.shape[0]/9)
    s1 = 1/80.
    s2 = 1/2.
    s3 = 1/5.

    for i in range (num_points):
        gt_tmp = []
        o_tmp = []
        for j in range (seg):
            gt_1 = [raw_train[2*seg+j,i]*s1]
            gt_2 = [raw_train[5*seg+j,i]*s2]
            gt_3 = [raw_train[8*seg+j,i]*s3]
            gt_tmp.append(gt_1+gt_2+gt_3)

            o_1 = [raw_train[0*seg+j,i]*s1]
            o_2 = [raw_train[3*seg+j,i]*s2]
            o_3 = [raw_train[6*seg+j,i]*s3]
            o_4 = [raw_train[1*seg+j,i]*s1]
            o_5 = [raw_train[4*seg+j,i]*s2]
            o_6 = [raw_train[7*seg+j,i]*s3]
            o_tmp.append(o_1+o_2+o_3+o_4+o_5+o_6)

        gt.append(gt_tmp)
        observations.append(o_tmp)

    gt = np.array(gt)
    # gt = np.swapaxes(gt,0,1)
    observations = np.array(observations)
    # observations = np.swapaxes(observations,0,1)
    observations = tf.reshape(observations, [num_points, seg, 1, 6])

    states_true = tf.reshape(gt, [num_points, seg, 1, 3])
    return states_true, observations

def transition_dataloader(states_true, observations):
    num_points = states_true.shape[0]

    observations = observations[0:num_points-1, :,:,:]
    input_state = states_true[0:num_points-1, :,:,:]
    states_true = states_true[1:num_points, :,:,:]
    return input_state, states_true, observations

'''
load data for training
'''
raw_train = get_joint_data('MN02')
states_true, observations = reformat_train_data(raw_train)
states_true, states_true_add1, observations = transition_dataloader(states_true, observations)

test = states_true[:, 10, :,:]
test = tf.reshape(test, [states_true.shape[0], 1, 1, 3])
test = np.array(test)

'''
plot the data with its traj
'''

plt_observation = []
plt_pred = []
gt_state = []
ori_gt = []
ori_pred = []
ind = 0

scale = 1

ensemble_1 = []
ensemble_2 = []
ensemble_3 = []
ensemble_4 = []
ensemble_5 = []


num_demos = 30

test_demo = []
with open('bio_transition_v2.1.pkl', 'rb') as f:
    data = pickle.load(f)
    for i in range (num_demos):
        test_demo.append(data['state'][i])
print(len(data['state'][0][0]))

# for i in range (201):
#     print(data['state'][0][i][0])



# collect all the state variables
for j in range (num_demos):
    ori_pred_tmp = []
    plt_pred_tmp = []

    for i in range (len(states_true_add1)):
        if j == 0:
        	gt_state.append(np.array(states_true_add1[i][ind][0][0:3]*scale ))
            # gt_state.append(np.array(states_true_add1[i][ind][0:3]*scale ))
        # plt_pred_tmp.append(np.array(test_demo[j][i][ind][0][0:3] *scale))
        plt_pred_tmp.append(np.array(test_demo[j][i][ind][0:3] *scale))
    plt_pred.append(np.array(plt_pred_tmp))

gt_state = np.array(gt_state)
plt_pred = np.array(plt_pred)
print(plt_pred.shape)
print(gt_state.shape)

# gt_state = np.reshape(gt_state, (200, 3))

# collect the variance of the state variables
pred_max = np.max(plt_pred, axis = 0)
pred_min = np.min(plt_pred, axis = 0)
pred_m = np.mean(plt_pred, axis = 0)



'''
visualize the states
'''
show_points = 200

x = list(range(1, gt_state.shape[0]+1))
plt.figure(figsize=(1, 3))
plt.subplot(1, 3, 1)
plt.plot(x, gt_state[:, 0].flatten(), color = '#e06666ff', linewidth=3.0,label = 'ground truth')
plt.plot(x[0:show_points], pred_m[0:show_points, 0].flatten() ,"--", color = '#0070c0ff' ,linewidth=1.2, alpha=0.5, label = 'prediction')
plt.fill_between(x[0:show_points], pred_m[0:show_points, 0] - (pred_m[0:show_points,0] - pred_min[0:show_points,0]), pred_m[0:show_points,0] + (pred_max[0:show_points,0] - pred_m[0:show_points,0]), color='#4a86e8ff', alpha=0.2)
# plt.scatter(x, plt_observation[:,0], s = 15, color = '#6aa84fff', lw=0.05, label = 'observation')
plt.ylim(-0.3, 0.45)
plt.legend(loc='upper right')
plt.ylabel('kinematics')
plt.grid()
# plt.xlabel('x')
# plt.ylabel('y')
plt.subplot(1, 3, 2)
plt.plot(x, gt_state[:, 1].flatten(), color = '#e06666ff', linewidth=3.0,label = 'ground truth')
plt.plot(x[0:show_points], pred_m[0:show_points, 1].flatten() ,"--", color = '#0070c0ff' ,linewidth=1.2, alpha=0.5, label = 'prediction')
plt.fill_between(x[0:show_points], pred_m[0:show_points, 1] - (pred_m[0:show_points,1] - pred_min[0:show_points,1]), pred_m[0:show_points,1] + (pred_max[0:show_points,1] - pred_m[0:show_points,1]), color='#4a86e8ff', alpha=0.2)
# for i in range (num_demos):
#     plt.plot(x, plt_pred[i][:, 1].flatten() ,"--", color = '#4a86e8ff' ,linewidth=1.2, alpha=0.5)
# plt.scatter(x, plt_observation[:,1], s = 15, color = '#6aa84fff', lw=0.05, label = 'observation')
plt.legend(loc='upper right')
plt.ylim(-0.5, 0.6)
plt.ylabel('kinetics')
plt.grid()
# plt.xlabel('x')
# plt.ylabel('y')

plt.subplot(1, 3, 3)
plt.plot(x, gt_state[:, 2].flatten(), color = '#e06666ff', linewidth=3.0,label = 'ground truth')
plt.plot(x[0:show_points], pred_m[0:show_points, 2].flatten() ,"--", color = '#0070c0ff' ,linewidth=1.2, alpha=0.5, label = 'prediction')
plt.fill_between(x[0:show_points], pred_m[0:show_points, 2] - (pred_m[0:show_points,2] - pred_min[0:show_points,2]), pred_m[0:show_points,2] + (pred_max[0:show_points,2] - pred_m[0:show_points,2]), color='#4a86e8ff', alpha=0.2)
plt.legend(loc='upper right')
plt.grid()
plt.ylabel('power')
plt.ylim(-0.1, 0.3)
# plt.xlabel('x')
# plt.ylabel('y')


plt.show()

