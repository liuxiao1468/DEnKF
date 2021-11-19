import pickle
import matplotlib.pyplot as plt
import numpy as np
import pdb
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf 
import pickle
import math

subtract_mean = False
sensor_idx = 0
num_sensors = 100

global name 
name = ['constant', 'exp']
global index
index = 1


def transition_data_loader_function(data_path):
    name = ['constant', 'exp']
    num_sensors = 100

    observations = []
    states_true = []
    states_true_add1 = []

    s = 1/25.
    with open(data_path, 'rb') as f:
        traj = pickle.load(f)
    for i in range (len(traj['xTrue'])-1):
        observation = []
        state = []
        state_add = []
        for j in range (num_sensors):
            observe = [(traj['sensors'][i][0][j] +0)*s, traj['sensors'][i][1][j]*s]
            observation.append(observe)
            angles = traj['xTrue'][i][2]
            xTrue = [(traj['xTrue'][i][0] + 0)*s, traj['xTrue'][i][1]*s, np.cos(angles), np.sin(angles), traj['xTrue'][i][3]]
            angles = traj['xTrue'][i+1][2]
            xTrue_add = [(traj['xTrue'][i+1][0] + 0)*s, traj['xTrue'][i+1][1]*s, np.cos(angles), np.sin(angles), traj['xTrue'][i+1][3]]
            state.append(xTrue)
            state_add.append(xTrue_add)

        observations.append(observation)
        states_true.append(state)
        states_true_add1.append(state_add)

    observations = np.array(observations)
    observations = tf.reshape(observations, [len(traj['xTrue'])-1, num_sensors, 1, 2])
    states_true = np.array(states_true)
    states_true = tf.reshape(states_true, [len(traj['xTrue'])-1, num_sensors, 1, 5])
    states_true_add1 = np.array(states_true_add1)
    states_true_add1 = tf.reshape(states_true_add1, [len(traj['xTrue'])-1, num_sensors, 1, 5])
    return observations, states_true, states_true_add1

'''
load the data with the correct index of test observation sequence
'''
observations, states_true, states_true_add1 = transition_data_loader_function('./dataset/100_demos_'+name[index]+'.pkl')

test = states_true[:, 98, :,:]
test = tf.reshape(test, [states_true.shape[0], 1, 1, 5])
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

scale = 25.

ensemble_1 = []
ensemble_2 = []
ensemble_3 = []
ensemble_4 = []
ensemble_5 = []


num_demos = 30

test_demo = []
with open('./output/transition_multi_01.pkl', 'rb') as f:
    data = pickle.load(f)
    for i in range (num_demos):
        test_demo.append(data['state'][i])



# collect all the state variables
for j in range (num_demos):
    ori_pred_tmp = []
    plt_pred_tmp = []

    for i in range (len(states_true_add1)):
        if j == 0:
        	gt_state.append(np.array(states_true_add1[i][ind][0][0:2]*scale ))
        	ori_gt.append(np.array(states_true_add1[i][ind][0][2:4]))
        ori_pred_tmp.append(np.array(test_demo[j][i][ind][0][2:4]))
        plt_pred_tmp.append(np.array(test_demo[j][i][ind][0][0:2] *scale))
    ori_pred.append(np.array(ori_pred_tmp))
    plt_pred.append(np.array(plt_pred_tmp))

gt_state = np.array(gt_state)
plt_pred = np.array(plt_pred)
plt_ori_gt = np.array(ori_gt)
plt_ori_pred = np.array(ori_pred)


# collect the variance of the state variables
pred_max = np.max(plt_pred, axis = 0)
pred_min = np.min(plt_pred, axis = 0)

ori_max = np.max(plt_ori_pred, axis = 0)
ori_min = np.min(plt_ori_pred, axis = 0)

pred_m = np.mean(plt_pred, axis = 0)
ori_m = np.mean(plt_ori_pred, axis = 0)



'''
visualize the predictions
'''

# fig = plt.figure()
# # fig.suptitle('output')
# plt.plot(gt_state[:, 0].flatten(),gt_state[:, 1].flatten(),color = '#e06666ff', linewidth=3.0,label = 'ground truth')
# for i in range (num_demos):
#     plt.plot(plt_pred[i][:, 0].flatten(),plt_pred[i][:, 1].flatten(), '--', color = '#4a86e8ff' ,linewidth=1, alpha=0.5)

# plt.xlim(-12,12)
# plt.ylim(-2,22)
# plt.legend(loc='lower right')
# plt.grid()
# # plt.xlabel('x')
# # plt.ylabel('y')
# plt.show()

'''
visualize the states
'''

x = list(range(1, gt_state.shape[0]+1))
plt.figure(figsize=(1, 4))
plt.subplot(1, 4, 1)
plt.plot(x, gt_state[:, 0].flatten(), color = '#e06666ff', linewidth=3.0,label = 'ground truth')
plt.plot(x, pred_m[:, 0].flatten() ,"--", color = '#0070c0ff' ,linewidth=1.2, alpha=0.5, label = 'prediction')
plt.fill_between(x, pred_m[:, 0] - (pred_m[:,0] - pred_min[:,0]), pred_m[:,0] + (pred_max[:,0] - pred_m[:,0]), color='#4a86e8ff', alpha=0.2)
# plt.scatter(x, plt_observation[:,0], s = 15, color = '#6aa84fff', lw=0.05, label = 'observation')
plt.legend(loc='upper right')
plt.grid()
# plt.xlabel('x')
# plt.ylabel('y')
plt.subplot(1, 4, 2)
plt.plot(x, gt_state[:, 1].flatten(), color = '#e06666ff', linewidth=3.0,label = 'ground truth')
plt.plot(x, pred_m[:, 1].flatten() ,"--", color = '#0070c0ff' ,linewidth=1.2, alpha=0.5, label = 'prediction')
plt.fill_between(x, pred_m[:, 1] - (pred_m[:,1] - pred_min[:,1]), pred_m[:,1] + (pred_max[:,1] - pred_m[:,1]), color='#4a86e8ff', alpha=0.2)
# for i in range (num_demos):
#     plt.plot(x, plt_pred[i][:, 1].flatten() ,"--", color = '#4a86e8ff' ,linewidth=1.2, alpha=0.5)
# plt.scatter(x, plt_observation[:,1], s = 15, color = '#6aa84fff', lw=0.05, label = 'observation')
plt.legend(loc='upper right')
plt.grid()
# plt.xlabel('x')
# plt.ylabel('y')

plt.subplot(1, 4, 3)
plt.plot(x, plt_ori_gt[:, 0].flatten(), color = '#e06666ff', linewidth=3.0,label = 'ground truth')
plt.plot(x, ori_m[:, 0].flatten() ,"--", color = '#0070c0ff' ,linewidth=1.2, alpha=0.5, label = 'prediction')
plt.fill_between(x, ori_m[:, 0] - (ori_m[:,0] - ori_min[:,0]), ori_m[:,0] + (ori_max[:,0] - ori_m[:,0]), color='#4a86e8ff', alpha=0.2)
plt.legend(loc='upper right')
plt.grid()
# plt.xlabel('x')
# plt.ylabel('y')


plt.subplot(1, 4, 4)
plt.plot(x, plt_ori_gt[:, 1].flatten(), color = '#e06666ff', linewidth=3.0,label = 'ground truth')
plt.plot(x, ori_m[:, 1].flatten() ,"--", color = '#0070c0ff' ,linewidth=1.2, alpha=0.5, label = 'prediction')
plt.fill_between(x, ori_m[:, 1] - (ori_m[:,1] - ori_min[:,1]), ori_m[:,1] + (ori_max[:,1] - ori_m[:,1]), color='#4a86e8ff', alpha=0.2)
# for i in range (num_demos):
#     plt.plot(x, plt_ori_pred[i][:, 1].flatten() ,"--", color = '#4a86e8ff' ,linewidth=1.2, alpha=0.5)
plt.legend(loc='upper right')
plt.grid()
# plt.xlabel('x')
# plt.ylabel('y')

plt.show()


'''
for creating the animation
'''

# import cv2

# image = cv2.imread('foo.png')
# save_width = image.shape[1] 
# height = image.shape[0]
# out0 = cv2.VideoWriter('ensemble_demo_02.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (save_width,height))

# fig = plt.figure()
# for i in range (gt_state.shape[0]):
# 	if i == 0:
# 		val_gt = gt_state[i,:]
# 		val_pred = plt_pred[i,:]
# 		val_observe = plt_observation[i,:]
# 	else:
# 		val_gt = np.vstack((val_gt, gt_state[i,:]))
# 		val_pred = np.vstack(( val_pred, plt_pred[i,:]))
# 		val_observe = np.vstack((val_observe, plt_observation[i,:]))
# 		plt.cla()
# 		# for stopping simulation with the esc key.
# 		plt.gcf().canvas.mpl_connect('key_release_event',lambda event: [exit(0) if event.key == 'escape' else None])
# 		plt.plot(val_gt[:, 0].flatten(),val_gt[:, 1].flatten(),color = '#e06666ff', linewidth=3.0,label = 'ground truth')
# 		plt.plot(val_pred[:, 0].flatten(),val_pred[:, 1].flatten(), '--', color = '#4a86e8ff' ,linewidth=1.5, alpha=0.5, label = 'prediction')
# 		plt.scatter(val_observe[:,0], val_observe[:,1], s = 15, color = '#6aa84fff', lw=0.05, label = 'observation')
# 		plt.axis("equal")
# 		plt.grid(True)
# 		plt.savefig('foo.png', bbox_inches='tight')
# 		plt.pause(0.1)
# 		image = cv2.imread('foo.png')
# 		out0.write(image)
# out0.release()
# cv2.destroyAllWindows()



'''
sanity check
'''
# with open('./output/sanity_v2.6.pkl', 'rb') as f:
#     data = pickle.load(f)
#     sanity = data['sanity']

# pos = []
# for i in range (len(sanity)):
# 	pos.append(np.array(sanity[i][0][0:2]*scale))

# pos = np.array(pos)
# print(pos)

# fig = plt.figure()
# # fig.suptitle('output')
# plt.plot(pos[:, 0].flatten(),pos[:, 1].flatten(),color = '#e06666ff', linewidth=3.0,label = 'sanity')

# # plt.xlim(-12,12)
# # plt.ylim(-2,22)
# plt.legend(loc='lower right')
# plt.grid()
# # plt.xlabel('x')
# # plt.ylabel('y')
# plt.show()
