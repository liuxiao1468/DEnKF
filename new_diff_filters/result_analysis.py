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


def data_loader_function(data_path):
    name = ['constant', 'exp']
    num_sensors = 100

    observations = []
    states_true = []

    s = 1/25.
    with open(data_path, 'rb') as f:
        traj = pickle.load(f)
    for i in range (len(traj['xTrue'])):
        observation = []
        state = []
        for j in range (num_sensors):
            observe = [(traj['sensors'][i][0][j])*s, traj['sensors'][i][1][j]*s]
            observation.append(observe)
            angles = traj['xTrue'][i][2]
            xTrue = [(traj['xTrue'][i][0])*s, traj['xTrue'][i][1]*s, np.cos(angles), np.sin(angles), traj['xTrue'][i][3]]
            state.append(xTrue)
        observations.append(observation)
        states_true.append(state)
    observations = np.array(observations)
    observations = tf.reshape(observations, [len(traj['xTrue']), num_sensors, 1, 2])
    states_true = np.array(states_true)
    states_true = tf.reshape(states_true, [len(traj['xTrue']), num_sensors, 1, 5])
    return observations, states_true

'''
load the data with the correct index of test observation sequence
'''
observations, states_true = data_loader_function('./dataset/100_demos_'+name[index]+'.pkl')
test = observations[:, 98, :,:]
test = tf.reshape(test, [observations.shape[0], 1, 1, 2])
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

with open('./output/v1.1_'+ name[index] +'_.pkl', 'rb') as f:
    data = pickle.load(f)
    test_demo = data['state']
    ensemble = data['ensemble']



for i in range (len(states_true)):
	gt_state.append(np.array(states_true[i][ind][0][0:2]*scale ))

	ori_gt.append(np.array(states_true[i][ind][0][2:4]))
	ori_pred.append(np.array(test_demo[i][ind][2:4]))
	plt_pred.append(np.array(test_demo[i][ind][0:2] *scale))

	plt_observation.append(np.array(test[i][ind][0][0:2]*scale))

	ensemble_1.append(np.array(ensemble[i][1][0:2] *scale))
	ensemble_2.append(np.array(ensemble[i][2][0:2] *scale))
	ensemble_3.append(np.array(ensemble[i][3][0:2] *scale))
	ensemble_4.append(np.array(ensemble[i][4][0:2] *scale))
	ensemble_5.append(np.array(ensemble[i][5][0:2] *scale))

gt_state = np.array(gt_state)
plt_pred = np.array(plt_pred)
plt_observation = np.array(plt_observation)
plt_ori_gt = np.array(ori_gt)
plt_ori_pred = np.array(ori_pred) 




ensemble_1 = np.array(ensemble_1)
ensemble_2 = np.array(ensemble_2)
ensemble_3 = np.array(ensemble_3)
ensemble_4 = np.array(ensemble_4)
ensemble_5 = np.array(ensemble_5)


'''
visualize the predictions
'''

fig = plt.figure()
# fig.suptitle('output')
plt.plot(gt_state[:, 0].flatten(),gt_state[:, 1].flatten(),color = '#e06666ff', linewidth=3.0,label = 'ground truth')

# plt.plot(ensemble_1[:, 0].flatten(),ensemble_1[:, 1].flatten(), '-*',linewidth=0.5, label = 'ensemble_1', alpha=0.3)
# plt.plot(ensemble_2[:, 0].flatten(),ensemble_2[:, 1].flatten(), '-*',linewidth=0.5, label = 'ensemble_2', alpha=0.3)
# plt.plot(ensemble_3[:, 0].flatten(),ensemble_3[:, 1].flatten(), '-*',linewidth=0.5, label = 'ensemble_3', alpha=0.3)
# plt.plot(ensemble_4[:, 0].flatten(),ensemble_4[:, 1].flatten(), '-*',linewidth=0.5, label = 'ensemble_4', alpha=0.3)
# plt.plot(ensemble_5[:, 0].flatten(),ensemble_5[:, 1].flatten(), '-*',linewidth=0.5, label = 'ensemble_5', alpha=0.3)

plt.plot(plt_pred[:, 0].flatten(),plt_pred[:, 1].flatten(), '--', color = '#4a86e8ff' ,linewidth=2, label = 'prediction', alpha=0.8)




plt.scatter(plt_observation[:,0], plt_observation[:,1], s = 15, color = '#6aa84fff', lw=0.05, label = 'observation')
plt.xlim(-12,12)
plt.ylim(-2,22)
plt.legend(loc='lower right')
plt.grid()
# plt.xlabel('x')
# plt.ylabel('y')
plt.show()

'''
visualize the states
'''

x = list(range(1, gt_state.shape[0]+1))
plt.figure(figsize=(2, 2))
plt.subplot(2, 2, 1)
plt.plot(x, gt_state[:, 0].flatten(), color = '#e06666ff', linewidth=3.0,label = 'ground truth')
plt.plot(x, plt_pred[:, 0].flatten() ,"--", color = '#4a86e8ff' ,linewidth=1.2, label = 'prediction', alpha=0.3)
plt.scatter(x, plt_observation[:,0], s = 15, color = '#6aa84fff', lw=0.05, label = 'observation')
plt.legend(loc='upper right')
plt.grid()
# plt.xlabel('x')
# plt.ylabel('y')
plt.subplot(2, 2, 2)
plt.plot(x, gt_state[:, 1].flatten(), color = '#e06666ff', linewidth=3.0,label = 'ground truth')
plt.plot(x, plt_pred[:, 1].flatten() ,"--", color = '#4a86e8ff' ,linewidth=1.2, label = 'prediction', alpha=0.3)
plt.scatter(x, plt_observation[:,1], s = 15, color = '#6aa84fff', lw=0.05, label = 'observation')
plt.legend(loc='upper right')
plt.grid()
# plt.xlabel('x')
# plt.ylabel('y')

plt.subplot(2, 2, 3)
plt.plot(x, plt_ori_gt[:, 0].flatten(), color = '#e06666ff', linewidth=3.0,label = 'ground truth')
plt.plot(x, plt_ori_pred[:, 0].flatten() ,"--", color = '#4a86e8ff' ,linewidth=1.2, label = 'prediction', alpha=0.3)
plt.legend(loc='upper right')
plt.grid()
# plt.xlabel('x')
# plt.ylabel('y')


plt.subplot(2, 2, 4)
plt.plot(x, plt_ori_gt[:, 1].flatten(), color = '#e06666ff', linewidth=3.0,label = 'ground truth')
plt.plot(x, plt_ori_pred[:, 1].flatten() ,"--", color = '#4a86e8ff' ,linewidth=1.2, label = 'prediction', alpha=0.3)
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
