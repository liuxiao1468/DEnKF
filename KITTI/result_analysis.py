import pickle
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Times New Roman"
plt.rcParams['savefig.dpi'] = 500
import numpy as np
import pdb
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf 
import pickle
import math


'''
load data for training
'''
global name 
# name = ['local_track']
name = ['checkstand', 'checkstandleft', 'coolerleft', 'entranceleft', 'sideright', 'smoothiebar']

global tracker_id
tracker_id = ['checkstand-6', 'checkstandleft-6', 'coolerleft-4', 'entranceleft-3', 'sideright-3', 'smoothiebar-3']

global index
index = 2

global version
version = 'v1.0'
old_version = version

# load the parameter
parameter = pickle.load(open('./dataset/parameter.pkl', 'rb'))

# pre state
m1 = parameter['m1']
std1 = parameter['std1']

# gt state
m2 = parameter['m2']
std2 = parameter['std2']

# bbx observation
m3 = parameter['m3']
std3 = parameter['std3']

# pose observation
m4 = parameter['m4']
std4 = parameter['std4']

(width, height) = (1920,1080) # the camera frame
(w, h) = (2550, 1650) # the floor plan

(width, height) = (1,1) # the camera frame
(w, h) = (1, 1) # the floor plan
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

k_list = [9]
size_1 = 1
size_2 = 1
fig = plt.figure(figsize=(size_1, size_2))
ids = 1
for k in k_list:
	plt_pred = []
	gt_state = []
	plt.subplot(size_1, size_2, ids)
	with open('./output/DEnKF_'+version+'_'+ name[index]+str(k).zfill(3)+'.pkl', 'rb') as f:
	    data = pickle.load(f)
	    test_demo = data['observation']
	    ensemble = data['ensemble']
	    gt_data = data['gt']

	test = np.array(test_demo)

	ensemble = np.array(ensemble)
	print(ensemble.shape)


	ensemble[:, :, 0] = ensemble[:, :, 0]*w
	ensemble[:, :, 1] = ensemble[:, :, 1]*h

	uncertain = np.array(ensemble)
	en_max = np.amax(uncertain, axis = 1)
	en_min = np.amin(uncertain, axis = 1)

	

	for i in range (len(gt_data)):
		gt_state.append(np.array([gt_data[i][ind][0][0]*w, gt_data[i][ind][0][1]*h] ))
		plt_pred.append(np.array([test_demo[i][ind][0][0] *w, test_demo[i][ind][0][1] *h]))
	gt_state = np.array(gt_state)
	plt_pred = np.array(plt_pred)
	# gt_state = gt_state*std2+m2
	# plt_pred = plt_pred*std1+m1
	print(gt_state.shape)
	print(plt_pred.shape)

	'''
	visualize the predictions
	'''
	# fig.suptitle('output')
	colors = plt.cm.Set3(np.linspace(0,1,ensemble.shape[1]))
	# for i in range (ensemble.shape[1]):
	# 	plt.plot(ensemble[:,i, 0].flatten(),ensemble[:,i, 1].flatten(), '.',linewidth=0.2, color =colors[i], alpha=0.2)
	plt.plot(gt_state[:, 0].flatten(),gt_state[:, 1].flatten(),color = '#e06666ff', linewidth=3.0,label = 'ground truth')
	plt.plot(plt_pred[:, 0].flatten(),plt_pred[:, 1].flatten(), '--', color = '#4a86e8ff' ,linewidth=2, label = 'observation', alpha=0.9)
	# plt.scatter(plt_observation[:,0], plt_observation[:,1], s = 15, color = '#6aa84fff', lw=0.05, label = 'observation')
	# plt.xlim(550,2050)
	# plt.ylim(0,500)
	plt.legend(loc='upper right')
	plt.grid()
	# plt.xlabel('x')
	# plt.ylabel('y')
	plt.title("Epoch "+str(k+1).zfill(2))
	ids = ids + 1
plt.show()


