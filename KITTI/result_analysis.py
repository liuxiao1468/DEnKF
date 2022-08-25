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


global name 
name = ['KITTI', 'sensor']

global index
index = 1

global version
version = 'vS.01'
old_version = version

only_sensor = True

def motion_model(inputs):
	v, theta_dot = inputs
	x = 0
	y = 0
	theta = 0
	final_x = []
	final_y = []
	v.shape[0]
	# print(v.shape)
	# print(theta_dot.shape)
	for i in range (v.shape[0]):
		# dt = 0.103
		theta = theta + theta_dot[i]
		x = x + v[i]* np.sin(theta)
		y = y + v[i]* np.cos(theta)
		final_x.append(x)
		final_y.append(y)
	final_x = np.array(final_x)
	final_y = np.array(final_y)
	return final_x, final_y

def inv_transform(state):
	parameters = pickle.load(open('parameters.pkl', 'rb'))
	v_m = parameters['v_m']
	v_std = parameters['v_std']
	theta_dot_m = parameters['theta_dot_m']
	theta_dot_std = parameters['theta_dot_std']

	v = state[:,0]
	theta_dot = state[:,1]

	v = v * v_std + v_m
	theta_dot = theta_dot * theta_dot_std + theta_dot_m

	state = np.vstack([v, theta_dot])
	state = state.T
	return state


scale = 1
dim_x = 2

k_list = [19]
size_1 = 2
size_2 = 1
fig = plt.figure(figsize=(size_1, size_2))
ids = 1
for k in k_list:
	plt_pred = []
	gt_state = []
	
	with open('./output/DEnKF_'+version+'_'+ name[index]+str(k).zfill(3)+'.pkl', 'rb') as f:
		data = pickle.load(f)
		if only_sensor == True:
			gt_data = data['gt_observation']
			obs = data['observation']
		else:
			test_demo = data['state']
			ensemble = data['ensemble']
			gt_data = data['gt']
			obs = data['observation']
			trans = data['transition']
	if only_sensor == True:
		gt = np.array(gt_data)
		obs = np.array(obs)
		gt = np.reshape(gt, (gt.shape[0], dim_x))
		obs = np.reshape(obs, (obs.shape[0], 2))
		x = np.linspace(1, gt.shape[0], gt.shape[0])
		obs = inv_transform(obs)
		gt = inv_transform(gt)
		gt_inputs = (gt[:,0], gt[:,1])
		sensor = (obs[:,0], obs[:,1])
		gt_x, gt_y  = motion_model(gt_inputs)
		obs_x, obs_y  = motion_model(sensor)
	else:
		pred = np.array(test_demo)
		gt = np.array(gt_data)
		obs = np.array(obs)
		trans = np.array(trans)
		pred = np.reshape(pred, (pred.shape[0], dim_x))
		gt = np.reshape(gt, (gt.shape[0], dim_x))
		obs = np.reshape(obs, (obs.shape[0], 2))
		trans = np.reshape(trans, (trans.shape[0], dim_x))
		ensemble = np.array(ensemble)
		uncertain = np.array(ensemble)
		en_max = np.amax(uncertain, axis = 1)
		en_min = np.amin(uncertain, axis = 1)
		x = np.linspace(1, gt.shape[0], gt.shape[0])
		if dim_x == 2:
			trans = inv_transform(trans)
			obs = inv_transform(obs)
			pred = inv_transform(pred)
			gt = inv_transform(gt)
			inputs = (pred[:,0], pred[:,1])
			gt_inputs = (gt[:,0], gt[:,1])
			sensor = (obs[:,0], obs[:,1])
			final_x, final_y  = motion_model(inputs)
			gt_x, gt_y  = motion_model(gt_inputs)
			obs_x, obs_y  = motion_model(sensor)
	'''
	visualize the predictions
	'''
	# plt.subplot(size_1, size_2, ids)
	# # # fig.suptitle('output')
	# # colors = plt.cm.Set3(np.linspace(0,1,ensemble.shape[1]))
	# # for i in range (10):
	# # 	plt.plot(ensemble[:,i, 0].flatten(),ensemble[:,i, 1].flatten(), '.',linewidth=0.2, color =colors[i], alpha=0.2)
	# # plt.plot(gt[:,0].flatten(),gt[:,1].flatten(),color = '#e06666ff', linewidth=3.0,label = 'ground truth')
	# # plt.plot(pred[:,0].flatten(),pred[:,1].flatten(), '--', color = '#4a86e8ff' ,linewidth=2, label = 'pred', alpha=0.9)
	# plt.plot(gt_x.flatten(),gt_y.flatten(),color = '#e06666ff', linewidth=3.0,label = 'ground truth')
	# # plt.plot(final_x.flatten(),final_y.flatten(), '--', color = '#4a86e8ff' ,linewidth=2, label = 'pred', alpha=0.9)
	# plt.plot(obs_x.flatten(),obs_y.flatten(), '--', color = 'g' ,linewidth=2, label = 'pred', alpha=0.9)
	# # plt.scatter(plt_observation[:,0], plt_observation[:,1], s = 15, color = '#6aa84fff', lw=0.05, label = 'observation')
	# plt.xlabel('x')
	# plt.ylabel('y')
	# plt.legend(loc='upper right')
	# plt.title('Epoch-'+str(k+1))
	# ids = ids + 1

	################ all state ################
	
	# for i in range (dim_x):
	# 	plt.subplot(size_1, size_2, ids)
	# 	plt.plot(x, pred[:,i].flatten(), '--', color = '#4a86e8ff' ,linewidth=2, label = 'pred', alpha=0.9)
	# 	plt.plot(x, trans[:,i].flatten(), '-' ,linewidth=2, label = 'trans', alpha=0.9)
	# 	plt.plot(x, gt[:,i].flatten(),color = '#e06666ff', linewidth=2.0,label = 'ground truth')
	# 	# plt.plot(x, obs[:,i].flatten(), '--', color = '#b6d7a8ff' ,linewidth=2, label = 'obs', alpha=0.9)
	# 	plt.legend(loc='upper right')
	# 	plt.grid()
	# 	# plt.title("Linear velocity on x axis")
	# 	ids = ids + 1

	################ sensor model ################
	plt.subplot(size_1, size_2, ids)
	plt.plot(x, obs[:,0].flatten(), '--', color = '#4a86e8ff' ,linewidth=2, label = 'pred', alpha=0.9)
	plt.plot(x, gt[:,0].flatten(),color = '#e06666ff', linewidth=2.0,label = 'ground truth')
	ids = ids + 1
	plt.subplot(size_1, size_2, ids)
	plt.plot(x, obs[:,1].flatten(), '--', color = '#4a86e8ff' ,linewidth=2, label = 'pred', alpha=0.9)
	plt.plot(x, gt[:,1].flatten(),color = '#e06666ff', linewidth=2.0,label = 'ground truth')
	plt.legend(loc='upper right')

plt.show()


