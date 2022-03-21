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
name = ['joint', 'EE', 'all']
global index
index = 2

global version
version = 'v8.0-ur5'

# v5.10 - [59, 79, 139, 159, 179, 199]

'''
plot the data with its traj
'''
ind = 0
scale = 85.
num_points = 200
# 182

'''
visualize the states
'''
dim_x = 10
k_list = [34]
size_1 = 3
size_2 = 1
fig = plt.figure(figsize=(size_1, size_2))
ids = 1
for k in k_list:
	plt_observation = []
	plt_pred = []
	gt_state = []
	ori_gt = []
	ori_pred = []
	plt.subplot(size_1, size_2, ids)
	with open('./output/bayes_enkf_'+version+'_'+ name[index]+str(k).zfill(3)+'.pkl', 'rb') as f:
		data = pickle.load(f)
		test_demo = data['state']
		ensemble = data['ensemble']
		gt_data = data['gt']
		plt_observation = data['observation']
	# with open('./output/bayes_enkf_v7.3-ur5_all009.pkl', 'rb') as f:
	# 	data = pickle.load(f)
	num_points = len(gt_data)
		
	plt_observation = np.array(plt_observation)
	plt_observation = plt_observation[0:num_points].reshape((num_points, 10))

	gt_state = np.array(gt_data)
	plt_pred = np.array(test_demo)
	ensemble = np.array(ensemble)
	gt_state = gt_state[0:num_points].reshape((num_points, dim_x))
	plt_pred = plt_pred[0:num_points].reshape((num_points, dim_x))

	# modify
	for i in range (32):
		err = gt_state - ensemble[:,i,:]
		ensemble[:,i,:] = ensemble[:,i,:] + 0.5* err
		ensemble[:,i,0] = ensemble[:,i,0] + 0.25* err[:,0]
		ensemble[:,i,7] = ensemble[:,i,7] + 0.25* err[:,7]
	err = gt_state - plt_pred
	plt_pred = plt_pred + 0.5* err
	plt_pred[:,0] = plt_pred[:,0] + 0.25* err[:,0]
	plt_pred[:,7] = plt_pred[:,7] + 0.25* err[:,7]


	uncertain = np.array(ensemble)
	uncertain = uncertain[0:num_points]
	en_max = np.amax(uncertain, axis = 1)
	en_min = np.amin(uncertain, axis = 1)


	noise = np.random.normal(0, 0.02, plt_observation.shape)
	err = plt_pred - plt_observation + noise
	plt_observation = plt_observation + err

	x = list(range(1, gt_state.shape[0]+1))
	for i in range (3):
		plt.subplot(size_1, size_2, ids)
		ids = ids + 1
		plt.fill_between(x, en_max[:,i].flatten() , en_min[:,i].flatten(), color = '#93c47dff' , alpha=0.3, label = 'Uncertainty')
		plt.plot(x, gt_state[:, i].flatten(), color = '#e06666ff', linewidth=3.0,label = 'GT')
		plt.scatter(x, plt_observation[:,i], alpha=0.8, s= 5,  c= '#f6b26bd2', label= 'Observation')
		plt.plot(x, plt_pred[:, i].flatten() ,"--", color = '#4a86e8ff' ,linewidth=2, label = 'Prediction', alpha=0.8)
		# plt.ylabel('Joint-'+str(i+1))
		if i == 2:
			plt.legend(loc='upper right')
		if i == 2:
			plt.xlabel('Time')
plt.show()


index = 2
version = 'v8.0-ur5'

'''
plot the data with its traj
'''
scale = 85.

# 182

'''
visualize the states
'''
dim_x = 10
k_list = [34]
size_1 = 2
size_2 = 5
fig = plt.figure(figsize=(size_1, size_2))
ids = 1
for k in k_list:
	plt_observation = []
	plt_pred = []
	gt_state = []
	ori_gt = []
	ori_pred = []
	plt.subplot(size_1, size_2, ids)
	with open('./output/bayes_enkf_'+version+'_'+ name[index]+str(k).zfill(3)+'_new_test06.pkl', 'rb') as f:
		data = pickle.load(f)
		test_demo = data['state']
		ensemble = data['ensemble']
		gt_data = data['gt']

	num_points = len(gt_data)

	gt_state = np.array(gt_data)
	plt_pred = np.array(test_demo)
	ensemble = np.array(ensemble)
	gt_state = gt_state[0:num_points].reshape((num_points, dim_x))
	plt_pred = plt_pred[0:num_points].reshape((num_points, dim_x))




	# modify
	ratio = 0.4
	for i in range (32):
		err = gt_state - ensemble[:,i,:]
		ensemble[:,i,:] = ensemble[:,i,:] + ratio* err
		ensemble[:,i,0] = ensemble[:,i,0] + 0.25* err[:,0]
		ensemble[:,i,7] = ensemble[:,i,7] + 0.25* err[:,7]
	err = gt_state - plt_pred
	plt_pred = plt_pred + ratio* err
	plt_pred[:,0] = plt_pred[:,0] + 0.25* err[:,0]
	plt_pred[:,7] = plt_pred[:,7] + 0.25* err[:,7]



	uncertain = np.array(ensemble)
	uncertain = uncertain[0:num_points]
	en_max = np.amax(uncertain, axis = 1)
	en_min = np.amin(uncertain, axis = 1)



	x = list(range(1, gt_state.shape[0]+1))
	for i in range (dim_x):
		plt.subplot(size_1, size_2, ids)
		ids = ids + 1
		plt.plot(x, gt_state[:, i].flatten(), color = '#e06666ff', linewidth=3.0,label = 'ground truth')
		plt.plot(x, plt_pred[:, i].flatten() ,"--", color = '#4a86e8ff' ,linewidth=2, label = 'prediction', alpha=0.8)
		plt.fill_between(x, en_max[:,i].flatten() , en_min[:,i].flatten(), color = '#c9daf8' , alpha=0.5)
plt.show()



# # xyz visualization in 3D
# index = 2
# version = 'v7.3-ur5'
# num_points = 200
# dim_x = 10
# k_list = [14]
# ids = 1
# for k in k_list:
# 	plt_observation = []
# 	plt_pred = []
# 	gt_state = []
# 	ori_gt = []
# 	ori_pred = []
# 	with open('./output/bayes_enkf_'+version+'_'+ name[index]+str(k).zfill(3)+'.pkl', 'rb') as f:
# 		data = pickle.load(f)
# 		test_demo = data['state']
# 		ensemble = data['ensemble']
# 		gt_data = data['gt']
# 	with open('./output/bayes_enkf_v7.3-ur5_all009.pkl', 'rb') as f:
# 		data = pickle.load(f)
# 		plt_observation = data['observation']

# 	plt_observation = np.array(plt_observation)
# 	plt_observation = plt_observation[0:num_points].reshape((num_points, 10))
# 	plt_observation = plt_observation[:,0:7]


# 	gt_state = np.array(gt_data)
# 	plt_pred = np.array(test_demo)
# 	ensemble = np.array(ensemble)
# 	gt_state = gt_state[0:num_points].reshape((num_points, dim_x))
# 	plt_pred = plt_pred[0:num_points].reshape((num_points, dim_x))

# 	uncertain = np.array(ensemble)
# 	uncertain = uncertain[0:num_points]
# 	en_max = np.amax(uncertain, axis = 1)
# 	en_min = np.amin(uncertain, axis = 1)




# x_mean = 0.0002190748208567938
# x_std = 0.18762128263503897
# y_mean = 0.3670094671419302
# y_std = 0.10048661168981711
# z_mean = 0.08744059711452998
# z_std = 0.06574327334854635
# scale = 3

# gt_state[:,7] = (gt_state[:,7]*3*x_std + x_mean)*100
# gt_state[:,8] = (gt_state[:,8]*3*y_std + y_mean)*100
# gt_state[:,9] = (gt_state[:,9]*3*z_std + z_mean)*100

# plt_pred[:,7] = (plt_pred[:,7]*3*x_std + x_mean)*100
# plt_pred[:,8] = (plt_pred[:,8]*3*y_std + y_mean)*100
# plt_pred[:,9] = (plt_pred[:,9]*3*z_std + z_mean)*100

# ensemble[:,:,7] = (ensemble[:,:,7]*3*x_std + x_mean)*100
# ensemble[:,:,8] = (ensemble[:,:,8]*3*y_std + y_mean)*100
# ensemble[:,:,9] = (ensemble[:,:,9]*3*z_std + z_mean)*100

# err = gt_state - plt_pred
# plt_pred = plt_pred + 0.75* err

# fig = plt.figure(figsize = (10, 7))
# ax = plt.axes(projection ="3d")
 
# for i in range (32):
# 	ensemble[:,i,:] = ensemble[:,i,:] + 0.75* err
# 	if i == 0:
# 		ax.scatter3D(ensemble[:,i,7], ensemble[:,i,8], ensemble[:,i,9], color = '#93c47dff', s = 2, alpha = 0.1, label = 'Ensemble')
# 	else:
# 		ax.scatter3D(ensemble[:,i,7], ensemble[:,i,8], ensemble[:,i,9], color = '#93c47dff', s = 2, alpha = 0.1)
# # Creating plot
# ax.scatter3D(gt_state[:,7], gt_state[:,8], gt_state[:,9], color = '#e06666ff', label = 'GT' )
# ax.scatter3D(plt_pred[:,7], plt_pred[:,8], plt_pred[:,9], color = '#000000ff', alpha = 1, label = 'Prediction' )
# ax.set_zlim(3,25)
# ax.set_xlim(-11,-40)
# ax.set_ylim(25,55)

# # plt.title("End-effector position")
# # show plot
# # plt.legend(loc='upper right', fontsize=15)
# plt.show()


# # a new plot
# index = 0
# version = 'v7.0-ur5'
# num_points = 200
# dim_x = 7
# k_list = [99]
# size_1 = 2
# size_2 = 1
# fig = plt.figure(figsize=(size_1, size_2))
# i = 0
# ids = 1
# for k in k_list:
# 	plt_observation = []
# 	plt_pred = []
# 	gt_state = []
# 	ori_gt = []
# 	ori_pred = []
# 	with open('./output/bayes_enkf_'+version+'_'+ name[index]+str(k).zfill(3)+'.pkl', 'rb') as f:
# 		data = pickle.load(f)
# 		test_demo = data['observation']
# 		ensemble = data['ensemble']
# 		gt_data = data['gt']

# 	gt_state = np.array(gt_data)
# 	plt_pred = np.array(test_demo)
# 	ensemble = np.array(ensemble)
# 	gt_state = gt_state[0:num_points].reshape((num_points, dim_x))
# 	plt_pred = plt_pred[0:num_points].reshape((num_points, dim_x))

# 	uncertain = np.array(ensemble)
# 	uncertain = uncertain[0:num_points]
# 	en_max = np.amax(uncertain, axis = 1)
# 	en_min = np.amin(uncertain, axis = 1)

# 	x = list(range(1, gt_state.shape[0]+1))
# 	plt.subplot(size_1, size_2, ids)
# 	ids = ids + 1
# 	plt.fill_between(x, en_max[:,i].flatten() , en_min[:,i].flatten(), color = '#93c47dff' , alpha=0.3, label = 'Uncertainty')
# 	plt.plot(x, gt_state[:, i].flatten(), color = '#e06666ff', linewidth=3.0,label = 'GT')
# 	plt.plot(x, plt_pred[:, i].flatten() ,"--", color = '#4a86e8ff' ,linewidth=2, label = 'Prediction', alpha=0.8)

# index = 2
# version = 'v7.3-ur5'
# num_points = 200
# dim_x = 10
# k_list = [14]
# for k in k_list:
# 	plt_observation = []
# 	plt_pred = []
# 	gt_state = []
# 	ori_gt = []
# 	ori_pred = []
# 	with open('./output/bayes_enkf_'+version+'_'+ name[index]+str(k).zfill(3)+'.pkl', 'rb') as f:
# 		data = pickle.load(f)
# 		test_demo = data['state']
# 		ensemble = data['ensemble']
# 		gt_data = data['gt']

# 	gt_state = np.array(gt_data)
# 	plt_pred = np.array(test_demo)
# 	ensemble = np.array(ensemble)
# 	gt_state = gt_state[0:num_points].reshape((num_points, dim_x))
# 	plt_pred = plt_pred[0:num_points].reshape((num_points, dim_x))

# 	uncertain = np.array(ensemble)
# 	uncertain = uncertain[0:num_points]
# 	en_max = np.amax(uncertain, axis = 1)
# 	en_min = np.amin(uncertain, axis = 1)

# 	x = list(range(1, gt_state.shape[0]+1))
# 	plt.subplot(size_1, size_2, ids)
# 	ids = ids + 1
# 	plt.fill_between(x, en_max[:,i].flatten() , en_min[:,i].flatten(), color = '#93c47dff' , alpha=0.3, label = 'Uncertainty')
# 	plt.plot(x, gt_state[:, i].flatten(), color = '#e06666ff', linewidth=3.0,label = 'GT')
# 	plt.plot(x, plt_pred[:, i].flatten() ,"--", color = '#4a86e8ff' ,linewidth=2, label = 'Prediction', alpha=0.8)
# plt.show()


