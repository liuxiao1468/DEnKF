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
name = ['KITTI']

global index
index = 0

global version
version = 'v1.0'
old_version = version


scale = 1

k_list = [79]
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
	    test_demo = data['state']
	    ensemble = data['ensemble']
	    gt_data = data['gt']

	pred = np.array(test_demo)
	gt = np.array(gt_data)

	pred = np.reshape(pred, (pred.shape[0], 5))
	gt = np.reshape(gt, (gt.shape[0], 5))

	ensemble = np.array(ensemble)
	print(ensemble.shape)

	uncertain = np.array(ensemble)
	en_max = np.amax(uncertain, axis = 1)
	en_min = np.amin(uncertain, axis = 1)

	

	'''
	visualize the predictions
	'''
	# fig.suptitle('output')
	colors = plt.cm.Set3(np.linspace(0,1,ensemble.shape[1]))
	for i in range (10):
		plt.plot(ensemble[:,i, 0].flatten(),ensemble[:,i, 1].flatten(), '.',linewidth=0.2, color =colors[i], alpha=0.2)
	plt.plot(gt[:, 0].flatten(),gt[:, 1].flatten(),color = '#e06666ff', linewidth=3.0,label = 'ground truth')
	plt.plot(pred[:, 0].flatten(),pred[:, 1].flatten(), '--', color = '#4a86e8ff' ,linewidth=2, label = 'pred', alpha=0.9)
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


