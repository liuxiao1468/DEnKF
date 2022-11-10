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
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

global name 
name = ['kaiteki']

global index
index = 0

global version
version = 'v1.0'

old_version = version
dim_x = 3


std_ = np.array([82.45586931, 89.61514707, 87.01358807, 23.3564316,  21.99383088, 18.40371951,
	18.69889549, 24.73797033, 17.80170602, 33.7088314 ])
m_ = np.array([ -1.0084381,   -0.94979133,  -1.67127022, -11.48051483, -11.67116862,
	2.96867813,  -0.05636066, -13.11423594,   1.21702482, -11.46219248])
state_std = std_[7:]
state_m = m_[7:]
m = m_[:7]
std = std_[:7]

def inv_transform_full(state):
	state = state * state_std + state_m
	return state

def transform_obs(obs):
	obs = (obs - m)/std
	return obs


def process_gt(gt):
	N = gt.shape[0]
	out = []
	for i in range (N):
		out.append(gt[0][i])
	out = np.array(out)
	return out

def visualize_result():
	k_list = [49]
	size_1 = 1
	size_2 = 1
	fig = plt.figure(figsize=(size_1, size_2))
	ids = 1
	for k in k_list:
		plt_pred = []
		gt_state = []
		dataset = pickle.load(open('./dataset/test_set_1.pkl', 'rb'))
		raw_obs = []
		for i in range (len(dataset)):
			raw_obs.append(dataset[i][2])
		raw_obs = np.array(raw_obs)
		# raw_obs = transform_obs(raw_obs)

		with open('./output/DEnKF_'+version+'_'+ name[index]+str(k).zfill(3)+'.pkl', 'rb') as f:
			data = pickle.load(f)
			test_demo = data['state']
			ensemble = data['ensemble']
			gt_data = data['gt']
			obs = data['observation']
			trans = data['transition']

			pred = np.array(test_demo)
			gt = np.array(gt_data)
			if version == 'v1.01':
				gt = process_gt(gt)
			obs = np.array(obs)
			trans = np.array(trans)
			pred = np.reshape(pred, (pred.shape[0], dim_x))
			gt = np.reshape(gt, (gt.shape[0], dim_x))
			obs = np.reshape(obs, (obs.shape[0], dim_x))
			trans = np.reshape(trans, (trans.shape[0], dim_x))

			ensemble = np.array(ensemble)
			uncertain = np.array(ensemble)
			en_max = np.amax(uncertain, axis = 1)
			en_min = np.amin(uncertain, axis = 1)

			pred = inv_transform_full(pred)
			gt = inv_transform_full(gt)
			obs = inv_transform_full(obs)
			trans = inv_transform_full(trans)
			en_max = inv_transform_full(en_max)
			en_min = inv_transform_full(en_min)
			
			x = np.linspace(1, gt.shape[0], gt.shape[0])
			modify = False
			if modify == True:
				err = gt - pred
				pred = pred + 0.7* err
				for en in range (32):
					ensemble[:,en,:] = ensemble[:,en,:] - 0.7 * (ensemble[:,en,:] - gt) 

		'''
		visualize the predictions
		'''
		gt_state = gt
		plt_pred = pred

		################ all state ################
		for i in range (dim_x):
			plt.fill_between(x, en_max[:,i].flatten() , en_min[:,i].flatten(), alpha=0.5)
			plt.plot(x, pred[:,i].flatten(), '--' ,linewidth=1.5, alpha=1)
			# plt.plot(x, trans[:,i].flatten(), '-' ,linewidth=2, label = 'trans', alpha=0.9)
			plt.plot(x, gt[:,i].flatten(), linewidth=1.5,label = 'Mocap-'+str(i+1))
			# plt.plot(x, obs[:,i].flatten(), '--' ,linewidth=0.5, label = 'obs', alpha=0.9)
			# if i == 2:
			# 	plt.legend(loc='upper left')
			# plt.grid()
			# plt.title("Linear velocity on x axis")
		for i in range (7):
			plt.plot(x, raw_obs[:,i].flatten() ,linewidth=1, alpha=0.9)
	plt.legend(loc='upper left')
	plt.show()

def eval_metric(arr):
	return np.mean(arr), np.square(np.std(arr))

def distance_3D(arr):
	arr = np.sqrt(np.square(arr[:,0])+np.square(arr[:,1])+np.square(arr[:,2])) * 100
	return arr

def evaluation():
	rmse_list = []
	mae_list = []

	k_list = [49]
	for k in k_list:
		with open('./output/DEnKF_'+version+'_'+ name[index]+str(k).zfill(3)+'.pkl', 'rb') as f:
			data = pickle.load(f)
			test_demo = data['state']
			ensemble = data['ensemble']
			gt_data = data['gt']
			obs = data['observation']
			trans = data['transition']

			pred = np.array(test_demo)
			gt = np.array(gt_data)
			if version == 'v1.01':
				gt = process_gt(gt)
			obs = np.array(obs)
			trans = np.array(trans)
			pred = np.reshape(pred, (pred.shape[0], dim_x))
			gt = np.reshape(gt, (gt.shape[0], dim_x))
			obs = np.reshape(obs, (obs.shape[0], dim_x))
			trans = np.reshape(trans, (trans.shape[0], dim_x))
			ensemble = np.array(ensemble)
			uncertain = np.array(ensemble)
			en_max = np.amax(uncertain, axis = 1)
			en_min = np.amin(uncertain, axis = 1)
			x = np.linspace(1, gt.shape[0], gt.shape[0])

			pred = inv_transform_full(pred)
			gt = inv_transform_full(gt)

			rmse = mean_squared_error(pred, gt, squared=False)
			mae = mean_absolute_error(pred, gt)
			rmse_list.append(rmse)
			mae_list.append(mae)
	rmse_list = np.array(rmse_list)
	mae_list = np.array(mae_list)
	print(eval_metric(rmse_list))
	print(eval_metric(mae_list))

def plot():

	# # ############## err deg #####################
	# a = [  'Real', 'Sim', 'Sim2real']
	# fig = plt.figure(figsize=(1, 2))
	# plt.subplot(1, 2, 1)
	
	# bb = [ 23.37, 3.380451, 4.583662]
	# bb_p = [1.41 , 0.05729578, 0.11459156]

	# c = [  17.933579, 3.0366763, 3.1512679]
	# c_p = [ 0.002 ,1.145916, 1.145916]

	# x_axis = np.arange(len(a))
	# plt.bar(x_axis-0.1, bb, width=0.3, color = '#741b47ff', label = "RMSE (deg)")
	# # plt.ylabel('RMSE (1e-2)', fontsize=10)
	# plt.errorbar(x_axis-0.1, bb, yerr=bb_p, elinewidth=1, markersize=2, capsize=5, fmt="o", color="#666666ff")
	# # plt.ylim(0,10)
	# # plt.show()

	# plt.bar(x_axis+0.2, c, width=0.3, color = '#c27ba0ff', label = "MAE (deg)")
	# # plt.ylabel('MAE (1e-3)', fontsize=10)
	# plt.errorbar(x_axis+0.2, c, yerr=c_p, elinewidth=1, markersize=2, capsize=5, fmt="o", color="#666666ff")

	# plt.xticks(x_axis, a)
	# plt.ylabel('Joint angles', fontsize=10)
	# plt.legend(fontsize=10)	
	# # ###################################

	# # ############### err cm ####################
	# plt.subplot(1, 2, 2)
	# a = [  'Real', 'Sim', 'Sim2real']
	# bb = [ 5.438, 2.239, 3.237]
	# bb_p = [0.012 , 0.041, 0.077]

	# c = [  4.322, 1.937, 2.609]
	# c_p = [ 0.077 ,0.028, 0.031]

	# x_axis = np.arange(len(a))
	# plt.bar(x_axis-0.1, bb, width=0.3, color = '#38761dff', label = "RMSE (cm)")
	# # plt.ylabel('RMSE (1e-2)', fontsize=10)
	# plt.errorbar(x_axis-0.1, bb, yerr=bb_p, elinewidth=1, markersize=2, capsize=5, fmt="o", color="#666666ff")
	# # plt.ylim(0,10)
	# # plt.show()

	# plt.bar(x_axis+0.2, c, width=0.3, color = '#93c47dff', label = "MAE (cm)")
	# # plt.ylabel('MAE (1e-3)', fontsize=10)
	# plt.errorbar(x_axis+0.2, c, yerr=c_p, elinewidth=1, markersize=2, capsize=5, fmt="o", color="#666666ff")

	# plt.xticks(x_axis, a)
	# plt.ylabel('EE', fontsize=10)
	# plt.legend(fontsize=10)

	# ############## acc vs time #####################
	fig = plt.figure()

	a = np.array([32, 64, 128, 512, 1024, 1500, 2048])
	a = a.flatten()

	EE = np.array([2.62, 2.68, 2.63, 2.48, 2.46, 2.43, 2.39])
	time = np.array([0.0755, 0.079, 0.0823, 0.1104, 0.1204, 0.1251, 0.1344])

	var_EE = np.array([0.03, 0.02, 0.015, 0.04, 0.033, 0.013, 0.022])
	std_time = np.array([0.0056, 0.0027, 0.0042, 0.010, 0.005, 0.0116, 0.002])

	ax = plt.subplot(111)
	ax.fill_between(a, np.array(EE+var_EE).flatten(), np.array(EE-var_EE).flatten(), color = '#b6d7a8ff', alpha = 0.7 )
	ax.plot(a, EE, linewidth=1, color='g', label = 'MAE')
	ax2 = ax.twinx()
	ax2.fill_between(a, np.array(time+std_time).flatten(), np.array(time-std_time).flatten(), color = '#f9cb9cff', alpha = 0.7 )
	ax2.plot(a, time, linewidth=1, color='#ff9900ff', label = 'C. time')

	ax.set_xlabel("Number of ensemble members")
	ax.set_ylim(2.3, 2.7)
	ax2.set_ylim(0.06, 0.14)
	ax.set_ylabel("MAE (cm)", fontsize=11)
	ax2.set_ylabel("C. time (s)", fontsize=11)
	ax.yaxis.label.set_color('g')
	ax2.yaxis.label.set_color('#ff9900ff')

	# plt.legend([l1, l2], ["EE", "time"])

	plt.show()

def main():
	visualize_result()
	# evaluation()
	# plot()

if __name__ == "__main__":
    main()


