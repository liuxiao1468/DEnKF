import pickle
import matplotlib.pyplot as plt
import numpy as np
import pdb
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf 
import pickle
import math

'''
data loader for training
observation = [timestep, batch_size, 1, dim_z] -> input data
states_true = [timestep, batch_size, 1, dim_x] -> ground truth
'''
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

'''
load the data with the correct index of test observation sequence
'''
raw_train = get_joint_data('MN02')
states_true, observations = reformat_train_data(raw_train)
test = observations[:, 40, :,:]
test = tf.reshape(test, [observations.shape[0], 1, 1, 6])
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

with open('bio_pred_v1.4.pkl', 'rb') as f:
    data = pickle.load(f)
    test_demo = data['state']
    ensemble = data['ensemble']

test_demo = np.array(test_demo)
print(test_demo.shape)
for i in range (len(states_true)):
	gt_state.append(np.array(states_true[i][ind][0][0:3]*scale ))

	# ori_gt.append(np.array(states_true[i][ind][0][2:4]))
	# ori_pred.append(np.array(test_demo[i][ind][0][2:4] ))

	plt_pred.append(np.array(test_demo[i][ind][0:3] *scale))
	# print(test_demo[i])
	# plt_observation.append(np.array(test[i][ind][0][0:2]*scale))

	# ensemble_1.append(np.array(ensemble[i][1][0:2] *scale))
	# ensemble_2.append(np.array(ensemble[i][2][0:2] *scale))
	# ensemble_3.append(np.array(ensemble[i][3][0:2] *scale))
	# ensemble_4.append(np.array(ensemble[i][4][0:2] *scale))
	# ensemble_5.append(np.array(ensemble[i][5][0:2] *scale))

gt_state = np.array(gt_state)
plt_pred = np.array(plt_pred)
# plt_observation = np.array(plt_observation)
# plt_ori_gt = np.array(ori_gt)
# plt_ori_pred = np.array(ori_pred) 




# # ensemble_1 = np.array(ensemble_1)
# # ensemble_2 = np.array(ensemble_2)
# # ensemble_3 = np.array(ensemble_3)
# # ensemble_4 = np.array(ensemble_4)
# # ensemble_5 = np.array(ensemble_5)


'''
visualize the predictions
'''

# fig = plt.figure()
# # fig.suptitle('output')
# plt.plot(gt_state[:, 0].flatten(),gt_state[:, 1].flatten(),color = '#e06666ff', linewidth=3.0,label = 'ground truth')

# plt.plot(ensemble_1[:, 0].flatten(),ensemble_1[:, 1].flatten(), '-*',linewidth=0.5, label = 'ensemble_1', alpha=0.3)
# plt.plot(ensemble_2[:, 0].flatten(),ensemble_2[:, 1].flatten(), '-*',linewidth=0.5, label = 'ensemble_2', alpha=0.3)
# plt.plot(ensemble_3[:, 0].flatten(),ensemble_3[:, 1].flatten(), '-*',linewidth=0.5, label = 'ensemble_3', alpha=0.3)
# plt.plot(ensemble_4[:, 0].flatten(),ensemble_4[:, 1].flatten(), '-*',linewidth=0.5, label = 'ensemble_4', alpha=0.3)
# plt.plot(ensemble_5[:, 0].flatten(),ensemble_5[:, 1].flatten(), '-*',linewidth=0.5, label = 'ensemble_5', alpha=0.3)

# plt.plot(plt_pred[:, 0].flatten(),plt_pred[:, 1].flatten(), '--', color = '#4a86e8ff' ,linewidth=2, label = 'prediction', alpha=0.8)




# plt.scatter(plt_observation[:,0], plt_observation[:,1], s = 15, color = '#6aa84fff', lw=0.05, label = 'observation')
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
plt.figure(figsize=(3, 1))
plt.subplot(3, 1, 1)
plt.plot(x, gt_state[:, 0].flatten(), color = '#e06666ff', linewidth=3.0,label = 'ground truth')
plt.plot(x, plt_pred[:, 0].flatten() ,"--", color = '#4a86e8ff' ,linewidth=1.2, label = 'prediction')
# plt.scatter(x, plt_observation[:,0], s = 15, color = '#6aa84fff', lw=0.05, label = 'observation')
plt.legend(loc='upper right')
plt.grid()
# plt.xlabel('x')
# plt.ylabel('y')
plt.subplot(3, 1, 2)
plt.plot(x, gt_state[:, 1].flatten(), color = '#e06666ff', linewidth=3.0,label = 'ground truth')
plt.plot(x, plt_pred[:, 1].flatten() ,"--", color = '#4a86e8ff' ,linewidth=1.2, label = 'prediction')
# plt.scatter(x, plt_observation[:,1], s = 15, color = '#6aa84fff', lw=0.05, label = 'observation')
plt.legend(loc='upper right')
plt.grid()
# plt.xlabel('x')
# plt.ylabel('y')

plt.subplot(3, 1, 3)
plt.plot(x, gt_state[:, 2].flatten(), color = '#e06666ff', linewidth=3.0,label = 'ground truth')
plt.plot(x, plt_pred[:, 2].flatten() ,"--", color = '#4a86e8ff' ,linewidth=1.2, label = 'prediction')
# plt.scatter(x, plt_observation[:,1], s = 15, color = '#6aa84fff', lw=0.05, label = 'observation')
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
