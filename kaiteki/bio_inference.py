import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as Rot
import random
import tensorflow as tf
import time
import pickle
import pdb
import tensorflow_probability as tfp

import diff_bioKF



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
define the training loop
'''
def run_filter(mode):
    if mode == True:
        # define batch_sizepython
        batch_size = 32

        # define number of ensemble
        num_ensemble = 32

        # define dropout rate
        dropout_rate = 0.4

        # load the model
        model = diff_bioKF.RNNmodel(batch_size, num_ensemble, dropout_rate)

        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        epoch = 50

        for k in range (epoch):
            '''
            data preprocessing steps
            '''
            select = random.sample(range(0, 35), batch_size)
            raw_sensor = []
            gt = []
            for idx in select:
                raw_sensor.append(observations[:, idx, :,:])
                gt.append(states_true[:, idx, :,:])
            raw_sensor = tf.convert_to_tensor(raw_sensor, dtype=tf.float32)
            raw_sensor = tf.reshape(raw_sensor, [observations.shape[0], batch_size, 1, 6])
            gt = tf.convert_to_tensor(gt, dtype=tf.float32)
            gt = tf.reshape(gt, [states_true.shape[0], batch_size, 1, 3])

            print("========================================= working on epoch %d =========================================: " % (k))

            for i in range(states_true.shape[0]):

                start = time.time()

                with tf.GradientTape() as tape:
                    out = model(raw_sensor[i])
                    state_h = out[0]
                    loss = get_loss._mse( gt[i] - state_h)

                # grads = tape.gradient(loss, model.variables)
                # optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
                grads = tape.gradient(loss, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))
                end = time.time()
                # print(model.summary())
                

                # Log every 50 batches.
                if i % 100 == 0:
                    print("Training loss at step %d: %.4f (took %.3f seconds) " %
                          (i, float(loss), float(end-start)))
                    print(state_h[0])
                    print(gt[i][0])
                    # print(out[4][0])
                    print('---')
            if (k+1) % 50 == 0:
                model.save_weights('bio_model.h5')
                print('model is saved at this epoch')
    else:
        # define batch_size
        batch_size = 1

        num_ensemble = 32

        dropout_rate = 0.4

        # load the model
        model = diff_bioKF.RNNmodel(batch_size, num_ensemble, dropout_rate)

        test_demo = observations[:, 40, :,:]
        test_demo = tf.reshape(test_demo, [observations.shape[0], 1, 1, 6])
        dummy = model(test_demo[0])
        model.load_weights('bio_model.h5')
        model.summary()

        '''
        run a test demo and save the state of the test demo
        '''
        data = {}
        data_save = []
        emsemble_save = []

        for t in range (states_true.shape[0]):
            out = model(test_demo[t])
            state_out = np.array(out[0])
            ensemble = np.array(tf.reshape(out[1], [num_ensemble, 3]))
            # print('----------')
            # print(ensemble)
            data_save.append(state_out)
            emsemble_save.append(ensemble)
        data['state'] = data_save
        data['ensemble'] = emsemble_save

        with open('bio_pred_v1.2.pkl', 'wb') as f:
            pickle.dump(data, f)


'''
load loss functions
'''
get_loss = diff_bioKF.getloss()

'''
load data for training
'''

raw_train = get_joint_data('MN02')
states_true, observations = reformat_train_data(raw_train)
# print(states_true.shape)
# print(observations.shape)


def main():

    training = True
    run_filter(training)

    training = False
    run_filter(training)

if __name__ == "__main__":
    main()


