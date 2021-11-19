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
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import tensorflow_probability as tfp

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)



# ########################### build model ##################
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

class getloss():
    def _mse(self, diff):
        """
        Returns the mean squared error of diff = label - pred plus their
        euclidean distance (dist)
        Parameters
        ----------
        diff : tensor
            difference between label and prediction
        reduce_mean : bool, optional
            if true, return the mean errors over the complete tensor. The
            default is False.
        Returns
        -------
        loss : tensor
            the mean squared error
        dist : tensor
            the euclidean distance
        """
        diff_a = tf.expand_dims(diff, axis=-1)
        diff_b = tf.expand_dims(diff, axis=-2)

        loss = tf.matmul(diff_b, diff_a)

        # the loss needs to be finite and positive
        loss = tf.where(tf.math.is_finite(loss), loss,
                        tf.ones_like(loss)*1e20)
        loss = tf.where(tf.greater_equal(loss, 0), loss,
                        tf.ones_like(loss)*1e20)

        loss = tf.squeeze(loss, axis=-1)
        dist = tf.sqrt(loss)

        loss = tf.reduce_mean(loss)
        dist = tf.reduce_mean(dist)
        loss = loss + dist

        return loss


class ProcessModel(tf.keras.Model):
    '''
    process model is taking the state and get a prediction state and 
    calculate the jacobian matrix based on the previous state and the 
    predicted state.
    new_state = [batch_size, 1, dim_x]
            F = [batch_size, dim_x, dim_x]
    state vector 4 -> fc 32 -> fc 64 -> 2
    '''
    def __init__(self, batch_size, dim_x, jacobian, rate):
        super(ProcessModel, self).__init__()
        self.batch_size = batch_size
        self.jacobian = jacobian
        self.dim_x = dim_x
        self.rate = rate

    def build(self, input_shape):
        self.process_fc1 = tf.keras.layers.Dense(
            units=32,
            activation=tf.nn.relu,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
            name='process_fc1')
        self.process_fc_add1 = tf.keras.layers.Dense(
            units=64,
            activation=tf.nn.relu,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
            name='process_fc_add1')
        self.process_fc2 = tf.keras.layers.Dense(
            units=64,
            activation=tf.nn.relu,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
            name='process_fc2')
        self.process_fc_add2 = tf.keras.layers.Dense(
            units=32,
            activation=tf.nn.relu,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
            name='process_fc_add2')
        self.process_fc3 = tf.keras.layers.Dense(
            units=self.dim_x,
            activation=None,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
            name='process_fc3')

    def call(self, last_state, training):
        last_state = tf.reshape(last_state, [self.batch_size, 1, self.dim_x])

        fc1 = self.process_fc1(last_state)
        # fc1 = tf.nn.dropout(fc1, rate=self.rate)
        fcadd1 = self.process_fc_add1(fc1)
        # fcadd1 = tf.nn.dropout(fcadd1, rate=self.rate)
        fc2 = self.process_fc2(fcadd1)
        fc2 = tf.nn.dropout(fc2, rate=self.rate)
        fcadd2 = self.process_fc_add2(fc2)
        fcadd2 = tf.nn.dropout(fcadd2, rate=self.rate)
        update = self.process_fc3(fcadd2)

        new_state = last_state + update
        new_state = tf.reshape(new_state, [self.batch_size, 1, self.dim_x])

        return new_state



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

def transition_dataloader(states_true):
    num_points = states_true.shape[0]

    states_true[0:num_points-1, :,:,:]
    states_true[1:num_points, :,:,:]
    input_state = states_true[0:num_points-1, :,:,:]
    states_true = states_true[1:num_points, :,:,:]
    return input_state, states_true

'''
define the training loop
'''
get_loss = getloss()


'''
define the training loop
'''
def run_filter(mode):
    if mode == True:
        # define batch_sizepython
        batch_size = 8

        # define dropout rate
        dropout_rate = 0.4

        process_model = ProcessModel(batch_size, 3, True, dropout_rate)

        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

        epoch = 40

        pred_steps = 4

        for k in range (epoch):
            '''
            data preprocessing steps
            '''
            input_state = []
            gt = []


            select = random.sample(range(0, 35), batch_size)
            for idx in select:
                input_state.append(states_true[:, idx, :,:])
                gt.append(states_true_add1[:, idx, :,:])
            gt = tf.convert_to_tensor(gt, dtype=tf.float32)
            gt = tf.reshape(gt, [states_true_add1.shape[0], batch_size, 1, 3])

            input_state = tf.convert_to_tensor(input_state, dtype=tf.float32)
            input_state = tf.reshape(input_state, [states_true.shape[0], batch_size, 1, 3])

            print("========================================= working on epoch %d =========================================: " % (k))
            for i in range(states_true.shape[0]-pred_steps):

                start = time.time()

                with tf.GradientTape() as tape:

                    for step in range (pred_steps):
                        if step == 0:
                            out = process_model(input_state[i+step], True)
                            state_h = out
                            loss = get_loss._mse( gt[i+step] - state_h)
                        else:
                            out = process_model(input_state[i+step], True)
                            state_h = out
                            loss = loss+ get_loss._mse( gt[i+step] - state_h)

                # grads = tape.gradient(loss, model.variables)
                # optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
                grads = tape.gradient(loss, process_model.trainable_weights)
                optimizer.apply_gradients(zip(grads, process_model.trainable_weights))
                end = time.time()
                # Log every 50 batches.
                if i % 100 == 0:
                    print("Training loss at step %d: %.4f (took %.3f seconds) " %
                          (i, float(loss), float(end-start)))
                    print(state_h[0])
                    print(gt[i][0])
                    print('---')
        if (k+1) % 40 == 0:
            process_model.save_weights('transition_model.h5')
            print('model is saved at this epoch')
    else:
        # define batch_size
        batch_size = 1

        dropout_rate = 0.4

        # load the model
        model = ProcessModel(batch_size, 3, True, dropout_rate)

        test_demo = states_true[:, 40, :,:]
        test_demo = tf.reshape(test_demo, [states_true.shape[0], 1, 1, 3])
        dummy = model(test_demo[0], True)
        model.load_weights('tmp.h5')
        model.summary()

        '''
        run a test demo and save the state of the test demo
        '''
        # data = {}
        # data_save = []

        # for t in range (states_true.shape[0]):
        #     out = model(test_demo[t], True)
        #     state_out = np.array(out)

        #     data_save.append(state_out)
        # data['state'] = data_save

        # with open('./output/transition.pkl', 'wb') as f:
        #     pickle.dump(data, f)

        data = {}

        state_save = []

        for i in range (30):
            data_save = []
            for t in range (math.floor(states_true.shape[0]/4)):
                out = model(test_demo[t], True)
                state_out = np.array(out)

                data_save.append(state_out)
            for t in range (states_true.shape[0] - math.floor(states_true.shape[0]/4)):
                if t == 0:
                    out = model(test_demo[t+math.floor(states_true.shape[0]/4)], True)
                    state_out = np.array(out)
                else:
                    out = model(out, True)
                    state_out = np.array(out)
                data_save.append(state_out)
            state_save.append(data_save)

        data['state'] = state_save


        with open('bio_transition.pkl', 'wb') as f:
            pickle.dump(data, f)




'''
load loss functions
'''
get_loss = getloss()

'''
load data for training
'''
raw_train = get_joint_data('MN02')
states_true, observations = reformat_train_data(raw_train)
states_true, states_true_add1 = transition_dataloader(states_true)
# print(states_true.shape)
# print(observations.shape)

def main():

    # training = True
    # run_filter(training)

    training = False
    run_filter(training)

if __name__ == "__main__":
    main()