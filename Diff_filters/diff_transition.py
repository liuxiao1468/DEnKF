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

'''
data loader for training the toy example
observation = [timestep, batch_size, 1, dim_z] -> input data
states_true = [timestep, batch_size, 1, dim_x] -> ground truth
'''
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
            observe = [(traj['sensors'][i][0][j] +0)*s, traj['sensors'][i][1][j]*s]
            observation.append(observe)
            angles = traj['xTrue'][i][2]
            xTrue = [(traj['xTrue'][i][0] + 0)*s, traj['xTrue'][i][1]*s, np.cos(angles), np.sin(angles), traj['xTrue'][i][3]]
            state.append(xTrue)
        observations.append(observation)
        states_true.append(state)
    observations = np.array(observations)
    observations = tf.reshape(observations, [len(traj['xTrue']), num_sensors, 1, 2])
    states_true = np.array(states_true)
    states_true = tf.reshape(states_true, [len(traj['xTrue']), num_sensors, 1, 5])
    return observations, states_true


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


get_loss = getloss()


'''
define the training loop
'''
def run_filter(mode):
    if mode == True:
        # define batch_sizepython
        batch_size = 64

        # define dropout rate
        dropout_rate = 0.4

        process_model = ProcessModel(batch_size, 5, True, dropout_rate)

        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        epoch = 40

        pred_steps = 20

        for k in range (epoch):
            '''
            data preprocessing steps
            '''
            input_state = []
            gt = []


            select = random.sample(range(0, 90), batch_size)
            for idx in select:
                input_state.append(states_true[:, idx, :,:])
                gt.append(states_true_add1[:, idx, :,:])
            gt = tf.convert_to_tensor(gt, dtype=tf.float32)
            gt = tf.reshape(gt, [states_true_add1.shape[0], batch_size, 1, 5])

            input_state = tf.convert_to_tensor(input_state, dtype=tf.float32)
            input_state = tf.reshape(input_state, [states_true.shape[0], batch_size, 1, 5])

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
        model = ProcessModel(batch_size, 5, True, dropout_rate)

        test_demo = states_true[:, 98, :,:]
        test_demo = tf.reshape(test_demo, [states_true.shape[0], 1, 1, 5])
        dummy = model(test_demo[0], True)
        model.load_weights('transition_model.h5')
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


        with open('./output/transition_multi_01.pkl', 'wb') as f:
            pickle.dump(data, f)

'''
load loss functions
'''
get_loss = getloss()

'''
load data for training
'''
global name 
name = ['constant', 'exp']
global index
index = 1
observations, states_true, states_true_add1 = transition_data_loader_function('./dataset/100_demos_'+name[index]+'.pkl')


def main():

    training = True
    run_filter(training)

    training = False
    run_filter(training)


if __name__ == "__main__":
    main()

