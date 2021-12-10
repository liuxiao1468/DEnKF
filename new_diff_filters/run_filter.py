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

import enKF_module
tf.compat.v1.disable_eager_execution()

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
            observe = [traj['sensors'][i][0][j]*s, traj['sensors'][i][1][j]*s]
            observation.append(observe)
            angles = traj['xTrue'][i][2]
            xTrue = [traj['xTrue'][i][0]*s, traj['xTrue'][i][1]*s, np.cos(angles), np.sin(angles), traj['xTrue'][i][3]]
            state.append(xTrue)
        observations.append(observation)
        states_true.append(state)
    observations = np.array(observations)
    observations = tf.reshape(observations, [len(traj['xTrue']), num_sensors, 1, 2])
    states_true = np.array(states_true)
    states_true = tf.reshape(states_true, [len(traj['xTrue']), num_sensors, 1, 5])
    return observations, states_true

def data_loader_teacher(states_true):
    gt_pre = states_true[0:-1, :, :, :]
    gt_now = states_true[1:,:, :, :]
    return gt_pre, gt_now

def load_train_teacher(batch_size, observations, gt_pre, gt_now):
    '''
    data preprocessing steps
    '''
    select = random.sample(range(0, 90), batch_size)
    raw_sensor = []
    gt_pre_ = []
    gt_now_ = []
    for idx in select:
        raw_sensor.append(observations[:, idx, :,:])
        gt_pre_.append(gt_pre[:, idx, :,:])
        gt_now_.append(gt_now[:, idx, :,:])

    raw_sensor = tf.convert_to_tensor(raw_sensor, dtype=tf.float32)
    raw_sensor = tf.reshape(raw_sensor, [observations.shape[0], batch_size, 1, 2])
    gt_pre_ = tf.convert_to_tensor(gt_pre_, dtype=tf.float32)
    gt_pre_ = tf.reshape(gt_pre_, [gt_pre.shape[0], batch_size, 1, 5])
    gt_now_ = tf.convert_to_tensor(gt_now_, dtype=tf.float32)
    gt_now_ = tf.reshape(gt_pre_, [gt_now.shape[0], batch_size, 1, 5])
    return raw_sensor, gt_pre_, gt_now_

def format_state(state, batch_size, num_ensemble):
    for n in range (batch_size):
        if n == 0:
            ensemble = tf.reshape(tf.stack([state[n]] * num_ensemble), [1, num_ensemble, 5])
        else:
            tmp = tf.reshape(tf.stack([state[n]] * num_ensemble), [1, num_ensemble, 5])
            ensemble = tf.concat([ensemble, tmp], 0)
    state_input = (ensemble, state)
    return state_input


'''
define the training loop
'''
def run_filter(mode):
    tf.keras.backend.clear_session()
    if mode == True:
        # define batch_sizepython
        batch_size = 64

        # define number of ensemble
        num_ensemble = 32

        # define dropout rate
        dropout_rate = 0

        # load the model
        model = enKF_module.ensembleKF(batch_size, num_ensemble, dropout_rate)

        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        epoch = 10

        '''
        teacher enforcing
        '''
        print('-----------------')
        print('teacher enforcing ')
        print('-----------------')
        raw_sensor, gt_pre_, gt_now_ = load_train_teacher(batch_size, observations, gt_pre, gt_now)
        for k in range (epoch):

            print("========================================= working on epoch %d =========================================: " % (k))

            for i in range(gt_now_.shape[0]):

                start = time.time()

                with tf.GradientTape() as tape:
                    states = format_state(gt_pre_[i], batch_size, num_ensemble)
                    out = model(raw_sensor[i], states)
                    state_h = out[1]
                    loss = get_loss._mse( gt_now_[i] - state_h)* 0.1
                    grads = tape.gradient(loss, model.trainable_weights)
                    optimizer.apply_gradients(zip(grads, model.trainable_weights))
                    end = time.time()
                    
                # Log every 50 batches.
                if i % 100 == 0:
                    print("Training loss at step %d: %.4f (took %.3f seconds) " %
                          (i, float(loss), float(end-start)))
                    print(state_h[0])
                    print(gt_now_[i][0])
                    print('---')
        '''
        student enforcing
        '''
        print('-----------------')
        print('student enforcing ')
        print('-----------------')

        raw_sensor, gt_pre_, gt_now_ = load_train_teacher(batch_size, observations, gt_pre, gt_now)
        for pred_steps in range(5, gt_now_.shape[0], 10):
            print("========================================= working on steps with %d =========================================: " % (pred_steps))
            for i in range(gt_now_.shape[0]-pred_steps):
                start = time.time()
                with tf.GradientTape() as tape:
                    for step in range (pred_steps):
                        if step == 0:
                            states = format_state(gt_pre_[i+step], batch_size, num_ensemble)
                            out = model(raw_sensor[i+step], states)
                            state_h = out[1]
                            loss = get_loss._mse( gt_now_[i+step] - state_h)* 0.1
                        else:
                            out = model(raw_sensor[i+step], states)
                            state_h = out[1]
                            loss = loss + get_loss._mse( gt_now_[i+step] - state_h)*0.1
                        states = out
                    grads = tape.gradient(loss, model.trainable_weights)
                    optimizer.apply_gradients(zip(grads, model.trainable_weights))
                    end = time.time()
                # Log every 50 batches.
                if i % 100 == 0: 
                    print("Training loss at step %d: %.4f (took %.3f seconds) " %
                          (i, float(loss), float(end-start)))
                    print(state_h[0])
                    print(gt_now_[i+pred_steps][0])
                    print('---')

        model.save_weights('./models/enkF_'+version+'_'+name[index]+str(49).zfill(3)+'.h5')
        print('model is saved at this epoch')
        print('--- start testing ---')
        for j in range(states_true.shape[0]):
            if j ==0:
                # init state
                X = tf.convert_to_tensor(np.array([0,0,1,0,1]), dtype=tf.float32)
                m_state = tf.stack([X] * batch_size)
                state_old = tf.stack([X] * (num_ensemble * batch_size))
                states = (state_old, m_state)

            out = model(raw_sensor[j], states)
            state_h = out[1]
            loss = get_loss._mse( gt_now_[j] - state_h)* 0.1

            # Log every 50 batches.
            if j % 100 == 0:
                print("Training loss at step %d: %.4f " %
                      (j, float(loss)))
                print(state_h[0])
                print(gt_now_[j][0])
                print('---')

            # update state for next iteration
            states = out

    else:
        # define batch_size
        batch_size = 1

        num_ensemble = 32

        dropout_rate = 0.4

        # load the model
        model = enKF_module.ensembleKF(batch_size, num_ensemble, dropout_rate)

        test_demo = observations[:, 0, :,:]
        test_demo = tf.reshape(test_demo, [observations.shape[0], 1, 1, 2])

        gt = states_true[:, 0, :,:]
        gt = tf.reshape(gt, [states_true.shape[0], batch_size, 1, 5])


        # init state
        X = tf.convert_to_tensor(np.array([0,0,1,0,1]), dtype=tf.float32)
        m_state = tf.stack([X] * batch_size)
        state_old = tf.stack([X] * (num_ensemble * batch_size))
        states = (state_old, m_state)

        dummy = model(test_demo[0], states)
        model.load_weights('./models/enkF_'+version+'_'+name[index]+str(49).zfill(3)+'.h5')
        model.summary()

        '''
        run a test demo and save the state of the test demo
        '''
        data = {}
        data_save = []
        emsemble_save = []

        for t in range (states_true.shape[0]):
            out = model(test_demo[t], states)
            state_out = np.array(out[1])
            print('----------')
            print(state_out)
            print(gt[t])
            ensemble = np.array(tf.reshape(out[0], [num_ensemble, 5]))
            data_save.append(state_out)
            emsemble_save.append(ensemble)
            states = out
        data['state'] = data_save
        data['ensemble'] = emsemble_save

        with open('./output/'+version+'_'+ name[index] +'_02.pkl', 'wb') as f:
            pickle.dump(data, f)


'''
load loss functions
'''
get_loss = enKF_module.getloss()

'''
load data for training
'''
global name 
name = ['constant', 'exp']
global index
index = 1
observations, states_true = data_loader_function('./dataset/100_demos_'+name[index]+'.pkl')
gt_pre, gt_now = data_loader_teacher(states_true)

global version
version = 'v2.1'

def main():

    training = True
    run_filter(training)

    training = False
    run_filter(training)

if __name__ == "__main__":
    main()