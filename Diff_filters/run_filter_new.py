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

import diff_enKF

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

def load_train(batch_size, observations, states_true):
    '''
    data preprocessing steps
    '''
    select = random.sample(range(0, 90), batch_size)
    raw_sensor = []
    gt = []
    for idx in select:
        raw_sensor.append(observations[:, idx, :,:])
        gt.append(states_true[:, idx, :,:])

    raw_sensor = tf.convert_to_tensor(raw_sensor, dtype=tf.float32)
    raw_sensor = tf.reshape(raw_sensor, [observations.shape[0], batch_size, 1, 2])
    gt = tf.convert_to_tensor(gt, dtype=tf.float32)
    gt = tf.reshape(gt, [states_true.shape[0], batch_size, 1, 5])
    return raw_sensor, gt

def load_test(batch_size, observations, states_true):
    '''
    data preprocessing steps
    '''
    raw_sensor = []
    gt = []
    idx = 98
    raw_sensor.append(observations[:, idx, :,:])
    gt.append(states_true[:, idx, :,:])

    raw_sensor = tf.convert_to_tensor(raw_sensor, dtype=tf.float32)
    raw_sensor = tf.reshape(raw_sensor, [observations.shape[0], batch_size, 1, 2])
    gt = tf.convert_to_tensor(gt, dtype=tf.float32)
    gt = tf.reshape(gt, [states_true.shape[0], batch_size, 1, 5])
    return raw_sensor, gt

def mask_observation(observations, ratio):
    length = observations.shape[0]
    observations = tf.Variable(observations)
    select = random.sample(range(0, length), int(length* ratio))
    for idx in select:
        observations[idx, :, :, :].assign(observations[idx, :, :, :]*0.0 + 0.0)
    masked_observation = tf.cast(observations, dtype=tf.float32)
    return masked_observation

def format_state(state, batch_size, num_ensemble):
    dim_x = 5
    diag = np.ones((dim_x)).astype(np.float32) * 0.01
    diag = diag.astype(np.float32)
    mean = np.zeros((dim_x))
    mean = tf.convert_to_tensor(mean, dtype=tf.float32)
    mean = tf.stack([mean] * batch_size)
    nd = tfp.distributions.MultivariateNormalDiag(loc=mean, scale_diag=diag)
    Q = tf.reshape(nd.sample(num_ensemble), [batch_size, num_ensemble, dim_x])
    for n in range (batch_size):
        if n == 0:
            ensemble = tf.reshape(tf.stack([state[n]] * num_ensemble), [1, num_ensemble, 5])
        else:
            tmp = tf.reshape(tf.stack([state[n]] * num_ensemble), [1, num_ensemble, 5])
            ensemble = tf.concat([ensemble, tmp], 0)
    ensemble = ensemble + Q
    state_input = (ensemble, state)
    return state_input

'''
define the training loop
'''
def run_filter(mode):
    epoch = 100
    tf.keras.backend.clear_session()
    if mode == True:
        # define batch_size
        batch_size = 64

        # define number of ensemble
        num_ensemble = 32

        # define dropout rate
        dropout_rate = 0.1

        # load the model
        model_p = diff_enKF.TransitionRNN(batch_size, num_ensemble, dropout_rate)
        model_u = diff_enKF.enKFUpdate(batch_size, num_ensemble, dropout_rate)

        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

        '''
        train with observations
        '''
        for k in range (epoch):
            '''
            data preprocessing steps
            '''
            raw_sensor, gt = load_train(batch_size, observations, states_true)

            print("========================================= working on epoch %d =========================================: " % (k))

            for i in range(states_true.shape[0]):

                start = time.time()

                with tf.GradientTape(persistent=True) as tape:
                    inputs = raw_sensor[i]
                    if i == 0:
                        # init state
                        init_states = format_state(gt[i], batch_size, num_ensemble)
                        states = init_states
                    out = model_p(states)
                    state_h = out[1]
                    loss_1 = get_loss._mse( gt[i] - state_h)*0.2
                    out = model_u(inputs, out)
                    state_h = out[1]
                    loss_2 = get_loss._mse( gt[i] - state_h)*0.2
                    states = out

                grads = tape.gradient(loss_1, model_p.trainable_weights)
                optimizer.apply_gradients(zip(grads, model_p.trainable_weights))
                grads = tape.gradient(loss_2, model_u.trainable_weights)
                optimizer.apply_gradients(zip(grads, model_u.trainable_weights))
                end = time.time()

                # Log every 50 batches.
                if i % 100 == 0:
                    print("Training loss1 and loss2 at step %d: %.4f and %.4f (took %.3f seconds) " %
                          (i, float(loss_1), float(loss_2), float(end-start)))
                    print(state_h[0])
                    print(gt[i][0][0])
                    print('---')
            if (k+1) % epoch == 0:
                model_p.save_weights('./models/transition_'+version+'_'+name[index]+str(epoch).zfill(3)+'.h5')
                model_u.save_weights('./models/enKF_update_'+version+'_'+name[index]+str(epoch).zfill(3)+'.h5')
                print('model is saved at this epoch')
        # '''
        # train with missing observations
        # '''
        # pred_steps = 5
        # epoch = 30
        # for k in range (epoch):
        #     '''
        #     data preprocessing steps
        #     '''
        #     raw_sensor, gt = load_train(batch_size, observations, states_true)
        #     print(raw_sensor.shape)
        #     raw_sensor = mask_observation(observations, 0.5)
        #     print(raw_sensor.shape)

        #     print("========================================= working on epoch %d =========================================: " % (k))

        #     for i in range(states_true.shape[0]-pred_steps):

        #         start = time.time()

        #         with tf.GradientTape() as tape:
        #             for step in range (pred_steps):
        #                 if step == 0:
        #                     out = model(raw_sensor[i+step])
        #                     state_h = out[0]
        #                     loss = get_loss._mse( gt[i+step] - state_h)
        #                 else:
        #                     out = model(raw_sensor[i+step])
        #                     state_h = out[0]
        #                     loss = loss + get_loss._mse( gt[i+step] - state_h)

        #         grads = tape.gradient(loss, model.trainable_weights)
        #         optimizer.apply_gradients(zip(grads, model.trainable_weights))
        #         end = time.time()

        #         # Log every 50 batches.
        #         if i % 100 == 0:
        #             print("Training loss at step %d: %.4f (took %.3f seconds) " %
        #                   (i, float(loss), float(end-start)))
        #             print(state_h[0])
        #             print(gt[i][0])
        #             print('---')
        #     if (k+1) % epoch == 0:
        #         model.save_weights('./models/enkF_'+version+'_'+name[index]+str(k).zfill(3)+'.h5')
        #         print('model is saved at this epoch')
    else:
        # define batch_size
        batch_size = 1

        num_ensemble = 32

        dropout_rate = 0.1

        # load the model
        model_p = diff_enKF.TransitionRNN(batch_size, num_ensemble, dropout_rate)
        model_u = diff_enKF.enKFUpdate(batch_size, num_ensemble, dropout_rate)

        raw_sensor, gt = load_train(batch_size, observations, states_true)

        # load init state 
        inputs = raw_sensor[0]
        init_states = format_state(gt[i], batch_size, num_ensemble)
        states = init_states
        out = model_p(states)
        out = model_u(inputs, out)
        model_p.load_weights('./models/transition_'+version+'_'+name[index]+str(epoch).zfill(3)+'.h5')
        model_u.load_weights('./models/enKF_update_'+version+'_'+name[index]+str(epoch).zfill(3)+'.h5')
        model_p.summary()
        model_u.summary()

        '''
        run a test demo and save the state of the test demo
        '''
        data = {}
        data_save = []
        emsemble_save = []

        for t in range (gt.shape[0]):
            inputs = raw_sensor[t]
            if t ==0:
                init_states = format_state(gt[t], batch_size, num_ensemble)
                states = init_states
            out = model_p(states)
            out = model_u(inputs, out)
            states = out

            state_h = out[1]
            loss_2 = get_loss._mse( gt[i][0] - state_h)*0.2
            if t%100 == 0 :
                print('---')
                print(loss_2)
                print(out[1])
                print(gt[t][0][0])

            state_out = np.array(out[1])
            ensemble = np.array(tf.reshape(out[0], [num_ensemble, 5]))
            data_save.append(state_out)
            emsemble_save.append(ensemble)

        data['state'] = data_save
        data['ensemble'] = emsemble_save

        with open('./output/'+version+'_'+ name[index] +'.pkl', 'wb') as f:
            pickle.dump(data, f)

'''
load loss functions
'''
get_loss = diff_enKF.getloss()

'''
load data for training
'''
global name 
name = ['constant', 'exp']
global index
index = 1
observations, states_true = data_loader_function('./dataset/100_demos_'+name[index]+'.pkl')

global version
version = 'v4.0'


def main():

    training = True
    run_filter(training)

    training = False
    run_filter(training)

if __name__ == "__main__":
    main()