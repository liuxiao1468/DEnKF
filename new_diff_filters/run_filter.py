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
        dropout_rate = 0.4

        # load the model
        model = diff_enKF.RNNmodel(batch_size, num_ensemble, dropout_rate)

        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        epoch = 50

        pred_steps = 1

        for k in range (epoch):
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

            print("========================================= working on epoch %d =========================================: " % (k))

            for i in range(states_true.shape[0]-pred_steps):

                start = time.time()

                with tf.GradientTape() as tape:
                    for step in range (pred_steps):
                        if step == 0:
                            out = model(raw_sensor[i+step])
                            state_h = out[0]
                            loss = get_loss._mse( gt[i+step] - state_h)
                        else:
                            out = model(raw_sensor[i+step])
                            state_h = out[0]
                            loss = loss + get_loss._mse( gt[i+step] - state_h)

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
                    # print(out[3][0])
                    # print(out[4][0])
                    print('---')
            if (k+1) % 50 == 0:
                model.save_weights('./models/ensemble_model_v3.4_weights_'+name[index]+str(k).zfill(3)+'.h5')
                print('model is saved at this epoch')
    else:
        # define batch_size
        batch_size = 1

        num_ensemble = 32

        dropout_rate = 0.4

        # load the model
        model = diff_enKF.RNNmodel(batch_size, num_ensemble, dropout_rate)

        test_demo = observations[:, 98, :,:]
        test_demo = tf.reshape(test_demo, [observations.shape[0], 1, 1, 2])
        dummy = model(test_demo[0])
        model.load_weights('./models/ensemble_model_v3.4_weights_'+name[index]+str(49).zfill(3)+'.h5')
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
            ensemble = np.array(tf.reshape(out[1], [num_ensemble, 5]))
            # print('----------')
            # print(ensemble)
            data_save.append(state_out)
            emsemble_save.append(ensemble)
        data['state'] = data_save
        data['ensemble'] = emsemble_save

        with open('./output/ensemble_v3.4_norm_'+ name[index] +'_02.pkl', 'wb') as f:
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
global count
count = 0

def main():

    training = True
    run_filter(training)

    training = False
    run_filter(training)

if __name__ == "__main__":
    main()