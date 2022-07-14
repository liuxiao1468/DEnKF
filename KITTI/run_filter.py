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
import csv
import cv2

import diff_enKF
from dataloader import DataLoader
DataLoader = DataLoader()

'''
define the training loop
'''
def run_filter(mode):

    tf.keras.backend.clear_session()
    dim_x = 2
    if mode == True:
        # define batch_size
        batch_size = 32

        # define number of ensemble
        num_ensemble = 32

        # define dropout rate
        dropout_rate = 0.1

        # load the model
        model = diff_enKF.enKFMLP(batch_size, num_ensemble, dropout_rate)

        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

        epoch = 100
        for k in range (epoch):
            print('end-to-end wholemodel')
            print("========================================= working on epoch %d =========================================: " % (k))
            # 
            steps = int(10000/batch_size)
            path_1 = './dataset/track_dataset_01.pkl'
            path_2 = './dataset/track_dataset_02.pkl'
            path_3 = './dataset/track_dataset_03.pkl'
            for step in range(steps):
                gt_pre, gt_now, raw_sensor_1, raw_sensor_2 = DataLoader.load_training_data(path_1, path_2, path_3, batch_size, name[index])
                raw_sensor = (raw_sensor_1, raw_sensor_2)
                with tf.GradientTape(persistent=True) as tape:
                    start = time.time()
                    states = DataLoader.format_state(gt_pre, batch_size, num_ensemble, dim_x)
                    out = model(raw_sensor,states)
                    state_h = out[1]
                    state_p = out[2]
                    y = out[3]
                    m = out[5]
                    loss_1 = get_loss._mse(gt_now - state_p)
                    loss_2 = get_loss._mse(gt_now - y)
                    loss_3 = get_loss._mse(gt_now - m)
                    loss = get_loss._mse(gt_now - state_h)
                    end = time.time()
                    if step %50 ==0:
                        print("Training loss at step %d: %.4f (took %.3f seconds) " %
                              (step, float(loss), float(end-start)))
                        print(state_p[0])
                        print(m[0])
                        print(y[0])
                        print(state_h[0])
                        print(gt_now[0])
                        print('---')
                grads = tape.gradient(loss, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))

                grads = tape.gradient(loss_1, model.layers[0].trainable_weights)
                optimizer.apply_gradients(zip(grads, model.layers[0].trainable_weights))

                grads = tape.gradient(loss_2, model.layers[3].trainable_weights)
                optimizer.apply_gradients(zip(grads, model.layers[3].trainable_weights))

                grads = tape.gradient(loss_3, model.layers[1].trainable_weights)
                optimizer.apply_gradients(zip(grads, model.layers[1].trainable_weights))

            if (k+1) % epoch == 0:
                model.save_weights('./models/DEnKF_'+version+'_'+name[index]+str(epoch).zfill(3)+'.h5')
                print('model is saved at this epoch')
            if (k+1) % 20 ==0:
                model.save_weights('./models/DEnKF_'+version+'_'+name[index]+str(k).zfill(3)+'.h5')
                print('model is saved at this epoch')

                # define batch_size
                test_batch_size = 1

                test_num_ensemble = 32

                test_dropout_rate = 0.1

                # load the model
                model_test = diff_enKF.enKFMLP(test_batch_size, test_num_ensemble, test_dropout_rate)

                path_2 = './dataset/track_dataset_02.pkl'

                test_gt_pre, test_gt_now, test_raw_sensor_1, test_raw_sensor_2 = DataLoader.load_testing_data(path_2, tracker_id[index])

                # load init state
                inputs = (test_raw_sensor_1[0], test_raw_sensor_2[0])
                init_states = DataLoader.format_init_state(test_gt_pre[0], test_batch_size, test_num_ensemble, dim_x)

                dummy = model_test(inputs, init_states)
                model_test.load_weights('./models/DEnKF_'+version+'_'+name[index]+str(k).zfill(3)+'.h5')
                for layer in model_test.layers:
                    layer.trainable = False
                model_test.summary()

                '''
                run a test demo and save the state of the test demo
                '''
                data = {}
                data_save = []
                emsemble_save = []
                gt_save = []
                transition_save = []
                observation_save = []

                for t in range (test_gt_now.shape[0]):
                    if t == 0:
                        states = init_states
                    test_raw_sensor = (test_raw_sensor_1[t], test_raw_sensor_2[t])
                    out = model_test(test_raw_sensor, states)
                    if t%3 == 0:
                        print('---')
                        print(out[1])
                        print(test_gt_now[t])
                    states = (out[0], out[1])
                    state_out = np.array(out[1])
                    gt_out = np.array(test_gt_now[t])
                    ensemble = np.array(tf.reshape(out[0], [test_num_ensemble, dim_x]))
                    transition_out = np.array(out[2])
                    observation_out = np.array(out[3])
                    data_save.append(state_out)
                    emsemble_save.append(ensemble)
                    gt_save.append(gt_out)
                    observation_save.append(observation_out)
                    transition_save.append(transition_out)
                data['state'] = data_save
                data['ensemble'] = emsemble_save
                data['gt'] = gt_save
                data['observation'] = observation_save
                data['transition'] = transition_save

                with open('./output/DEnKF_'+version+'_'+ name[index]+str(k).zfill(3)+'.pkl', 'wb') as f:
                    pickle.dump(data, f)

    else:
        k_list = [99]
        for k in k_list:
            # define batch_size
            test_batch_size = 1

            test_num_ensemble = 32

            test_dropout_rate = 0.1

            # load the model
            model_test = diff_enKF.enKFMLP(test_batch_size, test_num_ensemble, test_dropout_rate)

            path_2 = './dataset/track_dataset_02.pkl'

            test_gt_pre, test_gt_now, test_raw_sensor_1, test_raw_sensor_2 = DataLoader.load_testing_data(path_2, tracker_id[index])

            # load init state
            inputs = (test_raw_sensor_1[0], test_raw_sensor_2[0])
            init_states = DataLoader.format_init_state(test_gt_pre[0], test_batch_size, test_num_ensemble, dim_x)

            dummy = model_test(inputs, init_states)
            model_test.load_weights('./models/DEnKF_'+version+'_'+name[index]+str(k).zfill(3)+'.h5')
            for layer in model_test.layers:
                layer.trainable = False
            model_test.summary()

            transition_model = diff_enKF.bayesiantransition(test_batch_size, test_num_ensemble, test_dropout_rate)
            dummy = transition_model(init_states)
            transition_model.set_weights(model_test.layers[0].get_weights())
            transition_model.summary()

            '''
            run a test demo and save the state of the test demo
            '''
            data = {}
            data_save = []
            emsemble_save = []
            gt_save = []
            transition_save = []
            observation_save = []

            for t in range (test_gt_now.shape[0]):
                if t == 0:
                    states = init_states
                test_raw_sensor = (test_raw_sensor_1[t], test_raw_sensor_2[t])
                out = model_test(test_raw_sensor, states)
                if t%3 == 0:
                    print('---')
                    print(out[1])
                    print(test_gt_now[t])
                states = (out[0], out[1])
                state_out = np.array(out[1])
                gt_out = np.array(test_gt_now[t])
                ensemble = np.array(tf.reshape(out[0], [test_num_ensemble, dim_x]))
                transition_out = np.array(out[2])
                observation_out = np.array(out[3])
                data_save.append(state_out)
                emsemble_save.append(ensemble)
                gt_save.append(gt_out)
                observation_save.append(observation_out)
                transition_save.append(transition_out)
            data['state'] = data_save
            data['ensemble'] = emsemble_save
            data['gt'] = gt_save
            data['observation'] = observation_save
            data['transition'] = transition_save

            with open('./output/new_DEnKF_'+version+'_'+ name[index]+str(k).zfill(3)+'.pkl', 'wb') as f:
                pickle.dump(data, f)

            '''
            run a test demo and save the state of the test demo
            '''
            data = {}
            data_save = []
            emsemble_save = []
            gt_save = []
            transition_save = []
            observation_save = []
            for t in range (test_gt_now.shape[0]):
                if t == 0:
                    states = init_states
                test_raw_sensor = (test_raw_sensor_1[t], test_raw_sensor_2[t])
                draw = random.uniform(0, 1)
                if draw >= 0.3:
                    out = model_test(test_raw_sensor, states)
                else:
                    out = transition_model(states)
                if t%3 == 0:
                    print('---')
                    print(out[1])
                    # print(out[3])
                    print(test_gt_now[t])
                states = (out[0], out[1])
                state_out = np.array(out[1])
                gt_out = np.array(test_gt_now[t])
                ensemble = np.array(tf.reshape(out[0], [test_num_ensemble, dim_x]))
                # transition_out = np.array(out[2])
                # observation_out = np.array(out[3])
                data_save.append(state_out)
                emsemble_save.append(ensemble)
                gt_save.append(gt_out)
                # observation_save.append(observation_out)
                # transition_save.append(transition_out)
            data['state'] = data_save
            data['ensemble'] = emsemble_save
            data['gt'] = gt_save
            # data['observation'] = observation_save
            # data['transition'] = transition_save

            with open('./output/new_DEnKF_transition'+version+'_'+ name[index]+str(k).zfill(3)+'.pkl', 'wb') as f:
                pickle.dump(data, f)
        
'''
load loss functions
'''
get_loss = diff_enKF.getloss()

'''
load data for training
'''
global name 
# name = ['local_track']
name = ['checkstand', 'checkstandleft', 'coolerleft', 'entranceleft', 'sideright', 'smoothiebar']

global tracker_id
# tracker_id = ['checkstand-6', 'checkstandleft-6', 'coolerleft-4', 'entranceleft-3', 'sideright-3', 'smoothiebar-3']
tracker_id = ['checkstand-9', 'checkstandleft-7', 'coolerleft-7', 'entranceleft-12', 'sideright-13', 'smoothiebar-6']
global index
index = 1

global version
version = 'v1.0'
old_version = version


# path = './dataset/track_dataset.pkl'
# states_pre_save, states_gt_save, observation_save_1, observation_save_2 = DataLoader.load_training_data(path, 64)
# print(states_pre_save.shape)
# print('---')
# print(states_gt_save.shape)
# print('---')
# print(observation_save_1.shape)

def main():

    training = True
    run_filter(training)
    # global index

    # for index in range (6):
    # training = False
    # run_filter(training)

if __name__ == "__main__":
    main()