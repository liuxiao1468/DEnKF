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
    dim_x = 3
    if mode == True:
        # define batch_size
        batch_size = 32

        # define number of ensemble
        num_ensemble = 32

        # load the model
        model = diff_enKF.enKFMLP(batch_size, num_ensemble)
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

        epoch = 50
        counter = 0
        for k in range (epoch):
            print('end-to-end wholemodel')
            print("========================================= working on epoch %d =========================================: " % (k))
            # 
            steps = int(4000/batch_size)
            for step in range(steps):
                counter = counter + 1
                gt_pre, gt_now, raw_sensor = DataLoader.train_data('./dataset/train_set.pkl',batch_size)
                with tf.GradientTape(persistent=True) as tape:
                    start = time.time()
                    states = DataLoader.format_state(gt_pre, batch_size, num_ensemble, dim_x)
                    out = model(raw_sensor,states)
                    state_h = out[1]
                    state_d = out[2]
                    y = out[3]
                    m = out[5]
                    loss_1 = get_loss._mse(gt_now - state_d) # state transition
                    loss_2 = get_loss._mse(gt_now - y) # sensor model
                    loss_3 = get_loss._mse(gt_now - m) # observation model
                    loss = get_loss._mse(gt_now - state_h) # end-to-end state
                    end = time.time()
                    if step % 20 ==0:
                        print("Training loss at step %d: %.4f (took %.3f seconds) " %
                              (step, float(loss), float(end-start)))
                        print(loss_1)
                        print(loss_2)
                        print(loss_3)
                        with train_summary_writer.as_default():
                            tf.summary.scalar('sensor_loss', loss_2, step=counter)
                            tf.summary.scalar('total_loss', loss, step=counter)
                            tf.summary.scalar('observation_loss', loss_3, step=counter)
                            tf.summary.scalar('process_loss', loss_1, step=counter)
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
            if (k+1) % 10 ==0:
                model.save_weights('./models/DEnKF_'+version+'_'+name[index]+str(k).zfill(3)+'.h5')
                print('model is saved at this epoch')

                # define batch_size
                test_batch_size = 1

                test_num_ensemble = 32

                # load the model
                model_test = diff_enKF.enKFMLP(test_batch_size, test_num_ensemble)
                test_gt_pre, test_gt_now, test_raw_sensor = DataLoader.test_data('./dataset/test_set_1.pkl')


                # load init state
                inputs = test_raw_sensor[0]
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

                N = test_gt_now.shape[0]

                for t in range (N):
                    if t == 0:
                        states = init_states
                    raw_sensor = test_raw_sensor[t]
                    out = model_test(raw_sensor, states)
                    if t%10 == 0:
                        print('---')
                        print(out[1])
                        print(test_gt_now)
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

            # load the model
            model_test = diff_enKF.enKFMLP(test_batch_size, test_num_ensemble)
            test_gt_pre, test_gt_now, test_raw_sensor = DataLoader.load_test_data_All('dataset_UR5_yifan_test.csv', test_batch_size, norm=True)

            # load init state
            inputs = test_raw_sensor[0]
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

            N = test_gt_now.shape[0]

            for t in range (N):
                if t == 0:
                    states = init_states
                raw_sensor = test_raw_sensor[t]
                out = model_test(raw_sensor, states)
                if t%10 == 0:
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

            with open('./output/DEnKF_'+version+'_'+ name[index]+str(k).zfill(3)+'test.pkl', 'wb') as f:
                pickle.dump(data, f)

            # '''
            # run a test demo and save the state of the test demo
            # '''
            # data = {}
            # data_save = []
            # emsemble_save = []
            # gt_save = []
            # transition_save = []
            # observation_save = []
            # for t in range (test_gt_now.shape[0]):
            #     if t == 0:
            #         states = init_states
            #     test_raw_sensor = (test_raw_sensor_1[t], test_raw_sensor_2[t])
            #     draw = random.uniform(0, 1)
            #     if draw >= 0.3:
            #         out = model_test(test_raw_sensor, states)
            #     else:
            #         out = transition_model(states)
            #     if t%3 == 0:
            #         print('---')
            #         print(out[1])
            #         # print(out[3])
            #         print(test_gt_now[t])
            #     states = (out[0], out[1])
            #     state_out = np.array(out[1])
            #     gt_out = np.array(test_gt_now[t])
            #     ensemble = np.array(tf.reshape(out[0], [test_num_ensemble, dim_x]))
            #     # transition_out = np.array(out[2])
            #     # observation_out = np.array(out[3])
            #     data_save.append(state_out)
            #     emsemble_save.append(ensemble)
            #     gt_save.append(gt_out)
            #     # observation_save.append(observation_out)
            #     # transition_save.append(transition_out)
            # data['state'] = data_save
            # data['ensemble'] = emsemble_save
            # data['gt'] = gt_save
            # # data['observation'] = observation_save
            # # data['transition'] = transition_save

            # with open('./output/new_DEnKF_transition'+version+'_'+ name[index]+str(k).zfill(3)+'.pkl', 'wb') as f:
            #     pickle.dump(data, f)
        
'''
load loss functions
'''
get_loss = diff_enKF.getloss()

'''
load data for training
'''
global name 
name = ['kaiteki']

global index
index = 0

global version
version = 'v1.0'
old_version = version

os.system('rm -rf /tf/experiments/loss/'+version)

train_log_dir = "/tf/experiments/loss/"+version
train_summary_writer = tf.summary.create_file_writer(train_log_dir)

def main():
    training = True
    run_filter(training)

    # training = False
    # run_filter(training)

if __name__ == "__main__":
    main()