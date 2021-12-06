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
# session = InteractiveSession(config=config)

import enKF_module


class enKF:
    def __init__(self, batch_size, num_ensemble, dropout_rate,**kwargs):
        super(enKF, self).__init__(**kwargs)

        # initialization
        self.batch_size = batch_size
        self.num_ensemble = num_ensemble
        self.dim_x = 5
        self.dim_z = 2
        self.jacobian = True
        self.q_diag = np.ones((self.dim_x)).astype(np.float32) * 0.5
        self.q_diag = self.q_diag.astype(np.float32)
        self.r_diag = np.ones((self.dim_z)).astype(np.float32) * 0.5
        self.r_diag = self.r_diag.astype(np.float32)
        self.scale = 1
        self.dropout_rate = dropout_rate

    '''
    initialize the enFK framework for training
    '''
    def init_framework(self):
        '''
        import sub-modules for diff-enKF framework
        '''

        # learned process model
        self.process_model = enKF_module.ProcessModel(self.batch_size, self.num_ensemble, self.dim_x, self.jacobian, self.dropout_rate)

        # learned process noise
        self.process_noise_model = enKF_module.ProcessNoise(self.batch_size, self.num_ensemble, self.dim_x, self.q_diag)

        # learned observation model
        self.observation_model = enKF_module.ObservationModel(self.batch_size, self.num_ensemble, self.dim_x, self.dim_z, self.jacobian)

        # learned observation noise
        self.observation_noise_model = enKF_module.ObservationNoise(self.batch_size, self.num_ensemble, self.dim_z, self.r_diag, self.jacobian)

        # learned sensor model
        self.sensor_model = enKF_module.SensorModel(self.batch_size, self.dim_z)

        # define the initial belief of the filter
        X = np.array([0,0,0,0,1])
        X = tf.convert_to_tensor(X, dtype=tf.float32)
        z = np.array([0,0])
        z = tf.convert_to_tensor(z, dtype=tf.float32)
        ensemble_X = tf.stack([X] * (self.num_ensemble * self.batch_size))
        m_X = tf.stack([X] * self.batch_size)
        m_z = tf.stack([z] * self.batch_size)
        m_z = tf.reshape(m_z, [self.batch_size, 1, 2])

        _ = self.process_model(ensemble_X, True)
        _, _ = self.process_noise_model(m_X, True, True)
        _ = self.observation_model(ensemble_X, True, True)
        _, encoding = self.sensor_model(m_z, True, True)
        _, _ = self.observation_noise_model(encoding, True, True)

        epoch = 20
        # load each sub-module
        self.process_model.load_weights('./models/F_'+version+'_'+name[index]+str(epoch-1).zfill(3)+'.h5')
        self.process_noise_model.load_weights('./models/Q_'+version+'_'+name[index]+str(epoch-1).zfill(3)+'.h5')
        self.observation_model.load_weights('./models/H_'+version+'_'+name[index]+str(epoch-1).zfill(3)+'.h5')
        self.sensor_model.load_weights('./models/S_'+version+'_'+name[index]+str(epoch-1).zfill(3)+'.h5')
        self.observation_noise_model.load_weights('./models/R_'+version+'_'+name[index]+str(epoch-1).zfill(3)+'.h5')


    def run_enKF_filter(self, raw_, m_state, state_old):
        self.utils = enKF_module.utils()
        '''
        prediction step
        '''
        training = True

        # get prediction and noise of next state
        state_pred = self.process_model(state_old, training)
        Q, diag_Q = self.process_noise_model(m_state, training, True)

        state_pred = state_pred + Q

        '''
        update step
        '''
        learn = True
        H_X = self.observation_model(state_pred, training, learn)

        # get the emsemble mean of the observations
        m = tf.reduce_mean(H_X, axis = 1)
        for i in range (self.batch_size):
            if i == 0:
                mean = tf.reshape(tf.stack([m[i]] * self.num_ensemble), [self.num_ensemble, self.dim_z])
            else:
                tmp = tf.reshape(tf.stack([m[i]] * self.num_ensemble), [self.num_ensemble, self.dim_z])
                mean = tf.concat([mean, tmp], 0)
        mean = tf.reshape(mean, [self.batch_size, self.num_ensemble, self.dim_z])
        H_A = H_X - mean
        final_H_A = tf.transpose(H_A, perm=[0,2,1])
        final_H_X = tf.transpose(H_X, perm=[0,2,1])

        # get sensor reading
        z, encoding = self.sensor_model(raw_, training, True)

        # enable each ensemble to have a observation
        z = tf.reshape(z, [self.batch_size, self.dim_z])
        for i in range (self.batch_size):
            if i == 0:
                ensemble_z = tf.reshape(tf.stack([z[i]] * self.num_ensemble), [1, self.num_ensemble, self.dim_z])
            else:
                tmp = tf.reshape(tf.stack([z[i]] * self.num_ensemble), [1, self.num_ensemble, self.dim_z])
                ensemble_z = tf.concat([ensemble_z, tmp], 0)
        ensemble_z = tf.reshape(ensemble_z, [self.batch_size, self.num_ensemble, self.dim_z])

        # get observation noise
        R, diag_R = self.observation_noise_model(encoding, training, True)
        

        # incorporate the measurement with stochastic noise
        r_mean = np.zeros((self.dim_z))
        r_mean = tf.convert_to_tensor(r_mean, dtype=tf.float32)
        r_mean = tf.stack([r_mean] * self.batch_size)
        nd_r = tfp.distributions.MultivariateNormalDiag(loc=r_mean, scale_diag=diag_R)
        epsilon = tf.reshape(nd_r.sample(self.num_ensemble), [self.batch_size, self.num_ensemble, self.dim_z])

        # the measurement y
        y = ensemble_z + epsilon
        y = tf.transpose(y, perm=[0,2,1])

        # calculated innovation matrix s
        innovation = (1/(self.num_ensemble -1)) * tf.matmul(final_H_A,  H_A) + R

        # A matrix
        m_A = tf.reduce_mean(state_pred, axis = 1)
        for i in range (self.batch_size):
            if i == 0:
                mean_A = tf.reshape(tf.stack([m_A[i]] * self.num_ensemble), [1, self.num_ensemble, self.dim_x])
            else:
                tmp = tf.reshape(tf.stack([m_A[i]] * self.num_ensemble), [1, self.num_ensemble, self.dim_x])
                mean_A = tf.concat([mean_A, tmp], 0)
        A = state_pred - mean_A
        A = tf.transpose(A, perm = [0,2,1])

        try:
            innovation_inv = tf.linalg.inv(innovation)
        except:
            innovation = self.utils._make_valid(innovation)
            innovation_inv = tf.linalg.inv(innovation)

        # calculating Kalman gain
        K = (1/(self.num_ensemble -1)) * tf.matmul(tf.matmul(A, H_A), innovation_inv)

        # update state of each ensemble
        y_bar = y - final_H_X
        state_new = state_pred +  tf.transpose(tf.matmul(K, y_bar), perm=[0,2,1])

        # the ensemble state mean
        m_state_new = tf.reduce_mean(state_new, axis = 1)

        # print(diag_Q[0])
        # print(diag_R[0])
        # print(z[0])
        # print(m_state[0])
        # print(m_state_new[0])
        # print('--------')

        return m_state_new, state_new

    '''
    the actual training process
    '''
    def train(self, states_true, observations):
        # define training parameter
        get_loss = enKF_module.getloss()
        optimizer1 = tf.keras.optimizers.Adam(learning_rate=1e-4)
        optimizer2 = tf.keras.optimizers.Adam(learning_rate=1e-4)
        optimizer3 = tf.keras.optimizers.Adam(learning_rate=1e-4)
        optimizer4 = tf.keras.optimizers.Adam(learning_rate=1e-4)
        optimizer5 = tf.keras.optimizers.Adam(learning_rate=1e-4)
        epoch = 20

        for k in range (epoch):
            '''
            process input state and observations
            '''
            select = random.sample(range(0, 90), self.batch_size)
            raw_sensor = []
            gt = []
            # for idx in select:
            #     raw_sensor.append(observations[:, idx, :,:])
            #     gt.append(states_true[:, idx, :,:])

            raw_sensor = observations[:, 98, :,:]
            raw_sensor = tf.cast(raw_sensor, tf.float32)
            gt = states_true[:, 98, :,:]
            gt = tf.cast(gt, tf.float32)


            # raw_sensor = tf.convert_to_tensor(raw_sensor, dtype=tf.float32)
            raw_sensor = tf.reshape(raw_sensor, [observations.shape[0], self.batch_size, 1, 2])
            # gt = tf.convert_to_tensor(gt, dtype=tf.float32)
            gt = tf.reshape(gt, [states_true.shape[0], self.batch_size, 5])

            '''
            start one epoch of training
            '''
            print("========================================= working on epoch %d =========================================: " % (k))
            for i in range(states_true.shape[0]):
                start = time.time()
                with tf.GradientTape(persistent=True) as tape:
                    '''
                    diff enKF process start 
                    '''
                    raw_ = raw_sensor[i]
                    raw_ = tf.reshape(raw_, [self.batch_size, self.dim_z])
                    if i == 0:
                        X = tf.convert_to_tensor(np.array([0,0,0,0,1]), dtype=tf.float32)
                        m_state = tf.stack([X] * self.batch_size)
                        state_old = tf.stack([X] * (self.num_ensemble * self.batch_size))

                    '''
                    run the filter
                    '''
                    m_state_new, state_new = self.run_enKF_filter(raw_, m_state, state_old)

                    loss = get_loss._mse(gt[i] - m_state_new)
                    loss_1 = 0.5* loss
                    loss_2 = 0.5* loss

                    loss_3 = 0.5* loss
                    loss_4 = 0.5* loss

                grads1 = tape.gradient(loss_1, self.process_model.trainable_weights)
                optimizer1.apply_gradients(zip(grads1, self.process_model.trainable_weights))

                grads2 = tape.gradient(loss_3, self.process_noise_model.trainable_weights)
                optimizer2.apply_gradients(zip(grads2, self.process_noise_model.trainable_weights))

                grads3 = tape.gradient(loss_2, self.observation_model.trainable_weights)
                optimizer3.apply_gradients(zip(grads3, self.observation_model.trainable_weights))

                grads4 = tape.gradient(loss_1, self.sensor_model.trainable_weights)
                optimizer4.apply_gradients(zip(grads4, self.sensor_model.trainable_weights))

                grads5 = tape.gradient(loss_4, self.observation_noise_model.trainable_weights)
                optimizer5.apply_gradients(zip(grads5, self.observation_noise_model.trainable_weights))
                end = time.time()

                # update the state and ensembles 
                m_state = m_state_new
                state_old = state_new

                # Log every 50 batches
                if i % 100 == 0:
                    print("Training loss at step %d: %.4f (took %.3f seconds) " %(i, float(loss), float(end-start)))
                    print(m_state_new[0])
                    print(gt[i][0])
                    print('---')

            if (k+1) % epoch == 0:
                self.process_model.save_weights('./models/F_'+version+'_'+name[index]+str(k).zfill(3)+'.h5')
                self.process_noise_model.save_weights('./models/Q_'+version+'_'+name[index]+str(k).zfill(3)+'.h5')
                self.observation_model.save_weights('./models/H_'+version+'_'+name[index]+str(k).zfill(3)+'.h5')
                self.sensor_model.save_weights('./models/S_'+version+'_'+name[index]+str(k).zfill(3)+'.h5')
                self.observation_noise_model.save_weights('./models/R_'+version+'_'+name[index]+str(k).zfill(3)+'.h5')
                print('model is saved at this epoch')

    '''
    The actual testing process
    '''
    def test(self, observations):
        get_loss = enKF_module.getloss()

        test_demo = observations[:, 98, :,:]
        test_demo = tf.reshape(test_demo, [observations.shape[0], 1, 2])
        test_demo = tf.cast(test_demo, tf.float32)

        gt = states_true[:, 98, :,:]
        gt = tf.reshape(gt, [states_true.shape[0], 1, 5])
        gt = tf.cast(gt, tf.float32)


        data = {}
        data_save = []
        emsemble_save = []

        for t in range (observations.shape[0]):
            if t == 0:
                X = tf.convert_to_tensor(np.array([0,0,0,0,1]), dtype=tf.float32)
                m_state = tf.stack([X] * self.batch_size)
                state_old = tf.stack([X] * (self.num_ensemble * self.batch_size))

            # run the filter
            m_state_new, state_new = self.run_enKF_filter(test_demo[t], m_state, state_old)

            loss = get_loss._mse(gt[t] - m_state_new)

            # update states
            m_state = m_state_new
            state_old = state_new
            if t%100 == 0 :
                print('---')
                print(loss)
                print(m_state_new)
                print(gt[t])

            # save the test data
            state_out = np.array(m_state_new)
            ensemble = np.array(tf.reshape(state_new, [self.num_ensemble, 5]))
            data_save.append(state_out)
            emsemble_save.append(ensemble)



        data['state'] = data_save
        data['ensemble'] = emsemble_save

        with open('./output/'+version+'_'+ name[index] +'_.pkl', 'wb') as f:
            pickle.dump(data, f)


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

def main(mode):
    if mode == True:
        batch_size = 1
        num_ensemble = 32
        dropout_rate = 0.4
        filter_ = enKF(batch_size, num_ensemble, dropout_rate)
        filter_.init_framework()
        filter_.train(states_true, observations)
        filter_.test(observations)
    # else:
    #     batch_size = 1
    #     num_ensemble = 32
    #     dropout_rate = 0.4
    #     filter_ = enKF(batch_size, num_ensemble, dropout_rate)
    #     filter_.init_framework()
    #     filter_.test(observations)


global name 
name = ['constant', 'exp']
global index
index = 1
observations, states_true = data_loader_function('./dataset/100_demos_'+name[index]+'.pkl')
global version
version = 'v1.3'

if __name__ == "__main__":
    mode = True
    main(mode)
    # mode = False
    # main(mode)












# def init_framework():
#     '''
#     import sub-modules for diff-enKF framework
#     '''

#     # learned process model
#     process_model = enKF_module.ProcessModel(batch_size, num_ensemble, dim_x, jacobian, dropout_rate)

#     # learned process noise
#     process_noise_model = enKF_module.ProcessNoise(batch_size, num_ensemble, dim_x, q_diag)

#     # learned observation model
#     observation_model = enKF_module.ObservationModel(batch_size, num_ensemble, dim_x, dim_z, jacobian)

#     # learned observation noise
#     observation_noise_model = enKF_module.ObservationNoise(batch_size, num_ensemble, dim_z, r_diag, jacobian)

#     # learned sensor model
#     sensor_model = enKF_module.SensorModel(batch_size, dim_z)

#     # define the initial belief of the filter
#     X = np.array([0,0,0,0,1])
#     X = tf.convert_to_tensor(X, dtype=tf.float32)
#     z = np.array([0,0])
#     z = tf.convert_to_tensor(z, dtype=tf.float32)
#     ensemble_X = tf.stack([X] * (num_ensemble * batch_size))
#     m_X = tf.stack([X] * batch_size)
#     m_z = tf.stack([z] * batch_size)

#     _ = process_model(ensemble_X, True)
#     _, _ = process_noise_model(m_X, True, True)
#     _ = observation_model(ensemble_X, True, True)
#     _, encoding = sensor_model(m_z, True, True)
#     _, _ = observation_noise_model(encoding, True, True)
    
#     return process_model, process_noise_model, observation_model, sensor_model, observation_noise_model    


# def run_filter():



# class DiffenKF(tf.keras.layers.AbstractRNNCell):
#     def __init__(self, batch_size, num_ensemble, dropout_rate,**kwargs):

#         super(DiffenKF, self).__init__(**kwargs)

#         # initialization
#         self.batch_size = batch_size
#         self.num_ensemble = num_ensemble
        
#         self.dim_x = 5
#         self.dim_z = 2

#         self.jacobian = True

#         self.q_diag = np.ones((self.dim_x)).astype(np.float32) * 0.5
#         self.q_diag = self.q_diag.astype(np.float32)

#         self.r_diag = np.ones((self.dim_z)).astype(np.float32) * 0.3
#         self.r_diag = self.r_diag.astype(np.float32)

#         self.scale = 1

#         self.dropout_rate = dropout_rate


#         # predefine all the necessary sub-models
#         # learned sensor model for processing the images

#         # learned process model
#         self.process_model = ProcessModel(self.batch_size, self.num_ensemble, self.dim_x, self.jacobian, self.dropout_rate)

#         # # learned process noise
#         # self.process_noise_model = ProcessNoise(self.batch_size, self.num_ensemble, self.dim_x, self.q_diag)

#         # learned observation model
#         self.observation_model = ObservationModel(self.batch_size, self.num_ensemble, self.dim_x, self.dim_z, self.jacobian)

#         # learned observation noise
#         self.observation_noise_model = ObservationNoise(self.batch_size, self.num_ensemble, self.dim_z, self.r_diag, self.jacobian)

#         # learned sensor model
#         self.sensor_model = SensorModel(self.batch_size, self.dim_z)

#         # # optional: if action is needed
#         # self.add_actions = addAction(self.batch_size, self.dim_x)


#     @property
#     def state_size(self):
#         """size(s) of state(s) used by this cell.
#         It can be represented by an Integer, a TensorShape or a tuple of
#         Integers or TensorShapes.
#         """
#         # estimated state, its covariance, and the step number
#         return [[self.num_ensemble * self.dim_x], [self.dim_x], [1]]

#     @property
#     def output_size(self):
#         """Integer or TensorShape: size of outputs produced by this cell."""
#         # estimated state, observations, Q, R
#         return ([self.dim_x], [self.num_ensemble * self.dim_x], 
#                 [self.dim_z], [self.dim_z])

#     ###########################################################################
#     # convenience functions for ensuring stability

#     ###########################################################################
#     def _condition_number(self, s):
#         """
#         Compute the condition number of a matrix based on its eigenvalues s
#         Parameters
#         ----------
#         s : tensor
#             the eigenvalues of a matrix
#         Returns
#         -------
#         r_corrected : tensor
#             the condition number of the matrix
#         """
#         r = s[..., 0] / s[..., -1]

#         # Replace NaNs in r with infinite
#         r_nan = tf.math.is_nan(r)
#         r_inf = tf.fill(tf.shape(r), tf.constant(np.Inf, r.dtype))
#         r_corrected = tf.where(r_nan, r_inf, r)

#         return r_corrected

#     def _is_invertible(self, s, epsilon=1e-6):
#         """
#         Check if a matrix is invertible based on its eigenvalues s
#         Parameters
#         ----------
#         s : tensor
#             the eigenvalues of a matrix
#         epsilon : float, optional
#             threshold for the condition number
#         Returns
#         -------
#         invertible : tf.bool tensor
#             true if the matrix is invertible
#         """
#         # "c"
#         # Epsilon may be smaller with tf.float64
#         eps_inv = tf.cast(1. / epsilon, s.dtype)
#         cond_num = self._condition_number(s)
#         invertible = tf.logical_and(tf.math.is_finite(cond_num),
#                                     tf.less(cond_num, eps_inv))
#         return invertible

#     def _make_valid(self, covar):
#         """
#         Trys to make a possibly degenerate covariance valid by
#           - replacing nans and infs with high values/zeros
#           - making the matrix symmetric
#           - trying to make the matrix invertible by adding small offsets to
#             the smallest eigenvalues
#         Parameters
#         ----------
#         covar : tensor
#             a covariance matrix that is possibly degenerate
#         Returns
#         -------
#         covar_valid : tensor
#             a covariance matrix that is hopefully valid
#         """
#         # eliminate nans and infs (replace them with high values on the
#         # diagonal and zeros else)
#         bs = covar.get_shape()[0]
#         dim = covar.get_shape()[-1]
#         covar = tf.where(tf.math.is_finite(covar), covar,
#                          tf.eye(dim, batch_shape=[bs])*1e6)

#         # make symmetric
#         covar = (covar + tf.linalg.matrix_transpose(covar)) / 2.

#         # add a bit of noise to the diagonal of covar to prevent
#         # nans in the gradient of the svd
#         noise = tf.random.uniform(covar.get_shape().as_list()[:-1], minval=0,
#                                   maxval=0.001/self.scale**2)
#         s, u, v = tf.linalg.svd(covar + tf.linalg.diag(noise))
#         # test if the matrix is invertible
#         invertible = self._is_invertible(s)
#         # test if the matrix is positive definite
#         pd = tf.reduce_all(tf.greater(s, 0), axis=-1)

#         # try making a valid version of the covariance matrix by ensuring that
#         # the minimum eigenvalue is at least 1e-4/self.scale
#         min_eig = s[..., -1:]
#         eps = tf.tile(tf.maximum(1e-4/self.scale - min_eig, 0),
#                       [1, s.get_shape()[-1] ])
#         covar_invertible = tf.matmul(u, tf.matmul(tf.linalg.diag(s + eps), v,
#                                                   adjoint_b=True))

#         # if the covariance matrix is valid, leave it as is, else replace with
#         # the modified variant
#         covar_valid = tf.where(tf.logical_and(invertible, pd)[:, None, None],
#                                covar, covar_invertible)

#         # make symmetric again
#         covar_valid = \
#             (covar_valid + tf.linalg.matrix_transpose(covar_valid)) / 2.

#         return covar_valid
#     ###########################################################################


#     def call(self, inputs, states):
#         """
#         inputs: KF input, velocity/angular velocity
#         state: x, y, psi, v
#         mode: training or testing 
#         """
#         # decompose inputs and states
#         raw_sensor, actions = inputs

#         raw_sensor = tf.reshape(raw_sensor, [self.batch_size, 1, self.dim_z])
#         # actions = tf.reshape(actions, [self.batch_size, 1, 2])

#         state_old, m_state, step = states

#         state_old = tf.reshape(state_old, [self.batch_size, self.num_ensemble, self.dim_x])

#         m_state = tf.reshape(m_state, [self.batch_size, self.dim_x])

#         training = True

#         '''
#         prediction step
#         state_pred: x_{t}
#                  Q: process noise
#         '''
#         # get prediction and noise of next state
#         state_pred = self.process_model(state_old, training)


#         # Q, diag_Q = self.process_noise_model(m_state, training, True)


#         # # state_pred = state_pred
#         # state_pred = state_pred + Q

#         '''
#         update step
#         state_new: hat_x_{t}
#                 H: observation Jacobians
#                 S: innovation matrix
#                 K: kalman gain

#         '''
#         # get predicted observations
#         learn = True
#         H_X = self.observation_model(state_pred, training, learn)

#         # get the emsemble mean of the observations
#         m = tf.reduce_mean(H_X, axis = 1)
#         for i in range (self.batch_size):
#             if i == 0:
#                 mean = tf.reshape(tf.stack([m[i]] * self.num_ensemble), [self.num_ensemble, self.dim_z])
#             else:
#                 tmp = tf.reshape(tf.stack([m[i]] * self.num_ensemble), [self.num_ensemble, self.dim_z])
#                 mean = tf.concat([mean, tmp], 0)

#         mean = tf.reshape(mean, [self.batch_size, self.num_ensemble, self.dim_z])
#         H_A = H_X - mean

#         final_H_A = tf.transpose(H_A, perm=[0,2,1])
#         final_H_X = tf.transpose(H_X, perm=[0,2,1])

#         # get sensor reading
#         z, encoding = self.sensor_model(raw_sensor, training, True)

#         # enable each ensemble to have a observation
#         z = tf.reshape(z, [self.batch_size, self.dim_z])
#         for i in range (self.batch_size):
#             if i == 0:
#                 ensemble_z = tf.reshape(tf.stack([z[i]] * self.num_ensemble), [1, self.num_ensemble, self.dim_z])
#             else:
#                 tmp = tf.reshape(tf.stack([z[i]] * self.num_ensemble), [1, self.num_ensemble, self.dim_z])
#                 ensemble_z = tf.concat([ensemble_z, tmp], 0)

#         # make sure the ensemble shape matches
#         ensemble_z = tf.reshape(ensemble_z, [self.batch_size, self.num_ensemble, self.dim_z])


#         # get observation noise
#         R, diag_R = self.observation_noise_model(encoding, training, True)

#         # incorporate the measurement with stochastic noise
#         r_mean = np.zeros((self.dim_z))
#         r_mean = tf.convert_to_tensor(r_mean, dtype=tf.float32)
#         r_mean = tf.stack([r_mean] * self.batch_size)
#         nd_r = tfp.distributions.MultivariateNormalDiag(loc=r_mean, scale_diag=diag_R)
#         epsilon = tf.reshape(nd_r.sample(self.num_ensemble), [self.batch_size, self.num_ensemble, self.dim_z])

#         # the measurement y
#         y = ensemble_z + epsilon
#         y = tf.transpose(y, perm=[0,2,1])


#         # calculated innovation matrix s
#         innovation = (1/(self.num_ensemble -1)) * tf.matmul(final_H_A,  H_A) + R

#         # A matrix
#         m_A = tf.reduce_mean(state_pred, axis = 1)
#         for i in range (self.batch_size):
#             if i == 0:
#                 mean_A = tf.reshape(tf.stack([m_A[i]] * self.num_ensemble), [1, self.num_ensemble, self.dim_x])
#             else:
#                 tmp = tf.reshape(tf.stack([m_A[i]] * self.num_ensemble), [1, self.num_ensemble, self.dim_x])
#                 mean_A = tf.concat([mean_A, tmp], 0)
#         A = state_pred - mean_A
#         A = tf.transpose(A, perm = [0,2,1])

#         try:
#             innovation_inv = tf.linalg.inv(innovation)
#         except:
#             innovation = self._make_valid(innovation)
#             innovation_inv = tf.linalg.inv(innovation)


#         # calculating Kalman gain
#         K = (1/(self.num_ensemble -1)) * tf.matmul(tf.matmul(A, H_A), innovation_inv)


#         # update state of each ensemble
#         y_bar = y - final_H_X
#         state_new = state_pred +  tf.transpose(tf.matmul(K, y_bar), perm=[0,2,1])

#         # the ensemble state mean
#         m_state_new = tf.reduce_mean(state_new, axis = 1)

#         # change the shape as defined in the property function
#         state_new = tf.reshape(state_new, [self.batch_size, -1])

#         # tuple structure of updated state
#         state_hat = (state_new, m_state_new, step+1)

#         # tuple structure of the output
#         z = tf.reshape(z, [self.batch_size, -1])

#         # output = (m_state_new, state_new, z, 
#         #     tf.reshape(diag_R, [self.batch_size, -1]), 
#         #     tf.reshape(diag_Q, [self.batch_size, -1]))
#         output = (m_state_new, state_new, z, 
#             tf.reshape(diag_R, [self.batch_size, -1]))

#         # print('===========')
#         # print('output: ',state_hat)
#         # print('===========')

#         return output, state_hat

# class RNNmodel(tf.keras.Model):
#     def __init__(self, batch_size, num_ensemble, dropout_rate, hetero_q=False, hetero_r=True, **kwargs):

#         super(RNNmodel, self).__init__(**kwargs)

#         self.batch_size = batch_size

#         self.num_ensemble = num_ensemble

#         self.dropout_rate = dropout_rate

#         # instantiate the filter
#         self.filter = DiffenKF(batch_size, num_ensemble, dropout_rate)

#         # wrap the filter in a RNN layer
#         self.rnn_layer = tf.keras.layers.RNN(self.filter, return_sequences=True,unroll=False)

#         # define the initial belief of the filter
#         X = np.array([0,0,0,0,1])
#         X = tf.convert_to_tensor(X, dtype=tf.float32)

#         ensemble_X = tf.stack([X] * (self.num_ensemble * self.batch_size))
#         m_X = tf.stack([X] * self.batch_size)

#         ensemble_X = tf.reshape(ensemble_X, [batch_size, -1])
#         m_X = tf.reshape(m_X, [batch_size, -1])

#         step = tf.zeros([batch_size,1])

#         self.state_0 = (ensemble_X, m_X, step)

#     def call(self, inputs):
#         raw_sensor = inputs

#         # action is B*u, which is added to x_{t} after the prediction step 
#         action = np.array([1, np.radians(0.1)])
#         action = tf.convert_to_tensor(action, dtype=tf.float32)
#         action = tf.stack([action] * self.batch_size)
#         action = tf.reshape(action, [self.batch_size, 1, 2])

#         # fake_actions = tf.zeros([batch_size, 1, 2])
#         real_actions = action
#         inputs = (raw_sensor, real_actions)

#         # print('-------------------- ',inputs)

#         outputs = self.rnn_layer(inputs, initial_state=self.state_0)
        
#         return outputs

#     @classmethod
#     def from_config(cls, config):
#         return cls(**config)


