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
    def __init__(self, batch_size, num_ensemble, dim_x, jacobian, rate):
        super(ProcessModel, self).__init__()
        self.batch_size = batch_size
        self.num_ensemble = num_ensemble
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
        last_state = tf.reshape(last_state, [self.batch_size * self.num_ensemble, self.dim_x])

        fc1 = self.process_fc1(last_state)
        fcadd1 = self.process_fc_add1(fc1)
        fc2 = self.process_fc2(fcadd1)
        fc2 = tf.nn.dropout(fc2, rate=self.rate)
        fcadd2 = self.process_fc_add2(fc2)
        fcadd2 = tf.nn.dropout(fcadd2, rate=self.rate)
        update = self.process_fc3(fcadd2)

        new_state = last_state + update
        new_state = tf.reshape(new_state, [self.batch_size, self.num_ensemble, self.dim_x])

        return new_state


class ObservationModel(tf.keras.Model):
    '''
    process model is taking the state and get a prediction state and 
    calculate the jacobian matrix based on the previous state and the 
    predicted state.
    new_state = [batch_size, 1, dim_x]
            F = [batch_size, dim_x, dim_x]
    state vector 4 -> fc 32 -> fc 64 -> 2
    '''
    def __init__(self, batch_size, num_ensemble, dim_x, dim_z, jacobian):
        super(ObservationModel, self).__init__()
        self.batch_size = batch_size
        self.num_ensemble = num_ensemble
        self.jacobian = jacobian
        self.dim_x = dim_x
        self.dim_z = dim_z

    def build(self, input_shape):
        self.observation_fc1 = tf.keras.layers.Dense(
            units=32,
            activation=tf.nn.relu,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
            name='observation_fc1')
        self.observation_fc_add1 = tf.keras.layers.Dense(
            units=64,
            activation=tf.nn.relu,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
            name='observation_fc_add1')
        self.observation_fc2 = tf.keras.layers.Dense(
            units=64,
            activation=tf.nn.relu,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
            name='observation_fc2')
        self.observation_fc_add2 = tf.keras.layers.Dense(
            units=32,
            activation=tf.nn.relu,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
            name='observation_fc_add2')
        self.observation_fc3 = tf.keras.layers.Dense(
            units=self.dim_z,
            activation=None,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
            name='observation_fc3')

    def call(self, state, training, learn):
        state = tf.reshape(state, [self.batch_size* self.num_ensemble, 1, self.dim_x])
        if learn == False:
            H = tf.concat(
                [tf.tile(np.array([[[1, 0, 0, 0, 0]]], dtype=np.float32),
                         [self.batch_size* self.num_ensemble, 1, 1]),
                 tf.tile(np.array([[[0, 1, 0, 0, 0]]], dtype=np.float32),
                         [self.batch_size* self.num_ensemble, 1, 1])], axis=1)
            z_pred = tf.matmul(H, tf.transpose(state, perm=[0,2,1]))
            Z_pred = tf.transpose(z_pred, perm=[0,2,1])
            z_pred = tf.reshape(z_pred, [self.batch_size, self.num_ensemble, self.dim_z])
        else:
            fc1 = self.observation_fc1(state)
            fcadd1 = self.observation_fc_add1(fc1)
            fc2 = self.observation_fc2(fcadd1)
            fcadd2 = self.observation_fc_add2(fc2)
            z_pred = self.observation_fc3(fcadd2)
            z_pred = tf.reshape(z_pred, [self.batch_size, self.num_ensemble, self.dim_z])

        return z_pred
class SensorModel(tf.keras.Model):
    '''
    sensor model is used for modeling H with given states to get observation z
    it is not required for this model to take states only, if the obervation is 
    an image or higher dimentional tensor, it is supposed to learn a lower demention
    representation from the observation space.
    observation = [batch_size, dim_z]
    encoding = [batch_size, dim_fc2]
    '''
    def __init__(self, batch_size, dim_z):
        super(SensorModel, self).__init__()
        self.batch_size = batch_size
        self.dim_z = dim_z

    def build(self, input_shape):
        self.sensor_fc1 = tf.keras.layers.Dense(
            units=32,
            activation=tf.nn.relu,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
            name='sensor_fc1')
        self.sensor_fc_add1 = tf.keras.layers.Dense(
            units=64,
            activation=tf.nn.relu,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
            name='sensor_fc_add1')
        self.sensor_fc2 = tf.keras.layers.Dense(
            units=64,
            activation=tf.nn.relu,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
            name='sensor_fc2')
        self.sensor_fc_add2 = tf.keras.layers.Dense(
            units=32,
            activation=tf.nn.relu,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
            name='sensor_fc_add2')
        self.sensor_fc3 = tf.keras.layers.Dense(
            units=self.dim_z,
            activation=None,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
            name='sensor_fc3')

    def call(self, state, training, learn):
        if learn == True:
            fc1 = self.sensor_fc1(state)
            fcadd1 = self.sensor_fc_add1(fc1)
            fc2 = self.sensor_fc2(fcadd1)
            fcadd2 = self.sensor_fc_add2(fc2)
            observation = self.sensor_fc3(fcadd2)
            encoding = fcadd2
        else:
            observation = state
            encoding = state

        return observation, encoding


class ProcessNoise(tf.keras.Model):
    '''
    Noise model is asuming the noise to be heteroscedastic
    The noise is not constant at each step
    The fc neural network is designed for learning the diag(Q)
    Q = [batch_size, dim_x, dim_x]
    i.e., 
    if the state has 4 inputs
    state vector 4 -> fc 32 -> fc 64 -> 4
    the result is the diag of Q where Q is a 4x4 matrix
    '''
    def __init__(self, batch_size, num_ensemble, dim_x, q_diag):
        super(ProcessNoise, self).__init__()
        self.batch_size = batch_size
        self.num_ensemble = num_ensemble
        self.dim_x = dim_x
        self.q_diag = q_diag

    def build(self, input_shape):
        constant = np.ones(self.dim_x)* 1e-3
        init = np.sqrt(np.square(self.q_diag) - constant)
        self.fixed_process_noise_bias = self.add_weight(
            name = 'fixed_process_noise_bias',
            shape = [self.dim_x],
            regularizer = tf.keras.regularizers.l2(l=1e-3),
            initializer = tf.constant_initializer(constant))
        self.process_noise_fc1 = tf.keras.layers.Dense(
            units=32,
            activation=tf.nn.relu,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
            name='process_noise_fc1')
        self.process_noise_fc_add1 = tf.keras.layers.Dense(
            units=64,
            activation=tf.nn.relu,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
            name='process_noise_fc_add1')
        self.process_noise_fc2 = tf.keras.layers.Dense(
            units=64,
            activation=tf.nn.relu,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
            name='process_noise_fc2')
        self.process_noise_fc3 = tf.keras.layers.Dense(
            units=self.dim_x,
            activation=None,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
            name='process_noise_fc3')
        self.learned_process_noise_bias = self.add_weight(
            name = 'learned_process_noise_bias',
            shape = [self.dim_x],
            regularizer = tf.keras.regularizers.l2(l=1e-3),
            initializer = tf.constant_initializer(init))

    def call(self, state, training):
        fc1 = self.process_noise_fc1(state)
        fcadd1 = self.process_noise_fc_add1(fc1)
        fc2 = self.process_noise_fc2(fcadd1)
        diag = self.process_noise_fc3(fc2)
        
        diag = tf.square(diag + self.learned_process_noise_bias)
        diag = diag + self.fixed_process_noise_bias
        mean = np.zeros((self.dim_x))
        mean = tf.convert_to_tensor(mean, dtype=tf.float32)
        mean = tf.stack([mean] * self.batch_size)
        nd = tfp.distributions.MultivariateNormalDiag(loc=mean, scale_diag=diag)
        Q = tf.reshape(nd.sample(self.num_ensemble), [self.batch_size, self.num_ensemble, self.dim_x])

        return Q, diag


class ObservationNoise(tf.keras.Model):
    '''
    Noise model is asuming the noise to be heteroscedastic
    The noise is not constant at each step
    inputs: an intermediate representation of the raw observation
    denoted as encoding 
    R = [batch_size, dim_z, dim_z]
    The fc neural network is designed for learning the diag(R)
    i.e., 
    if the state has 4 inputs, the encoding has size 64,
    observation vector z is with size 2, the R has the size
    2 + (64 -> fc 2 -> 2) + fixed noise,
    the result is the diag of R where R is a 2x2 matrix
    '''
    def __init__(self, batch_size, num_ensemble, dim_z, r_diag, jacobian):
        super(ObservationNoise, self).__init__()
        self.batch_size = batch_size
        self.num_ensemble = num_ensemble
        self.jacobian = jacobian
        self.dim_z = dim_z
        self.r_diag = r_diag

    def build(self, input_shape):
        constant = np.ones(self.dim_z)* 1e-3
        init = np.sqrt(np.square(self.r_diag) - constant)
        self.fixed_observation_noise_bias = self.add_weight(
            name = 'fixed_observation_noise_bias',
            shape = [self.dim_z],
            regularizer = tf.keras.regularizers.l2(l=1e-3),
            initializer = tf.constant_initializer(constant))

        self.observation_noise_fc1 = tf.keras.layers.Dense(
            units=self.dim_z,
            activation=None,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
            name='observation_noise_fc1')

        self.learned_observation_noise_bias = self.add_weight(
            name = 'learned_observation_noise_bias',
            shape = [self.dim_z],
            regularizer = tf.keras.regularizers.l2(l=1e-3),
            initializer = tf.constant_initializer(init))

    def call(self, inputs, training):
        diag = self.observation_noise_fc1(inputs)
        # print(inputs)
        # print(diag)
        diag = tf.square(diag + self.learned_observation_noise_bias)
        diag = diag + self.fixed_observation_noise_bias
        R = tf.linalg.diag(diag)
        R = tf.reshape(R, [self.batch_size, self.dim_z, self.dim_z])
        diag = tf.reshape(diag, [self.batch_size, self.dim_z])

        return R, diag

class ensembleKF(tf.keras.Model):
    def __init__(self, batch_size, num_ensemble, dropout_rate,**kwargs):
        super(ensembleKF, self).__init__()
                # initialization
        self.batch_size = batch_size
        self.num_ensemble = num_ensemble
        
        self.dim_x = 3
        self.dim_z = 6

        self.jacobian = True

        self.q_diag = np.ones((self.dim_x)).astype(np.float32) * 1
        self.q_diag = self.q_diag.astype(np.float32)

        self.r_diag = np.ones((self.dim_z)).astype(np.float32) * 1
        self.r_diag = self.r_diag.astype(np.float32)

        self.scale = 1

        self.dropout_rate = dropout_rate

        # predefine all the necessary sub-models
        # learned sensor model for processing the images

        # learned process model
        self.process_model = ProcessModel(self.batch_size, self.num_ensemble, self.dim_x, self.jacobian, self.dropout_rate)

        # learned process noise
        self.process_noise_model = ProcessNoise(self.batch_size, self.num_ensemble, self.dim_x, self.q_diag)

        # learned observation model
        self.observation_model = ObservationModel(self.batch_size, self.num_ensemble, self.dim_x, self.dim_z, self.jacobian)

        # learned observation noise
        self.observation_noise_model = ObservationNoise(self.batch_size, self.num_ensemble, self.dim_z, self.r_diag, self.jacobian)

        # learned sensor model
        self.sensor_model = SensorModel(self.batch_size, self.dim_z)

    @property
    def state_size(self):
        """size(s) of state(s) used by this cell.
        It can be represented by an Integer, a TensorShape or a tuple of
        Integers or TensorShapes.
        """
        # estimated state, its covariance, and the step number
        return [[self.num_ensemble * self.dim_x], [self.dim_x], [1]]

    @property
    def output_size(self):
        """Integer or TensorShape: size of outputs produced by this cell."""
        # estimated state, observations, Q, R
        return ([self.dim_x], [self.num_ensemble * self.dim_x], 
                [self.dim_x])

    ###########################################################################
    # convenience functions for ensuring stability

    ###########################################################################
    def _condition_number(self, s):
        """
        Compute the condition number of a matrix based on its eigenvalues s
        Parameters
        ----------
        s : tensor
            the eigenvalues of a matrix
        Returns
        -------
        r_corrected : tensor
            the condition number of the matrix
        """
        r = s[..., 0] / s[..., -1]

        # Replace NaNs in r with infinite
        r_nan = tf.math.is_nan(r)
        r_inf = tf.fill(tf.shape(r), tf.constant(np.Inf, r.dtype))
        r_corrected = tf.where(r_nan, r_inf, r)

        return r_corrected

    def _is_invertible(self, s, epsilon=1e-6):
        """
        Check if a matrix is invertible based on its eigenvalues s
        Parameters
        ----------
        s : tensor
            the eigenvalues of a matrix
        epsilon : float, optional
            threshold for the condition number
        Returns
        -------
        invertible : tf.bool tensor
            true if the matrix is invertible
        """
        # "c"
        # Epsilon may be smaller with tf.float64
        eps_inv = tf.cast(1. / epsilon, s.dtype)
        cond_num = self._condition_number(s)
        invertible = tf.logical_and(tf.math.is_finite(cond_num),
                                    tf.less(cond_num, eps_inv))
        return invertible

    def _make_valid(self, covar):
        """
        Trys to make a possibly degenerate covariance valid by
          - replacing nans and infs with high values/zeros
          - making the matrix symmetric
          - trying to make the matrix invertible by adding small offsets to
            the smallest eigenvalues
        Parameters
        ----------
        covar : tensor
            a covariance matrix that is possibly degenerate
        Returns
        -------
        covar_valid : tensor
            a covariance matrix that is hopefully valid
        """
        # eliminate nans and infs (replace them with high values on the
        # diagonal and zeros else)
        bs = covar.get_shape()[0]
        dim = covar.get_shape()[-1]
        covar = tf.where(tf.math.is_finite(covar), covar,
                         tf.eye(dim, batch_shape=[bs])*1e6)

        # make symmetric
        covar = (covar + tf.linalg.matrix_transpose(covar)) / 2.

        # add a bit of noise to the diagonal of covar to prevent
        # nans in the gradient of the svd
        noise = tf.random.uniform(covar.get_shape().as_list()[:-1], minval=0,
                                  maxval=0.001/self.scale**2)
        s, u, v = tf.linalg.svd(covar + tf.linalg.diag(noise))
        # test if the matrix is invertible
        invertible = self._is_invertible(s)
        # test if the matrix is positive definite
        pd = tf.reduce_all(tf.greater(s, 0), axis=-1)

        # try making a valid version of the covariance matrix by ensuring that
        # the minimum eigenvalue is at least 1e-4/self.scale
        min_eig = s[..., -1:]
        eps = tf.tile(tf.maximum(1e-4/self.scale - min_eig, 0),
                      [1, s.get_shape()[-1] ])
        covar_invertible = tf.matmul(u, tf.matmul(tf.linalg.diag(s + eps), v,
                                                  adjoint_b=True))

        # if the covariance matrix is valid, leave it as is, else replace with
        # the modified variant
        covar_valid = tf.where(tf.logical_and(invertible, pd)[:, None, None],
                               covar, covar_invertible)

        # make symmetric again
        covar_valid = \
            (covar_valid + tf.linalg.matrix_transpose(covar_valid)) / 2.

        return covar_valid
    ###########################################################################


    def call(self, inputs, states, mode):
        """
        inputs: KF input, velocity/angular velocity
        state: x, y, psi, v
        mode: training or testing 
        """

        # decompose inputs and states
        raw_sensor = inputs

        raw_sensor = tf.reshape(raw_sensor, [self.batch_size, 1, self.dim_z])
        # actions = tf.reshape(actions, [self.batch_size, 1, 2])

        state_old, m_state, step = states

        state_old = tf.reshape(state_old, [self.batch_size, self.num_ensemble, self.dim_x])

        m_state = tf.reshape(m_state, [self.batch_size, self.dim_x])

        training = True

        '''
        prediction step
        state_pred: x_{t}
                 Q: process noise
        '''
        # get prediction and noise of next state
        state_pred = self.process_model(state_old, training)


        Q, diag_Q = self.process_noise_model(m_state, training)


        # state_pred = state_pred
        state_pred = state_pred + Q
        if mode == True:

            '''
            update step
            state_new: hat_x_{t}
                    H: observation Jacobians
                    S: innovation matrix
                    K: kalman gain

            '''
            # get predicted observations
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
            z, encoding = self.sensor_model(raw_sensor, training, learn = True)

            # enable each ensemble to have a observation
            z = tf.reshape(z, [self.batch_size, self.dim_z])
            for i in range (self.batch_size):
                if i == 0:
                    ensemble_z = tf.reshape(tf.stack([z[i]] * self.num_ensemble), [1, self.num_ensemble, self.dim_z])
                else:
                    tmp = tf.reshape(tf.stack([z[i]] * self.num_ensemble), [1, self.num_ensemble, self.dim_z])
                    ensemble_z = tf.concat([ensemble_z, tmp], 0)

            # make sure the ensemble shape matches
            ensemble_z = tf.reshape(ensemble_z, [self.batch_size, self.num_ensemble, self.dim_z])


            # get observation noise
            R, diag_R = self.observation_noise_model(encoding, training)

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
                innovation = self._make_valid(innovation)
                innovation_inv = tf.linalg.inv(innovation)


            # calculating Kalman gain
            K = (1/(self.num_ensemble -1)) * tf.matmul(tf.matmul(A, H_A), innovation_inv)


            # update state of each ensemble
            y_bar = y - final_H_X
            state_new = state_pred +  tf.transpose(tf.matmul(K, y_bar), perm=[0,2,1])

        else:
            state_new = state_pred


        # the ensemble state mean
        m_state_new = tf.reduce_mean(state_new, axis = 1)

        # change the shape as defined in the property function
        state_new = tf.reshape(state_new, [self.batch_size, -1])

        # tuple structure of updated state
        state_hat = (state_new, m_state_new, step+1)

        # # tuple structure of the output
        # z = tf.reshape(z, [self.batch_size, -1])

        # output = (m_state_new, state_new, z, 
        #     tf.reshape(diag_R, [self.batch_size, -1]), 
        #     tf.reshape(diag_Q, [self.batch_size, -1]))
        output = (m_state_new, state_new, tf.reshape(diag_Q, [self.batch_size, -1]))

        return output, state_hat



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
        batch_size = 32

        # define number of ensemble
        num_ensemble = 32

        # define dropout rate
        dropout_rate = 0.4

        model = ensembleKF(batch_size, num_ensemble, dropout_rate)

        # process_model = ProcessModel(batch_size, 3, True, dropout_rate)

        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

        epoch = 100

        pred_steps = 1

        # define the initial belief of the filter
        X = np.array([0,0,0])
        X = tf.convert_to_tensor(X, dtype=tf.float32)
        ensemble_X = tf.stack([X] * (num_ensemble * batch_size))
        m_X = tf.stack([X] * batch_size)
        ensemble_X = tf.reshape(ensemble_X, [batch_size, -1])
        m_X = tf.reshape(m_X, [batch_size, -1])
        step = tf.zeros([batch_size,1])
        state_0 = (ensemble_X, m_X, step)

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
                    thr = random.uniform(0, 1)
                    observe = True
                    if i == 0:
                        out, state = model(raw_sensor[i], state_0, observe)
                        state_h = tf.reshape(out[0], [batch_size, 1, 3])
                        loss = get_loss._mse( gt[i] - state_h)
                    else:
                        out, state = model(raw_sensor[i], state, observe)
                        state_h = tf.reshape(out[0], [batch_size, 1, 3])
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
            if (k+1) % 100 == 0:
                model.save_weights('new_bio_model_v1.5.h5')
                print('model is saved at this epoch')
    else:
        # define batch_size
        batch_size = 1

        num_ensemble = 32

        dropout_rate = 0.4

        # define the initial belief of the filter
        X = np.array([0,0,0])
        X = tf.convert_to_tensor(X, dtype=tf.float32)
        ensemble_X = tf.stack([X] * (num_ensemble * batch_size))
        m_X = tf.stack([X] * batch_size)
        ensemble_X = tf.reshape(ensemble_X, [batch_size, -1])
        m_X = tf.reshape(m_X, [batch_size, -1])
        step = tf.zeros([batch_size,1])
        state_0 = (ensemble_X, m_X, step)

        # load the model
        model = ensembleKF(batch_size, num_ensemble, dropout_rate)

        test_demo = observations[:, 40, :,:]
        test_demo = tf.reshape(test_demo, [observations.shape[0], 1, 1, 6])
        dummy = model(test_demo[0], state_0, True)
        model.load_weights('new_bio_model.h5')
        model.summary()

        model.layers[0].save_weights('tmp.h5')
        trans_model = ProcessModel(batch_size, num_ensemble, 3, True, dropout_rate)
        dummy = trans_model(ensemble_X, True)
        trans_model.load_weights('tmp.h5')
        trans_model.summary()

        '''
        run a test demo and save the state of the test demo
        '''
        data = {}
        data_save = []
        emsemble_save = []

        for t in range (states_true.shape[0]):
            if t == 0:
                out, state = model(test_demo[t], state_0, True)
            else:
                out, state = model(test_demo[t], state, True)
            state_out = np.array(out[0])
            ensemble = np.array(tf.reshape(out[1], [num_ensemble, 3]))
            # print('----------')
            # print(ensemble)
            data_save.append(state_out)
            emsemble_save.append(ensemble)
        data['state'] = data_save
        data['ensemble'] = emsemble_save

        with open('bio_pred_v1.5.pkl', 'wb') as f:
            pickle.dump(data, f)


        '''
        look at the transition model
        '''
        data = {}
        state_save = []

        for i in range (30):
            data_save = []
            for t in range (math.floor(states_true.shape[0]/4)):
                if t == 0:
                    out, state = model(test_demo[t], state_0, True)
                else:
                    out, state = model(test_demo[t], state, True)
                state_out = np.array(out[0])

                data_save.append(state_out)
            for t in range (states_true.shape[0] - math.floor(states_true.shape[0]/4)):
                if t == 0:
                    state_ensemble = trans_model(out[1], True)
                    state = tf.reduce_mean(state_ensemble, axis = 1)
                    state_out = np.array(state)
                else:
                    state_ensemble = trans_model(state_ensemble, True)
                    state = tf.reduce_mean(state_ensemble, axis = 1)
                    state_out = np.array(state)

                data_save.append(state_out)
            state_save.append(data_save)

        data['state'] = state_save

        with open('bio_transition_v1.5.pkl', 'wb') as f:
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
# states_true, states_true_add1 = transition_dataloader(states_true)

def main():

    training = True
    run_filter(training)

    training = False
    run_filter(training)

if __name__ == "__main__":
    main()