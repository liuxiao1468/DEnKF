# Copyright (c) 2020 Max Planck Gesellschaft

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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

class ProcessModel(tf.keras.Model):
    '''
    process model is taking the state and get a distribution of a prediction state,
    which is represented as ensemble.
    new_state = [batch_size, num_particles, dim_x]
    state vector dim_x -> fc 32 -> fc 64 -> fc 32 -> dim_x
    '''
    def __init__(self, batch_size, num_particles, dim_x):
        super(ProcessModel, self).__init__()
        self.batch_size = batch_size
        self.num_particles = num_particles
        self.dim_x = dim_x

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

    def call(self, last_state):
        last_state = tf.reshape(last_state, [self.batch_size * self.num_particles, self.dim_x])

        # we pass the action into the process model with the cosine and sine
        theta = tf.reshape(last_state[:,2], [self.batch_size * self.num_particles, 1])
        theta = -(theta-np.pi/2)
        st = tf.sin(theta)
        ct = tf.cos(theta)
        data_in = tf.concat([last_state[:,3:], ct, st], axis = -1)

        fc1 = self.process_fc1(data_in)
        fcadd1 = self.process_fc_add1(fc1)
        fc2 = self.process_fc2(fcadd1)
        fcadd2 = self.process_fc_add2(fc2)
        update = self.process_fc3(fcadd2)

        new_state = last_state + update
        new_state = tf.reshape(new_state, [self.batch_size, self.num_particles, self.dim_x])

        return new_state

class ImageSensorModel(tf.keras.Model):
    '''
    sensor model is used for learning a representation of the current observation,
    the representation can be explainable or latent.  
    observation = [batch_size, img_h, img_w, channel]
    encoding = [batch_size, dim_fc3]
    '''
    def __init__(self, batch_size, dim_z):
        super(ImageSensorModel, self).__init__()
        self.batch_size = batch_size
        self.dim_z = dim_z

    def build(self, input_shape):
        self.sensor_conv1 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=7,
            strides=[1, 1],
            activation=tf.nn.relu,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
            name='sensor_conv1')

        self.sensor_conv2 = tf.keras.layers.Conv2D(
            filters=32, kernel_size=5,
            strides=[1, 2],
            activation=tf.nn.relu,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
            name='sensor_conv2')

        self.sensor_conv3 = tf.keras.layers.Conv2D(
            filters=32, kernel_size=5,
            strides=[1, 2],
            activation=tf.nn.relu,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
            name='sensor_conv3')

        self.sensor_conv4 = tf.keras.layers.Conv2D(
            filters=32, kernel_size=5,
            strides=[2, 2],
            activation=tf.nn.relu,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
            name='sensor_conv4')

        self.flatten = tf.keras.layers.Flatten()

        # bayesian neural networks
        self.sensor_fc1 = tf.keras.layers.Dense(
            units=128,
            activation=tf.nn.relu,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
            name='sensor_fc1')
        self.sensor_fc2 = tf.keras.layers.Dense(
            units=128,
            activation=tf.nn.relu,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
            name='sensor_fc2')
        self.sensor_fc3 = tf.keras.layers.Dense(
            units=self.dim_z,
            activation=None,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
            name='sensor_fc3')

    def call(self, image):
        conv1 = self.sensor_conv1(image)
        # conv1 = tf.nn.max_pool2d(conv1, 2, 2, padding='SAME')
        conv2 = self.sensor_conv2(conv1)
        # conv2 = tf.nn.max_pool2d(conv2, 2, 2, padding='SAME')
        conv3 = self.sensor_conv3(conv2)
        # conv3 = tf.nn.max_pool2d(conv3, 2, 2, padding='SAME')
        conv4 = self.sensor_conv4(conv3)

        conv4 = tf.nn.dropout(conv4, rate=0.3)

        inputs = self.flatten(conv4)

        fc1 = self.sensor_fc1(inputs)
        fc2 = self.sensor_fc2(fc1)
        observation = self.sensor_fc3(fc2)
        encoding = fc2

        observation = tf.reshape(observation, [self.batch_size, 1, self.dim_z])
        encoding = tf.reshape(encoding, [self.batch_size, 1, 128])

        return observation, encoding

class Likelihood(tf.keras.Model):
    '''
    likelihood function is used to generate the probability for each particle with given
    observation encoding
    particles = [batch_size, num_particles, dim_x]
    like = [batch_size, num_particles]
    '''
    def __init__(self, batch_size, num_particles):
        super(Likelihood, self).__init__()
        self.batch_size = batch_size
        self.num_particles = num_particles

    def build(self, input_shape):
        self._fc_layer_1 = tf.keras.layers.Dense(
            units=128,
            activation=tf.nn.relu,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
            name='_fc_layer_1')
        self._fc_layer_2 = tf.keras.layers.Dense(
            units=128,
            activation=tf.nn.relu,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
            name='_fc_layer_2')
        self._fc_layer_3 = tf.keras.layers.Dense(
            units=1,
            activation=tf.nn.sigmoid,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
            name='_fc_layer_3')

    def call(self, inputs):
        # unpack the inputs
        particles, encoding = inputs

        # expand the encoding into particles
        for n in range (self.batch_size):
            if n == 0:
                encodings = tf.reshape(tf.stack([encoding[n]] * self.num_particles), [1, self.num_particles, 128])
            else:
                tmp = tf.reshape(tf.stack([encoding[n]] * self.num_particles), [1, self.num_particles, 128])
                encodings = tf.concat([encodings, tmp], 0)

        encodings = tf.reshape(encodings, [self.batch_size * self.num_particles, 128])
        like = self._fc_layer_1(encodings)
        like = self._fc_layer_2(like)
        like = self._fc_layer_3(like)

        like = tf.reshape(like, [self.batch_size, self.num_particles])
        return like

class Particle_filter(tf.keras.Model):
    def __init__(self, batch_size, num_particles, **kwargs):
        super(Particle_filter, self).__init__()

        # initialization
        self.batch_size = batch_size
        self.num_particles = num_particles
        
        self.dim_x = 5
        self.dim_z = 2

        # learned process model
        self.process_model = ProcessModel(self.batch_size, self.num_particles, self.dim_x)

        # learned likelihood model
        self.likelihood_model = Likelihood(self.batch_size, self.num_particles)

        # learned sensor model
        self.sensor_model = ImageSensorModel(self.batch_size, self.dim_z)

    def call(self, inputs, states):
        raw_sensor = inputs

        particles, weights, m_state = states

        particles_new = self.process_model(particles)

        z, encoding = self.sensor_model(raw_sensor)

        like = self.likelihood_model(encoding)

        weights = weights + like

        # resample based on new weights
        w = tf.reduce_sum(weights, axis=1)
        w = tf.stack([w]*self.num_particles)
        w = tf.transpose(w, perm=[1,0])
        weights = weights/w

        for i in range (self.batch_size):
            weight = weights[i]
            N = len(weight)
            # make N subdivisions, and chose a random position within each one
            positions = (random(N) + range(N)) / N
            indexes = np.zeros(N, 'i')
            cumulative_sum = np.cumsum(weight)
            i, j = 0, 0
            while i < N:
                if positions[i] < cumulative_sum[j]:
                    indexes[i] = j
                    i += 1
                else:
                    j += 1
            idx.append(indexes)
        idx = np.array(idx)

        for i in range (self.batch_size):
            if i == 0:
                new_particle = tf.reshape(tf.gather(particles[i], idx[i]), [1, self.num_particles, self.dim_x])
            else:
                tmp = tf.reshape(tf.gather(particles[i], idx[i]), [1, self.num_particles, self.dim_x])
                new_particle = tf.concat([new_particle, tmp], 0)

        weights = tf.expand_dims(weights,1)
        m_state = tf.matmul(weights, new_particle)
        
        return new_particle, weights, m_state



        






