import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf 
import numpy as np 
import pickle
import pdb
import math

# ###################################################################
# '''
# basic set up for calculate tf api
# '''
# F = np.array([
#     [1.0, 0, 0, 0],
#     [0, 1.0, 0, 0],
#     [0, 0, 1.0, 0],
#     [0, 0, 0, 0]])
# F = tf.convert_to_tensor(F, dtype=tf.float32)
# print(F)

# ###################################################################
# '''
# init state_0
# '''
# batch_size = 2

# X = np.zeros((4))
# P = np.eye(4)
# P = tf.convert_to_tensor(P, dtype=tf.float32)
# P = tf.stack([P] * batch_size)
# P = tf.reshape(P, [batch_size, -1])

# X = tf.convert_to_tensor(X, dtype=tf.float32)
# X = tf.stack([X] * batch_size)
# X = tf.reshape(X, [batch_size, -1])

# step = tf.zeros([batch_size,1])
# state_0 = (X, P, step)
# print(state_0)

# action = np.array([1, 0.1])
# action = tf.convert_to_tensor(action, dtype=tf.float32)
# action = tf.stack([action] * batch_size)
# action = tf.reshape(action, [batch_size, 1, 2])
# print(action)


# X = np.zeros((1, 4))
# X = tf.convert_to_tensor(X, dtype=tf.float32)
# X = tf.stack([X] * batch_size)
# print(X)

# # out = tf.reshape(tf.tile(F, tf.constant([batch_size]), [batch_size, tf.shape(F) ]))
# # # out = tf.reshape(F, [batch_size, 4, 4])
# # print(out)
# # print(F)
# # F = tf.squeeze(F)
# # F = tf.reshape(F, [1,4,4])

# # print(F)

# # F = F+tf.eye(4)*0.00000001
# # print(F.get_shape())
# print("========= working on epoch %d =========: " % (1))

# # initial_covariance = np.array([10.0, 10.0, 5.0, 5.0])/ 60.
# # initial_covariance = initial_covariance.astype(np.float32)
# # covar_start = tf.square(initial_covariance)
# # covar_start = tf.linalg.tensor_diag(covar_start)
# # print(covar_start)
# # covar_start = tf.tile(covar_start[None, :, :],[1, 1, 1])
# # print(covar_start)


# # DT = 0.1
# # B = np.array([
# #     [DT * np.cos(np.radians(state[0][0][2])), 0],
# #     [DT * np.sin(np.radians(state[0][0][2])), 0],
# #     [0.0, DT],
# #     [1.0, 0.0]])
# # print(B.shape)
# # B = tf.reshape(B, [1,4,2])
# # print(B)


###################################################################
'''
test the gradient and jacobian calculation
'''

# x = tf.constant([[4, 2],[1, 3]], dtype=tf.dtypes.float32)

# # with tf.GradientTape() as g:
# #   g.watch(x)
# #   y = x * x * x
# # jacobian = g.jacobian(y, x)
# # print(jacobian)

# # Using GradientTape
# with tf.GradientTape() as gfg:
#   gfg.watch(x)
#   y = x * x * x
#   print(y)

# # Computing jacobian
# res  = gfg.batch_jacobian(y, x) 
  
# # Printing result
# print("res: ",res)

# c = tf.matmul(x, res, transpose_b=True)
# print(c)


###################################################################
# '''
# test the tf tensor operations
# '''
# dim_x = 4
# q_diag = dim_x
# init_const = np.ones(dim_x) * 1e-1
# init = np.sqrt(np.square(q_diag) - init_const)

# # print(init_const)
# print('---')
# # print(tf.linalg.diag(init))
# init = tf.convert_to_tensor(init, dtype=tf.float32)
# init_const = tf.convert_to_tensor(init_const, dtype=tf.float32)
# # print(init)
# # print(init - init_const)
# # init = tf.expand_dims(init, -1)
# # print(init.get_shape())

# # a = dim_x // dim_x
# # print('a: ', a)


# batch_size = 2
# H = tf.concat(
#         [tf.tile(np.array([[[1, 0, 0, 0]]], dtype=np.float32),
#                  [batch_size, 1, 1]),
#          tf.tile(np.array([[[0, 1, 0, 0]]], dtype=np.float32),
#                  [batch_size, 1, 1])], axis=1)
# print('H: ')
# print(H)

# z_pred = tf.matmul(tf.tile(H, [1, 1, 1]),
#                    tf.expand_dims(init, -1))
# print('z: ')
# z_pred = tf.reshape(z_pred, [2, 2])
# print(z_pred)
# print(z_pred[0])


# ###################################################################
'''
setup multiple class object 
'''
# class AAA:
#     def addition(self, a, b):
#         return a+b

# class bbb:
#     def mul(self, a, b):
#         return a*b

# class ccc:
#     def operation(self, a,b):
#         c = AAA.addition(self, a,b)
#         d = bbb.mul(self, a,b)

#         return c, d
# C = ccc()
# print(C.operation(2,3))

# class DiffEKF(tf.keras.layers.AbstractRNNCell):
#     def __init__(self, dim_x, **kwargs):
#         self.dim_x = dim_x
#         # self.dim_z = dim_z
#         # self.batch_size = batch_size
#         super(DiffEKF, self).__init__(**kwargs)
#         self.CC = ccc()

#     @property
#     def state_size(self):
#         """size(s) of state(s) used by this cell.
#         It can be represented by an Integer, a TensorShape or a tuple of
#         Integers or TensorShapes.
#         """
#         # estimated state, its covariance, and the step number
#         return self.dim_x

#     @property
#     def output_size(self):
#         """Integer or TensorShape: size of outputs produced by this cell."""
#         # estimated state and covariance, observations, R, Q
#         return self.dim_x

#     def call(self, inputs, states):
#         a = states[0]
#         output = self.CC.operation(a, a)
#         return output, output

# cell = DiffEKF(5)
# x = tf.keras.Input((5))
# print(x)
# layer = tf.keras.layers.RNN(cell)
# y = layer(x)
# print(y)


# class MinimalRNNCell(tf.keras.layers.AbstractRNNCell):

#     def __init__(self, dim_x, **kwargs):
#       self.dim_x = dim_x
#       super(MinimalRNNCell, self).__init__(**kwargs)

#     @property
#     def state_size(self):
#       return self.dim_x

#     def build(self, input_shape):
#       self.kernel = self.add_weight(shape=(input_shape[-1], self.dim_x),
#                                     initializer='uniform',
#                                     name='kernel')
#       self.recurrent_kernel = self.add_weight(
#           shape=(self.dim_x, self.dim_x),
#           initializer='uniform',
#           name='recurrent_kernel')
#       self.built = True

#     def call(self, inputs, states):
#       prev_output = states[0]
#       print('---inputs: ',inputs[0])
#       print('---states: ',states[0])
#       h = tf.keras.backend.dot(inputs, self.kernel)
#       output = h + tf.keras.backend.dot(prev_output, self.recurrent_kernel)
#       return output, output



# cell = MinimalRNNCell(32)
# x = tf.keras.Input((None, 5))
# # print(x)
# layer = tf.keras.layers.RNN(cell)
# print(layer.trainable_weights)
# y = layer(x)
# print(y)
###################################################################
# '''
# reading a txt/pickle file
# '''
import matplotlib.pyplot as plt


# print(len(traj['xTrue']))
# print(len(traj['action']))
# print(len(traj['sensors'][0][0]))


# def data_loader_function(data_path):
#     name = ['constant', 'exp']
#     num_sensors = 100

#     observations = []
#     states_true = []
#     with open(data_path, 'rb') as f:
#         traj = pickle.load(f)
#     for i in range (len(traj['xTrue'])):
#         observation = []
#         state = []
#         for j in range (num_sensors):
#             observe = [traj['sensors'][i][0][j]*(1/20.), traj['sensors'][i][1][j]*(1/20.) ]
#             observation.append(observe)
#             # xTrue = traj['xTrue'][i]

#             angles = traj['xTrue'][i][2]
#             xTrue = [traj['xTrue'][i][0]*(1/20.), traj['xTrue'][i][1]*(1/20.), np.arctan(np.sin(angles), np.cos(angles)), traj['xTrue'][i][1]]
#             # print(xTrue)
#             state.append(xTrue)
#         observations.append(observation)
#         states_true.append(state)
#     observations = np.array(observations)
#     observations = tf.reshape(observations, [len(traj['xTrue']), num_sensors, 1, 2])
#     states_true = np.array(states_true)
#     states_true = tf.reshape(states_true, [len(traj['xTrue']), num_sensors, 1, 4])
#     return observations, states_true

# observations, states_true = data_loader_function('./dataset/100_demos_constant.pkl')

# # print(observations[:, 90, :,:] .shape)
# print(states_true.shape)
# print(states_true[:, 90, :,:])



# import random
# batch_size = 8
# select = random.sample(range(0, 100), batch_size)
# raw_sensor = []
# gt = []
# for idx in select:
#     raw_sensor.append(observations[:, idx, :,:])
#     gt.append(states_true[:, idx, :,:])
# print('check', observations[0,select[0], :, :])
# test = observations[:, idx, :,:]
# test = tf.reshape(test, [627, 1, 1, 2])
# print(np.array(test[0][0][0]))

# raw_sensor = tf.convert_to_tensor(raw_sensor, dtype=tf.float32)
# raw_sensor = tf.reshape(raw_sensor, [627, batch_size, 1, 2])
# gt = tf.convert_to_tensor(gt, dtype=tf.float32)
# gt = tf.reshape(gt, [states_true.shape[0], batch_size, 1, 4])



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

get_loss = getloss()
a = np.array([1,2,3,4])
b = np.array([-2,-3,-4,-5])
a = tf.convert_to_tensor(a, dtype=tf.float32)
b = tf.convert_to_tensor(b, dtype=tf.float32)

diff = b - a
print(diff)
loss, dist = get_loss._mse(diff)
print(loss)




# test = observations[:, 98, :,:]
# test = tf.reshape(test, [observations.shape[0], 1, 1, 2])
# test = np.array(test)
# print(np.array(test[0]))
# print(np.array(test[0][0][0]))

# # '''
# # plot traj to visualize 
# # '''
# with open('./output/output_constant_01.pkl', 'rb') as f:
#     test_demo = pickle.load(f)
# print('==========')
# print(test_demo[0][0][0][0])
# print(len(test_demo))

# fig = plt.figure()
# fig.suptitle('output')
# for i in range(len(test_demo)):
#     xTrue = test_demo[i][0][0][0]*(1/20.)
#     yTrue = test_demo[i][0][0][1]*(1/20.)
#     plt.scatter(xTrue, yTrue, s = 5, color = 'r', lw=0.05)
#     plt.scatter(test[i][0][0][0], test[i][0][0][1], s = 5, color = 'b', lw=0.05)

#     # plt.scatter(xTrue, yTrue, s = 5, color = 'r')

# plt.grid()
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()


###########################################################
# import matplotlib.pyplot as plt
# mean = [0, 0, 0]
# cov = [[1, 0, 0], [0, 100, 0], [0, 0, 1] ]  # diagonal covariance
# tmp= np.random.multivariate_normal(mean, cov, 5000).T
# print(tmp.shape)
# # plt.plot(x, y, 'x')
# # plt.axis('equal')
# # plt.show()



# import tensorflow_probability as tfp
# # Let mean vector and co-variance be:
# mu = [1., 2] 
# cov = [[ 1,  3/5],[ 3/5,  2]]

# # dim_x = 4
# # mean = [0. ,0., 0., 0.]
# # diag = [1., 1., 1., 1.]
# # sam = tfp.distributions.MultivariateNormalDiag(loc=mean, scale_diag=diag)
# # noise = sam.sample(32)
# # print(noise)



# tfd = tfp.distributions
# nd = tfp.distributions.MultivariateNormalDiag(loc=[0., 10., 1.], scale_diag=[1., 4., 1.])
# sample = tf.reshape(nd.sample(5*2), [2,5,3])
# # print(sample)
# # print(nd.sample(5))

batch_size = 4
num_ensemble = 32

###################################### try tf probability distribution ##########################
# X = np.zeros((4))
# P = np.ones(4)
# P = tf.convert_to_tensor(P, dtype=tf.float64)
# P = tf.stack([P] * batch_size)
# X = tf.stack([X] * batch_size)
# print(P)
# print(X)
# tfd = tfp.distributions
# nd = tfp.distributions.MultivariateNormalDiag(loc=X, scale_diag=P)
# sample = tf.reshape(nd.sample(32), [batch_size, 32, 4])
# sample = tf.cast(sample, tf.float32)



# sample = tf.reshape(sample, [batch_size* num_ensemble, 1, 4])



###################################### try to ensemble the observation ##########################
# H = tf.concat(
#         [tf.tile(np.array([[[1, 0, 0, 0]]], dtype=np.float32),
#                  [batch_size*num_ensemble, 1, 1]),
#          tf.tile(np.array([[[0, 1, 0, 0]]], dtype=np.float32),
#                  [batch_size*num_ensemble, 1, 1])], axis=1)
# print('--- ensemble ---: ',sample.shape)
# print(H.shape)
# z_pred = tf.matmul(H, tf.transpose(sample, perm=[0,2,1]))
# Z_pred = tf.transpose(z_pred, perm=[0,2,1])
# z_pred = tf.reshape(z_pred, [batch_size, num_ensemble, 2])
# print('--- observation ---: ',z_pred.shape)

# m = tf.reduce_mean(z_pred, axis = 1)
# print(m.shape)
# # m = tf.stack([m] * num_ensemble)
# for i in range (batch_size):
#     if i == 0:
#         mean = tf.reshape(tf.stack([m[i]] * num_ensemble), [1, num_ensemble, 2])
#     else:
#         tmp = tf.reshape(tf.stack([m[i]] * num_ensemble), [1, num_ensemble, 2])
#         mean = tf.concat([mean, tmp], 0)
# mean = tf.reshape(mean, [batch_size, num_ensemble, 2])
# # print(mean)
# # print(m)
# res = z_pred - mean
# res = tf.transpose(res, perm=[0,2,1])
# print(res)
