import pickle
import matplotlib.pyplot as plt
import numpy as np
import pdb

subtract_mean = False
sensor_idx = 0
num_sensors = 100

name = ['constant', 'exp']

# plt.figure()

'''
plot noise
'''
fig = plt.figure(figsize=(2, 2))
fig.suptitle('Noise in Observation')
for j in range(2):
    with open('./dataset/'+str(num_sensors) + '_demos_'+name[j]+'.pkl', 'rb') as f:
        traj = pickle.load(f)
    sensors = np.array(traj['sensors'])
    for sensor_idx in range(2):
        for i in range(len(sensors)):
            if subtract_mean:
                mean = np.mean(sensors[i][sensor_idx, :])
            else:
                mean = 0
            data = sensors[i][sensor_idx, :] - mean
            plt.subplot(2, 2, 2*j+sensor_idx+1)
            plt.grid()
            plt.xlabel('time step')
            plt.ylabel('Observation')
            plt.scatter(np.ones(num_sensors)*i, data, lw=0.05)
plt.show()


'''
plot demos
'''
fig = plt.figure(figsize = (1,2))
fig.suptitle('demonstrations')
for j in range(2):
    with open('./dataset/'+str(num_sensors) + '_demos_'+name[j]+'.pkl', 'rb') as f:
        traj = pickle.load(f)
    sensors = np.array(traj['sensors'])
    plt.subplot(1, 2, j+1)
    for i in range(len(sensors)):
        data_1 = sensors[i][0, :]
        data_2 = sensors[i][1, :]
        plt.grid()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.scatter(data_1, data_2, s = 5, lw=0.05)
    for i in range(len(sensors)):
        xTrue = traj['xTrue'][i][0]
        yTrue = traj['xTrue'][i][1]
        plt.scatter(xTrue, yTrue, s = 10, color = 'r', lw=0.05)
plt.show()