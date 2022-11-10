import scipy.io
import pickle
import numpy as np
import matplotlib.pyplot as plt

col = ['gyro_speed', 'hall_angle', 'hall_volt_raw', 'mocap_angle', 'time_sensor']
# mat = scipy.io.loadmat('sensor_data_trial3.mat')
# data_1 = mat[col[3]]

# # print(data_1.shape)
# # print(data_1[0][0].shape)
# # print(data_1[0][1].shape)
# # print(data_1[0][2].shape)
# print(mat[col[0]][0][0].shape[0])

def create_dataset(path):
    mat = scipy.io.loadmat(path)
    length = mat[col[0]][0][0].shape[0]
    # gyro_speed
    w1 = mat[col[0]][0][0]
    w2 = mat[col[0]][0][1]
    w3 = mat[col[0]][0][2]
    # hall_angle
    h1 = mat[col[1]][0][0]
    h2 = mat[col[1]][0][1]
    h3 = mat[col[1]][0][2]
    h4 = mat[col[1]][0][3]
    # mocao_angle
    theta1 = mat[col[3]][0][0]
    theta2 = mat[col[3]][0][1]
    theta3 = mat[col[3]][0][2]
    # all data
    data = np.concatenate((w1, w2, w3, h1, h2, h3, h4, theta1, theta2, theta3), axis=1)
    return data

path_1 = 'sensor_data_trial2.mat'
data_1 = create_dataset(path_1)
path_2 = 'sensor_data_trial3.mat'
data_2 = create_dataset(path_2)

############ create dataset ############
data_1 = data_1[250:1800]
data_2 = data_2[315:1950]
total_data = np.concatenate((data_1, data_2), axis=0)
print('total data: ',total_data.shape)
std = np.std(total_data, axis=0)
m = np.mean(total_data, axis=0)
# print(std)
# print(m)
def create_dataset_KF(data):
    test_data = data[-100:, :]
    train_data = data[:-100, :]
    train_set = []
    for i in range (train_data.shape[0]-1):
        state_pre = train_data[i,:]
        state_gt = train_data[i+1,:]
        obs = train_data[i+1,:7]
        train_set.append([state_pre, state_gt, obs])
    test_set = []
    for i in range (test_data.shape[0]-1):
        state_pre = test_data[i,:]
        state_gt = test_data[i+1,:]
        obs = test_data[i+1,:7]
        test_set.append([state_pre, state_gt, obs])
    return train_set, test_set

train1, test1 = create_dataset_KF(data_1)
train2, test2 = create_dataset_KF(data_2)
train = train1 + train2
f = open('train_set.pkl', 'wb')
pickle.dump(train, f)
f = open('test_set_1.pkl', 'wb')
pickle.dump(test1, f)
f = open('test_set_2.pkl', 'wb')
pickle.dump(test2, f)





# ############ to plot the data ############
# data_1 = data_1[250:1800]
# data_1 = data_2[315:1950]
# x = np.linspace(1, data_1.shape[0], data_1.shape[0])
# fig = plt.figure()
# for i in range (data_1.shape[1]):
#     plt.plot(x, data_1[:,i].flatten(),linewidth=1, alpha=0.5)
#     if i >= 7:
#         plt.plot(x, data_1[:,i].flatten(),linewidth=2, alpha=1, label= 'Mocap')
# plt.legend(loc='upper left')
# plt.show()
# ###########################################


    
