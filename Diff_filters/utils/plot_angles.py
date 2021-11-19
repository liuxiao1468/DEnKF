import pickle
import matplotlib.pyplot as plt
import pdb
import numpy as np

name = ['constant', 'exp']
index = 0 
data_path = './dataset/100_demos_'+name[index]+'.pkl'
with open(data_path, 'rb') as f:
    traj = pickle.load(f)

angles = np.array(traj['xTrue']).reshape(-1,4)[:,2]
y_vals = np.sin(angles)
x_vals = np.cos(angles)
angles_new = np.arctan(y_vals, x_vals)
pdb.set_trace()
plt.figure()
plt.plot(angles_new)
plt.plot(angles)
plt.show()