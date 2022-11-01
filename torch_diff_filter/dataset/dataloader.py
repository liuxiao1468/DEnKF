import os
import random
from PIL import Image
import cv2
import numpy as np
from torch.utils.data import Dataset
import pickle
import torch
from einops import rearrange, repeat

class transform:
    def __init__(self):
        super(transform, self).__init__()
        parameters = pickle.load(open('./dataset/full_parameters.pkl', 'rb'))
        self.state_m = parameters['state_m']
        self.state_std = parameters['state_std']
        self.obs_m = parameters['obs_m']
        self.obs_std = parameters['obs_std']

    def state_transform(self, state):
        '''
        state -> [num_ensemble, dim_x]
        '''
        state = (state - self.state_m)/self.state_std
        return state

    def obs_transform(self, obs):
        '''
        obs -> [num_ensemble, dim_z]
        '''
        obs = (obs - self.obs_m)/self.obs_std
        return obs

    def state_inv_transform(self, state):
        '''
        state -> [num_ensemble, dim_x]
        '''
        state = (state * self.state_std) + self.state_m 
        return state

    def obs_inv_transform(self, obs):
        '''
        obs -> [num_ensemble, dim_z]
        '''
        obs = (obs * self.obs_std) + self.obs_m 
        return obs

class utils:
    def __init__(self, num_ensemble, dim_x, dim_z):
        self.num_ensemble = num_ensemble
        self.dim_x = dim_x
        self.dim_z = dim_z

    def format_state(self, state):
        state = repeat(state, 'k dim -> n k dim', n = self.num_ensemble)
        state = rearrange(state, 'n k dim -> (n k) dim')
        state = state.to(dtype=torch.float32)
        return state


class KITTIDataset(Dataset):
    # Basic Instantiation
    def __init__(self, num_ensemble, dim_x, dim_z, mode):
        self.dataset_path = '/Users/xiao.lu/project/KITTI_dataset/'
        self.dataset = pickle.load(open('./dataset/KITTI_VO_dataset.pkl', 'rb'))
        self.dataset_length = len(self.dataset)
        self.num_ensemble = num_ensemble
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.transform_ = transform()
        self.utils_ = utils(self.num_ensemble, self.dim_x, self.dim_z)


    def preprocessing(self, data, mode):
        img_2 = cv2.imread(self.dataset_path+data[3][1])
        img_1 = cv2.imread(self.dataset_path+data[3][0])
        # img_2 = cv2.imread(data[3][1])
        # img_1 = cv2.imread(data[3][0])
        # if mode == 'train':
        #     img_2 = cv2.flip(img_2, 1)
        #     img_1 = cv2.flip(img_1, 1)
        # 150, 50
        # 640, 192
        img_2 = cv2.resize(img_2, (150, 50), interpolation=cv2.INTER_LINEAR)
        img_1 = cv2.resize(img_1, (150, 50), interpolation=cv2.INTER_LINEAR)
        img_2_ = img_2.astype(np.float32)/255.
        img_1_ = img_1.astype(np.float32)/255.
        diff = img_2_ - img_1_
        diff = diff*0.5 + 0.5
        img = np.concatenate((img_2_, diff), axis=-1)
        return diff

    # Length of the Dataset
    def __len__(self):
        self.dataset_length = 128
        return self.dataset_length

    # Fetch an item from the Dataset
    def __getitem__(self, idx):
        state_gt = torch.tensor(self.dataset[idx][1], dtype=torch.float32)
        state_pre = torch.tensor(self.dataset[idx][0], dtype=torch.float32)
        obs = torch.tensor(self.dataset[idx][2], dtype=torch.float32)
        raw_obs = torch.tensor(self.preprocessing(self.dataset[idx], 
            mode='train'), dtype=torch.float32)

        state_gt = rearrange(state_gt, '(k dim) -> k dim', k=1)
        state_pre = rearrange(state_pre, '(k dim) -> k dim', k=1)
        obs = rearrange(obs, '(k dim) -> k dim', k=1)
        raw_obs = rearrange(raw_obs, 'h w c -> c h w')

        # apply the transformation
        state_gt = self.transform_.state_transform(state_gt).to(dtype=torch.float32)
        state_pre = self.transform_.state_transform(state_pre).to(dtype=torch.float32)
        obs = self.transform_.obs_transform(obs).to(dtype=torch.float32)

        state_ensemble = self.utils_.format_state(state_pre)

        return state_gt, state_pre, obs, raw_obs, state_ensemble

# if __name__ == '__main__':
#     dataset = KITTIDataset(32,5,2, 'train')
#     dataloader = torch.utils.data.DataLoader(dataset, batch_size=2,
#                                           shuffle=True, num_workers=1)
#     for state_gt, state_pre, obs, raw_obs, state_ensemble in dataloader:
#         print(state_ensemble.shape)
#         print("check -------- ",state_ensemble.dtype)
        # print(raw_obs.shape)
