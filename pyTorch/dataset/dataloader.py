import os
import random
from PIL import Image
import cv2
import numpy as np
from torch.utils.data import Dataset
import pickle
import torch
from einops import rearrange, repeat
from torch.distributions.multivariate_normal import MultivariateNormal
import math
import pdb


class CarDataset(Dataset):
    # Basic Instantiation
    def __init__(self, args, mode):
        self.args = args
        self.mode = mode
        if self.mode == "train":
            self.dataset_path = self.args.train.data_path
            self.num_ensemble = self.args.train.num_ensemble
        elif self.mode == "test":
            self.dataset_path = self.args.test.data_path
            self.num_ensemble = self.args.test.num_ensemble
        self.dataset = pickle.load(open(self.dataset_path, "rb"))
        self.dataset_length = len(self.dataset)
        self.dim_x = self.args.train.dim_x
        self.dim_z = self.args.train.dim_z
        self.dim_a = self.args.train.dim_a
        self.win_size = self.args.train.win_size

    def process_image(self, img_path):
        img_array = cv2.imread(img_path)
        img_array = cv2.resize(img_array, (224, 224))
        img_array = img_array / 255.0
        return img_array

    # Length of the Dataset
    def __len__(self):
        # self.dataset_length = 50
        return self.dataset_length - self.win_size - 2

    # Fetch an item from the Dataset
    def __getitem__(self, idx):
        # make sure always take the data from the same sequence
        not_valid = True
        while not_valid:
            try:
                if self.dataset[idx][0] == self.dataset[idx + self.win_size][0]:
                    not_valid = False
                else:
                    idx = random.randint(0, self.dataset_length)
            except:
                idx = random.randint(0, self.dataset_length)

        # the observation to the model
        obs_now = self.dataset[idx + self.win_size][2]
        scaler = 224 / 90
        x = int(self.dataset[idx + self.win_size][2][0] * scaler - 10)
        y = int(self.dataset[idx + self.win_size][2][1] * scaler)
        obs_now[:2] = obs_now[:2] / 85.0
        obs_now = torch.tensor(obs_now, dtype=torch.float32)
        obs_now = rearrange(obs_now, "(k dim) -> k dim", k=1)

        # a stack of image to the model
        images = []
        for i in range(self.win_size):
            img_path = "./dataset" + self.dataset[idx + i][3]
            img_array = self.process_image(img_path)
            images.append(img_array)
        images = np.array(images)
        images = torch.tensor(images, dtype=torch.float32)
        images = rearrange(images, "k h w ch -> k ch h w")

        # gt image
        img_path = "./dataset" + self.dataset[idx + self.win_size][3]
        gt_image = self.process_image(img_path)

        gt_image = torch.tensor(gt_image, dtype=torch.float32)
        gt_image = rearrange(gt_image, "h w ch -> ch h w")

        return obs_now, images, gt_image
