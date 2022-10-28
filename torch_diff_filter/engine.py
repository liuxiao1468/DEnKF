import torch
import torch.nn as nn
from dataset import KITTIDataset
from model import Ensemble_KF
from torch.utils.tensorboard import SummaryWriter

class Engine():
    def __init__(self):
        self.batch_size = 2
        self.dim_x = 5
        self.dim_z = 2
        self.num_ensemble = 32

        self.dataset = KITTIDataset()
        self.model = Ensemble_KF(self.num_ensemble, self.dim_x, self.dim_z)

    def train(self):
        loss = nn.MSELoss()
        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size = self.batch_size,
                                          shuffle=True, num_workers=1)
        pytorch_total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('Total number of parameters: ',pytorch_total_params)
        print(list(self.model.children())[0])
        # for state_gt, state_pre, obs, raw_obs in dataloader:
        #     # define the training curriculum

if __name__ == '__main__':
    train_engine = Engine()
    train_engine.train()

