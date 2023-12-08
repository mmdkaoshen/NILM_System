import numpy as np
import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, data):
        aggregation = np.array(data)

        self.main_power = torch.from_numpy(aggregation[:, 0])
        self.appliance = torch.from_numpy(aggregation[:, 1:])

        self.main_power = self.main_power.float()
        self.appliance = self.appliance.float()

        # main_mean = torch.mean(self.main_power)
        # main_std = torch.std(self.main_power)
        # appliance_mean = torch.mean(self.appliance, dim=(-3, -1), keepdim=True)
        # appliance_std = torch.std(self.appliance, dim=(-3, -1), keepdim=True)

        # self.main_power = (self.main_power - main_mean) / main_std
        # self.appliance = (self.appliance - appliance_mean) / appliance_std

        self.main_power /= 4000
        self.appliance /= 4000

        self.num = self.main_power.shape[0]

    def __getitem__(self, index):
        return self.main_power[index], self.appliance[index]

    def __len__(self):
        return self.num
