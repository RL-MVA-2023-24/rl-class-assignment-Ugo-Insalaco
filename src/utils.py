import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np

class DQN(nn.Module):
    def __init__(self, device='cpu'):
        self.in_features = 6
        self.out_size = 4
        self.hidden_size = 128
        self.hidden_size2 = 64
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(self.in_features, self.hidden_size, device=device)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size2, device=device)
        self.fc3 = nn.Linear(self.hidden_size2, self.out_size, device=device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class ReplayBuffer:
    def __init__(self, capacity, device):
        self.device = device
        self.capacity = capacity # capacity of the buffer
        self.data = []
        self.index = 0 # index of the next cell to be filled
    def append(self, sars): 
        # sars: tuple
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = sars
        self.index = (self.index + 1) % self.capacity

    def rolling_reward(self, time):
        r = [self.data[i][2] for i in range(self.index - time, self.index)]
        return sum(r)/time
    
    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        return list(map(lambda x:torch.Tensor(np.array(x)).to(self.device), list(zip(*batch))))
    def __len__(self):
        return len(self.data)