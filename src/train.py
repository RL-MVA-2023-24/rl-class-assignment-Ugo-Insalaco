from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from copy import deepcopy

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!
class DQN(nn.Module):
    def __init__(self, device='cpu'):
        self.in_features = 6
        self.out_size = 4
        self.hidden_size = 256
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(self.in_features, self.hidden_size, device=device)
        self.fc2 = nn.Linear(self.hidden_size, self.out_size, device=device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
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

    @property
    def rolling_reward(self):
        r = [d[2] for d in self.data]
        return sum(r)/len(self.data)
    
    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        return list(map(lambda x:torch.Tensor(np.array(x)).to(self.device), list(zip(*batch))))
    def __len__(self):
        return len(self.data)
    
class ProjectAgent:
    def __init__(self):
        self.config = {
            "device": 'cuda:0',
            "training_steps": 5000,
            "batch_size": 32,
            "learning_rate": 1e-3,
            "epsilon_greedy": 0.1,
            "gamma": 0.95,
            "update_target_tau":0.05,
            "buffer_size": 1000,
            "log_delay": 50,
        }
        self.dqn = DQN(self.config['device'])
        self.target_dqn = deepcopy(self.dqn).to(self.config['device'])
        self.replay_buffer = ReplayBuffer(self.config['buffer_size'], self.config['device'])
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.dqn.parameters(), lr=self.config['learning_rate'])

    def sample_action_greedy(self, s):
        with torch.no_grad():
            s = torch.tensor(s, device=self.config['device'], dtype=torch.float32)
            Q = self.dqn(s)
        return torch.argmax(Q).cpu().item()
    
    def sample_action_eps_greedy(self, env, s, eps):
        e = np.random.sample()
        if e>eps:
            return self.sample_action_greedy(s)
        return env.action_space.sample()
    
    def update_target(self):
        target_state_dict = self.target_dqn.state_dict()
        model_state_dict = self.dqn.state_dict()
        tau = self.config["update_target_tau"]
        for key in model_state_dict:
            target_state_dict[key] = tau*model_state_dict[key] + (1-tau)*target_state_dict[key]
        self.target_dqn.load_state_dict(target_state_dict)

    def act(self, observation, use_random=False):
        return self.sample_action_greedy(observation)

    def save(self, path):
        torch.save({'model_state_dict': self.dqn.state_dict()}, path)

    def load(self):
        base_path = "dqn.pth"
        chkpt = torch.load(base_path)
        self.dqn.load_state_dict(chkpt['model_state_dict'])
        self.dqn.eval()
        self.target_dqn =  deepcopy(self.dqn).to(self.config['device'])

    def gradient_step(self):
        if len(self.replay_buffer) >= self.config['batch_size']:
            samples = self.replay_buffer.sample(self.config['batch_size'])
            s, a, r, sp, d = samples
            QYmax = self.target_dqn(sp).max(1)[0].detach()
            update = torch.addcmul(r, 1-d, QYmax, value=self.config["gamma"])
            QXA = self.dqn(s).gather(1, a.to(torch.long).unsqueeze(1))
            loss = self.criterion(QXA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step() 
            
    def train(self, env):
        s, _ = env.reset()
        cum_reward= 0
        for step in range(self.config['training_steps']):
            # Choose action
            # print(s)
            a = self.sample_action_eps_greedy(env, s, self.config['epsilon_greedy'])
            sp, r, d, t, _ = env.step(a)
            cum_reward+=r
            # Fill buffer
            self.replay_buffer.append((s, a, r, sp, d))

            # Gradient step
            self.gradient_step()

            # Update target
            self.update_target()

            if (step+1) % self.config['log_delay']==0:
                print(f'step {step+1} | rolling reward: {self.replay_buffer.rolling_reward}')
            
            s = sp
        print(f"Total reward {cum_reward}")
if __name__ == '__main__':
    agent = ProjectAgent()
    # agent.load()
    agent.train(env)
    agent.save('dqn.pth')
