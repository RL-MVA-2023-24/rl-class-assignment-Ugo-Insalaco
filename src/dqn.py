from utils import DQN, ReplayBuffer
from copy import deepcopy
import torch.nn as nn
import torch
import numpy as np

class DQNAgent:
    def __init__(self):
        self.config = {
            "device": 'cuda:0',
            "training_steps": 10000,
            "batch_size": 32,
            "learning_rate": 1e-3,
            "epsilon_greedy": 0.1,
            "gamma": 0.95,
            "update_target_tau":0.03,
            "buffer_size": 2000,
            "log_delay": 50,
        }
        self.dqn = DQN()
        self.target_dqn = deepcopy(self.dqn)
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
        self.config['device'] = 'cpu'
        base_path = "dqn.pth"
        chkpt = torch.load(base_path, map_location=torch.device('cpu'))
        self.dqn.load_state_dict(chkpt['model_state_dict'])
        self.dqn.to(self.config['device'])
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
        self.dqn.to(self.config['device'])
        self.target_dqn.to(self.config['device'])
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
                rolling_reward = self.replay_buffer.rolling_reward(self.config['log_delay'])
                print(f'step {step+1} | rolling reward: {rolling_reward}')
            
            s = sp
        print(f"Total reward {cum_reward}")