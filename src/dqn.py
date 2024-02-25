from utils import DQN, ReplayBuffer
from copy import deepcopy
import torch.nn as nn
import torch
import numpy as np
from evaluate import evaluate_HIV

class DQNAgent:
    def __init__(self):
        self.config = {
            "device": 'cuda:0',
            "training_steps": 40000,
            "ep_length": 200,
            "batch_size": 512,
            "learning_rate": 1e-3,
            "gradient_steps": 3,
            "epsilon_min": 0.01,
            "epsilon_max": 1.,
            "epsilon_decay_period": 17000,
            "epsilon_delay_decay": 500,
            "gamma": 0.95,
            "update_target_tau":0.001,
            "update_target_replace": 400,
            "update_replace": False,
            "buffer_size": 100000,
            "log_delay": 50,
            "save_delay": 1000
        }
        self.config['epsilon_step'] =(self.config['epsilon_max']-self.config['epsilon_min'])/self.config['epsilon_decay_period']

        self.dqn = DQN()
        self.target_dqn = deepcopy(self.dqn)
        self.replay_buffer = ReplayBuffer(self.config['buffer_size'], self.config['device'])
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.dqn.parameters(), lr=self.config['learning_rate'])
        self.max_update = 1e16
        self.max_reward = 1e8

    def sample_action_greedy(self, s):
        with torch.no_grad():
            s = torch.tensor(s, device=self.config['device'], dtype=torch.float32)
            self.dqn.eval()
            Q = self.dqn(s)
            self.dqn.train()
        return torch.argmax(Q).cpu().item()
    
    def sample_action_eps_greedy(self, env, s, eps):
        e = np.random.sample()
        if e>eps:
            return self.sample_action_greedy(s[None, :]), 0
        return env.action_space.sample(), 1
    
    def update_target(self, step):
        if self.config['update_replace'] and step % self.config['update_target_replace'] == 0:
            self.target_dqn = deepcopy(self.dqn)
            print("Update Target")
        else:
            target_state_dict = self.target_dqn.state_dict()
            model_state_dict = self.dqn.state_dict()
            tau = self.config["update_target_tau"]
            for key in model_state_dict:
                target_state_dict[key] = tau*model_state_dict[key] + (1-tau)*target_state_dict[key]
            self.target_dqn.load_state_dict(target_state_dict)

    def act(self, observation, use_random=False):
        return self.sample_action_greedy(observation[None, :])

    def save(self, path):
        torch.save({'model_state_dict': self.dqn.state_dict()}, path)

    def load(self):
        self.config['device'] = 'cpu'
        base_path = "dqn_best.pth"
        chkpt = torch.load(base_path, map_location=torch.device('cpu'))
        self.dqn.load_state_dict(chkpt['model_state_dict'])
        self.dqn.to(self.config['device'])
        self.dqn.eval()
        self.target_dqn =  deepcopy(self.dqn).to(self.config['device'])

    def gradient_step(self):
        if len(self.replay_buffer) >= self.config['batch_size']:
            e = 1e-3
            samples = self.replay_buffer.sample(self.config['batch_size'])
            s, a, r, sp, d = samples
            QYmax = self.target_dqn(sp).max(1)[0].detach()
            # QYmax = QYmax*self.max_update
            # QYmax = torch.exp(QYmax)
            # QYmax = torch.log(QYmax + e)
            # r = torch.log(torch.abs(r + e))
            # print(r)
            # r = r/self.max_reward
            # print("reward: ", r)
            update = torch.addcmul(r, 1-d, QYmax, value=self.config["gamma"])
            # self.max_update = np.max((np.max(update.cpu().detach().numpy()), self.max_update))
            # print(self.max_update)
            # print('update before: ', update)
            # update = update / self.max_update
            # update = torch.log(update)
            s.requires_grad = True
            QXA = self.dqn(s).gather(1, a.to(torch.long).unsqueeze(1))
            # QXA = torch.log(QXA + e)
            # print('update after: ', update)
            loss = self.criterion(QXA, update.unsqueeze(1))
            # print(loss)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step() 
            return loss.detach()
        return 0 
    
    def train(self, env):
        self.dqn.to(self.config['device'])
        self.dqn.train()
        self.target_dqn.to(self.config['device'])
        s, _ = env.reset()
        cum_reward= 0
        epsilon = self.config['epsilon_max']
        random_actions = 0
        losses = 0
        best_score = 0
        for step in range(self.config['training_steps']):
            # Choose action
            if step > self.config['epsilon_delay_decay']:
                epsilon = max(self.config['epsilon_min'], epsilon-self.config['epsilon_step'])
            a, rand_action = self.sample_action_eps_greedy(env, s, epsilon)
            random_actions += rand_action
            sp, r, d, t, _ = env.step(a)
            cum_reward+=r
            # Fill buffer
            self.replay_buffer.append((s, a, r, sp, d))

            # Gradient step
            for _ in range(self.config['gradient_steps']):
                losses += self.gradient_step()

            # Update target
            self.update_target(step)

            if (step+1) % self.config['log_delay']==0:
                rolling_reward = self.replay_buffer.rolling_reward(self.config['log_delay'])
                print(f'step {step+1} | rolling reward: {"{:e}".format(rolling_reward)} | random actions: {random_actions} | loss: {"{:e}".format(losses/self.config["log_delay"])}')
                random_actions = 0
                losses = 0
            
            if step % self.config['save_delay'] == 0:
                self.save(f'models/dqn_{step}.pth')

            if step % self.config['ep_length'] == 0:
                val_score = evaluate_HIV(agent = self, nb_episode=1)
                print(f" == End of episode. Score: {'{:e}'.format(val_score)} ==")
                if val_score > best_score:
                    self.save(f'dqn_best.pth')
                    best_score = val_score
                s, _ = env.reset()
            else:
                s = sp
        print(f"Total reward {cum_reward}")