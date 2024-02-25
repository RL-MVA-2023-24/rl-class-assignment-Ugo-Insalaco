from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
from dqn import DQNAgent
from ddqn import DDQNAgent
import numpy as np

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!
    
    
class ProjectAgent:
    def __init__(self):
        self.agent_name = "DQN"
        if self.agent_name=="DQN":
            self.agent = DQNAgent() 
        elif self.agent_name=="DDQN":
            self.agent = DDQNAgent() 

    def act(self, observation, use_random=False):
        return self.agent.act(observation, use_random)

    def save(self, path):
        self.agent.save(path)

    def load(self):
        self.agent.load()
            
    def train(self, env):
        return self.agent.train(env)

if __name__ == '__main__':
    agent = ProjectAgent()
    # agent.load()
    rs = agent.train(env)
    # np.save('training.npy', rs)
