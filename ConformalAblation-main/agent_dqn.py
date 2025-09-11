import datetime
import random
from collections import deque

import numpy as np
import torch

from constants import SAVE_DIR
from dqn_model import DQN


class Agent_DQN():
    def __init__(self, env, hyper_params, datetime_now):
        self.use_cuda = torch.cuda.is_available()
        print(f"Using CUDA: {self.use_cuda}")

        self.save_dir = SAVE_DIR + f"/{datetime_now}"

        self.env = env
        self.state_dim = self.env.observation_space.shape
        self.action_dim = self.env.action_space.n

        self.net = DQN(hyper_params["NUM_ACTIONS"], self.action_dim).float()
        if self.use_cuda:
            self.net = self.net.to(device="cuda")
        
        self.exploration_rate = hyper_params["EXP_RATE"]
        self.exploration_rate_min = hyper_params["EXP_RATE_MIN"]
        # self.exploration_step = 3e7
        # self.exploration_rate_decay = (1 - self.exploration_rate_min) / self.exploration_step
        self.exploration_rate_decay = hyper_params["EXP_DECAY"]

        self.memory_size = hyper_params["MEMORY_SIZE"]
        self.memory = deque(maxlen=self.memory_size)
        self.batch_size = hyper_params["BATCH_SIZE"]
        
        self.curr_step = 0
        self.cur_angle = 0

        self.gamma = hyper_params["GAMMA"]
        self.lr = hyper_params["LR"]
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        self.loss_fn = torch.nn.HuberLoss()

        self.burnin = hyper_params["BURNIN"]
        self.learn_every = hyper_params["LEARN_EVERY"]
        self.sync_every = hyper_params["SYNC_EVERY"]
        self.test_every = hyper_params["TEST_EVERY"]

    def load_trained_model(self, DQN_path):
        print(f'Loading trained model: {DQN_path}')
        self.net.load_state_dict(torch.load(DQN_path)["model"])
        self.net.eval()
        if self.use_cuda:
            self.net = self.net.to(device="cuda")

    def make_action(self, observation, test=False):
        # Test mode, make action using trained model.
        if test:
            state = observation.__array__()
            if self.use_cuda:
                state = torch.tensor(state).cuda()
            else:
                state = torch.tensor(state)
            state = state.unsqueeze(0)
            action_values = self.net(state, model="online")
            action = torch.argmax(action_values, axis=1).item()
            return action
        
        # Train, Explore.
        if np.random.rand() < self.exploration_rate:
            action = np.random.randint(self.action_dim)

        # Train, Expolit.
        else:
            state = observation.__array__()
            if self.use_cuda:
                state = torch.tensor(state).cuda()
            else:
                state = torch.tensor(state)
            state = state.unsqueeze(0)
            action_values = self.net(state, model="online")
            action = torch.argmax(action_values, axis=1).item()

        # decrease exploration_rate
        self.exploration_rate *= self.exploration_rate_decay
        # self.exploration_rate -= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        # increment step
        self.curr_step += 1

        return action
    
    def push(self, state, next_state, action, reward, done ):
        state = state.__array__()
        next_state = next_state.__array__()

        if self.use_cuda:
            state = torch.tensor(state).cuda()
            next_state = torch.tensor(next_state).cuda()
            action = torch.tensor([action]).cuda()
            reward = torch.tensor([reward]).cuda()
            done = torch.tensor([done]).cuda()
        else:
            state = torch.tensor(state)
            next_state = torch.tensor(next_state)
            action = torch.tensor([action])
            reward = torch.tensor([reward])
            done = torch.tensor([done])

        self.memory.append((state, next_state, action, reward, done))        
        
    def replay_buffer(self):
        batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze() 

    def td_estimate(self, state, action):
        current_Q = self.net(state.float(), model="online")[
            np.arange(0, self.batch_size), action
        ]  # Q_online(s,a)
        return current_Q

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_state_Q = self.net(next_state.float(), model="online")
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_state.float(), model="target")[
            np.arange(0, self.batch_size), best_action
        ]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()   

    def update_Q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()
        # return loss.item()

        # DQN gradient clipping
        self.optimizer.zero_grad()
        loss.backward()
       
        for param in self.net.online.parameters():
            param.grad.data.clamp_(-1, 1)

        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())

    def save(self):
        save_path = (self.save_dir + f"/net_{int(self.curr_step)}.chkpt")
        torch.save(
            dict(model=self.net.state_dict(), exploration_rate=self.exploration_rate),
            save_path,
        )
        with open(self.save_dir + "/log.txt", 'a') as f:
            f.write(f"Date: {datetime.datetime.now()}\n")
            f.write(f"DQN saved at step {self.curr_step}.\n")

    def learn(self):
        if self.curr_step < self.burnin:
            return None, None
        
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        # Sample from memory
        state, next_state, action, reward, done = self.replay_buffer()

        # Get TD Estimate
        td_est = self.td_estimate(state, action)

        # Get TD Target
        td_tgt = self.td_target(reward, next_state, done)

        # Backpropagate loss through Q_online
        loss = self.update_Q_online(td_est, td_tgt)

        return (td_est.mean().item(), loss)

