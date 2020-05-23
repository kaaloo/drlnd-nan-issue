import numpy as np
import random
from collections import namedtuple, deque

from dueling_model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim
import sys

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
ALPHA = 0.5             # amount of priorization to use
BETA = 0.4              # beta start value
BETA_ANNEALING = 0.001  # beta annealing factor
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate
UPDATE_EVERY = 4        # how often to update the network
PRIORITY_EPSILON = 1e-6  # Don't allow zero probabilities in replay buffer
GRAD_CLIP = 100.0       # Clip gradients to this value

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Torch device: {}".format(device))


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(
            state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(
            state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = PriorityReplayBuffer(
            action_size, state_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        # BETA hyperparameter needs to be annealed over time to reach 1.0
        self.beta = BETA

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones, indices, probs = experiences

        # Compute and minimize the loss

        sampling_weights = len(self.memory) * probs
        sampling_weights = np.power(sampling_weights, -self.beta)
        sampling_weights = sampling_weights / sampling_weights.max()
        sampling_weights = torch.from_numpy(sampling_weights).to(device)

        self.beta = self.beta + BETA_ANNEALING * self.beta
        self.beat = min(self.beta, 1.0)

        # Use Double DQN
        # Get max predicted Q values (for next states) from target model
        local_actions = self.qnetwork_local(
            next_states).detach().argmax(dim=1).unsqueeze(1)
        Q_targets_next = self.qnetwork_target(
            next_states).gather(1, local_actions).detach()

        # Compute Q targets for current states
        Q_targets = torch.clamp(rewards, min=-1.0, max=1.0) + \
            (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        elementwise_loss = F.smooth_l1_loss(
            Q_expected, Q_targets, reduction='none')
        loss = torch.mean(elementwise_loss * sampling_weights)

        self.optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_value_(
        #     self.qnetwork_local.parameters(), GRAD_CLIP)
        self.optimizer.step()

        # Update priorities
        priorities = elementwise_loss.detach().cpu().squeeze().numpy()
        self.memory.update(indices, priorities)

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(
                tau*local_param.data + (1.0-tau)*target_param.data)


class PriorityReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, state_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.buffer_size = buffer_size
        self.states = np.zeros((buffer_size, state_size))
        self.actions = np.zeros(buffer_size)
        self.rewards = np.zeros(buffer_size)
        self.next_states = np.zeros((buffer_size, state_size))
        self.dones = np.zeros(buffer_size)
        self.priorities = np.zeros(buffer_size)
        self.ptr = 0
        self.full = False
        self.max_p = 1.0  # Initialize maximum priority
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=[
                                     "state", "action", "reward", "next_state", "done"])
        random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        self.priorities[self.ptr] = self.max_p ** ALPHA

        self.ptr = self.ptr + 1
        if self.ptr == self.buffer_size and not self.full:
            self.full = True
        self.ptr = self.ptr % self.buffer_size

    def update(self, indices, priorities):
        self.priorities[indices] = np.power(priorities, ALPHA)
        self.max_p = np.max(priorities)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        probs = self.priorities[0:len(self)]
#        probs = np.nan_to_num(probs)
        probs = probs + PRIORITY_EPSILON
        probs = probs / probs.sum()

        indices = np.random.choice(
            len(self), size=self.batch_size, p=probs)

        actions = torch.from_numpy(
            self.actions[indices]).long().unsqueeze(1).to(device)
        states = torch.from_numpy(self.states[indices]).float().to(device)
        rewards = torch.from_numpy(
            self.rewards[indices]).float().unsqueeze(1).to(device)
        next_states = torch.from_numpy(
            self.next_states[indices]).float().to(device)
        dones = torch.from_numpy(
            self.dones[indices]).float().unsqueeze(1).to(device)

        return (states, actions, rewards, next_states, dones, indices, probs)

    def __len__(self):
        """Return the current size of internal memory."""
        return self.buffer_size if self.full else self.ptr
