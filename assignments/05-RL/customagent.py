import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """
    Q-value Network
    """

    def __init__(self, n_features, n_actions):
        """
        Initialize the QNetwork
        Arguments:
            n_features: The number of observation features.
            n_actions: The number of actions.
        """
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(in_features=n_features, out_features=40)
        self.fc2 = nn.Linear(in_features=40, out_features=n_actions)

    def forward(self, x):
        """
        Forward pass of the network
        Arguments:
            x: The input data.
        Returns:
            The output of the network.
        """
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Agent:
    """
    Q-learning Agent
    """

    def __init__(
        self, action_space: gym.spaces.Discrete, observation_space: gym.spaces.Box
    ):
        """
        Initialize the agent
        Arguments:
            action_space: Space of actions.
            observation_space: Space of observations.
        """
        self.action_space = action_space
        self.observation_space = observation_space
        self.q_net = QNetwork(observation_space.shape[0], action_space.n)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=0.01)
        self.last_observation = None
        self.last_action = None
        self.epsilon = 0.01
        self.discount = 0.9
        self.episodes = 0

    def act(self, observation: gym.spaces.Box) -> gym.spaces.Discrete:
        """
        Compute the action given the observation
        Arguments:
            observation: an observation.
        Returns:
            An action.
        """
        self.last_observation = observation
        if torch.rand(1) < self.epsilon:
            self.last_action = self.action_space.sample()
            return self.last_action
        with torch.no_grad():
            q_pred = self.q_net(torch.from_numpy(observation).type(torch.float32))
            if self.episodes < 1000:
                p = torch.softmax(q_pred, 0).numpy()
                self.last_action = np.random.choice(self.action_space.n, p=p)
            else:
                self.last_action = q_pred.argmax()
            return self.last_action

    def learn(
        self,
        observation: gym.spaces.Box,
        reward: float,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """
        Do one step of Q-learning
        Arguments:
            observation: an observation
            reward: a reward
            terminated: whether the episode has terminated
            truncated: whether the episode was truncated
        """
        if self.last_observation is None:
            self.last_observation = observation
            return
        q_pred = self.q_net(
            torch.from_numpy(self.last_observation).type(torch.float32)
        )[self.last_action]
        with torch.no_grad():
            q_target = self.q_net(
                torch.from_numpy(observation).type(torch.float32)
            ).max()
            q_target = reward + self.discount * q_target
        loss = F.mse_loss(q_pred, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.last_observation = observation
        if terminated or truncated:
            self.last_observation = None
            self.last_action = None
            self.episodes += 1
