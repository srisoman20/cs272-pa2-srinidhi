import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Actor-Critic model
# - Actor: outputs action probabilities
# - Critic: estimates value of a state
class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()

        self.shared = nn.Sequential(  # shared feature layer (extracts useful info from state)
            nn.Linear(state_size, 128),
            nn.ReLU(),
        )

        self.actor = nn.Sequential(  # actor head; outputs probability distribution over actions
            nn.Linear(128, action_size),
            nn.Softmax(dim=-1)
        )

        self.critic = nn.Linear(128, 1) # critic head; outputs state value

    def forward(self, x):
        x = self.shared(x)
        return self.actor(x), self.critic(x)


class ACAgent:
    def __init__(self, state_size, action_size):
        self.model = ActorCritic(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.gamma = 0.99

    def select_action(self, state):  # select action based on current state
        state = torch.FloatTensor(state)
        probs, _ = self.model(state)
        dist = torch.distributions.Categorical(probs)  # create a categorical distribution and sample action
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def update(self, state, log_prob, reward, next_state, done):  # update model using Actor-Critic learning
        state = torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state)

        _, value = self.model(state)
        _, next_value = self.model(next_state)

        value = value.squeeze()
        next_value = next_value.squeeze()

        # TD target
        target = reward + self.gamma * next_value * (1 - int(done))

        # TD error
        delta = target - value

        # losses
        actor_loss = -log_prob * delta.detach()
        critic_loss = delta ** 2

        loss = actor_loss + critic_loss

        # backpropogation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()