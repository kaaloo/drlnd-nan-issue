import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64, head_units=32):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        # trunk
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)

        # adv head
        self.adv1 = nn.Linear(fc2_units, head_units)
        self.adv2 = nn.Linear(head_units, action_size)

        # val head
        self.val1 = nn.Linear(fc2_units, head_units)
        self.val2 = nn.Linear(head_units, 1)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        # trunk
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        # adv head
        adv = F.relu(self.adv1(x))
        adv = self.adv2(adv)

        # val head
        val = F.relu(self.val1(x))
        val = self.val2(val)

        return val.expand(adv.size()) + adv - adv.mean(dim=1, keepdim=True)
