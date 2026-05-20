import torch
import torch.nn as nn
import torch.nn.functional as F


class ChessCNN(nn.Module):

    def __init__(self):
        super().__init__()

        # CNN trunk
        self.conv1 = nn.Conv2d(13, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        # VALUE HEAD
        self.value_fc1 = nn.Linear(128 * 8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)

        # POLICY HEAD
        self.policy_conv = nn.Conv2d(128, 32, kernel_size=1)
        self.policy_fc = nn.Linear(32 * 8 * 8, 4672)

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # VALUE HEAD
        value = x.view(x.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))

        # POLICY HEAD
        policy = F.relu(self.policy_conv(x))
        policy = policy.view(policy.size(0), -1)
        policy = self.policy_fc(policy)

        return value, policy