import torch
from torch.distributions import Normal
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from itertools import accumulate

from .net_utils import build_net, build_conv1d_net
from .utils import format_input_state


class SpecificConvFeatureExtractor(nn.Module):
    def __init__(self, input_channels, hid_channels, kernel_size,
                 input_series_length, output_features, activation=nn.ReLU):
        super(SpecificConvFeatureExtractor, self).__init__()
        self.conv = build_conv1d_net(input_channels, hid_channels, kernel_size, activation)
        self.fc = nn.Linear(self._get_net_out((input_channels, input_series_length)),
                            output_features)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def _get_net_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.view(1, -1).size()))


class FeatureExtractor(nn.Module):
    def __init__(self, state_num_groups, window_size, hid_channels, kernel_size, kernel_size_special, state_features, activation):
        super(FeatureExtractor, self).__init__()
        self.state_indexes = list(accumulate(state_num_groups))
        self.state_num_groups = state_num_groups
        self.extractors = nn.ModuleList([
            SpecificConvFeatureExtractor(
                input_channels=state_num,
                hid_channels=hid_channels,
                kernel_size=kernel_size if state_num==1 else kernel_size_special,
                input_series_length=window_size,
                output_features=state_features * state_num,
                activation=activation)
            for state_num in self.state_num_groups
        ])

    def forward(self, state):
        state_splits = [state[:, start:end, :]
                        for start, end in zip([0] + self.state_indexes[:-1], self.state_indexes)]
        state_features = []
        for extractor, group_state in zip(self.extractors, state_splits):
            state_features.append(extractor(group_state))
        return torch.cat(state_features, dim=1)


class Actor(nn.Module):
    def __init__(self, state_dims, action_dim, hid_shape,
                 conv_hid_channels, conv_kernel_size,conv_kernel_size_special, conv_state_feature,
                 conv_activation=nn.ReLU, h_activation=nn.ReLU, o_activation=nn.ReLU):
        super(Actor, self).__init__()

        self.feature_extractor = FeatureExtractor(
            state_num_groups=state_dims[0],
            window_size=state_dims[1],
            hid_channels=conv_hid_channels,
            kernel_size=conv_kernel_size,
            kernel_size_special=conv_kernel_size_special,
            state_features=conv_state_feature,
            activation=conv_activation,
        )
        layers = [self._get_net_out((sum(state_dims[0]), state_dims[1]))] + list(hid_shape)
        self.a_net = build_net(layers, h_activation, o_activation)
        self.mu_layer = nn.Linear(layers[-1], action_dim)
        self.log_std_layer = nn.Linear(layers[-1], action_dim)

        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -20

    def forward(self, state, deterministic=False, with_logprob=True):
        """Network with Enforcing Action Bounds"""

        features = format_input_state(state)
        state_features = self.feature_extractor(features)

        net_out = self.a_net(state_features)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)  # 总感觉这里clamp不利于学习
        std = torch.exp(log_std)
        dist = Normal(mu, std)

        if deterministic:
            u = mu
        else:
            u = dist.rsample()  # '''reparameterization trick of Gaussian'''#
        a = torch.tanh(u)

        if with_logprob:
            # get probability density of logp_pi_a from probability density of u, which is given by the original paper.
            # logp_pi_a = (dist.log_prob(u) - torch.log(1 - a.pow(2) + 1e-6)).sum(dim=1, keepdim=True)

            # Derive from the above equation. No a, thus no tanh(h), thus less gradient vanish and more stable.
            logp_pi_a = dist.log_prob(u).sum(axis=1, keepdim=True) - (2 * (np.log(2) - u - F.softplus(-2 * u))).sum(
                axis=1, keepdim=True)
        else:
            logp_pi_a = None

        return a, logp_pi_a

    def _get_net_out(self, shape):
        o = self.feature_extractor(torch.zeros(1, *shape))
        return int(np.prod(o.view(1, -1).size()))


class Critic(nn.Module):
    def __init__(self, state_dims, action_dim, hid_shape,
                 conv_hid_channels, conv_kernel_size, conv_kernel_size_special, conv_state_feature,
                 conv_activation=nn.ReLU, h_activation=nn.ReLU, o_activation=nn.Identity):
        super(Critic, self).__init__()

        self.feature_extractor = FeatureExtractor(
            state_num_groups=state_dims[0],
            window_size=state_dims[1],
            hid_channels=conv_hid_channels,
            kernel_size=conv_kernel_size,
            kernel_size_special = conv_kernel_size_special,
            state_features=conv_state_feature,
            activation=conv_activation
        )

        layers = [self._get_net_out((sum(state_dims[0]), state_dims[1])) + action_dim] + list(hid_shape) + [1]

        self.Q_1 = build_net(layers, h_activation, o_activation)
        self.Q_2 = build_net(layers, h_activation, o_activation)

    def forward(self, state, action):
        format_input_state(state)

        state_features = self.feature_extractor(state)

        sa = torch.cat([state_features, action], 1)
        q1 = self.Q_1(sa)
        q2 = self.Q_2(sa)
        return q1, q2

    def _get_net_out(self, shape):
        o = self.feature_extractor(torch.zeros(1, *shape))
        return int(np.prod(o.view(1, -1).size()))
