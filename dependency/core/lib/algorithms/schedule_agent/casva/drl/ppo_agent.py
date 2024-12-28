import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy
import os

from .network import Actor, Critic

__all__ = ('DualClippedPPO',)


# TODO: modify real PPO architecture (currently is soft actor-critic architecture)
class DualClippedPPO(object):
    def __init__(
            self,
            state_dims,
            action_dim,
            gamma=0.99,
            hid_shape=(256, 256),
            conv_kernel_size=(4, 4),
            conv_kernel_size_special=(1, 1),
            conv_hid_channels=(128,),
            conv_state_features=128,
            a_lr=3e-4,
            c_lr=3e-4,
            tau=0.005,
            batch_size=256,
            alpha=0.2,
            adaptive_alpha=True,
            device='cpu',
            **param
    ):
        # Initialize Actor and Critic networks
        self.actor = Actor(state_dims, action_dim, hid_shape,
                           conv_hid_channels, conv_kernel_size, conv_kernel_size_special,
                           conv_state_features).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=a_lr)

        self.q_critic = Critic(state_dims, action_dim, hid_shape,
                               conv_hid_channels, conv_kernel_size, conv_kernel_size_special,
                               conv_state_features).to(device)

        self.q_critic_optimizer = torch.optim.Adam(self.q_critic.parameters(), lr=c_lr)
        self.q_critic_target = copy.deepcopy(self.q_critic)
        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.q_critic_target.parameters():
            p.requires_grad = False

        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        self.alpha = alpha
        self.adaptive_alpha = adaptive_alpha
        if adaptive_alpha:
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            self.target_entropy = torch.tensor(-action_dim, dtype=torch.float, requires_grad=True, device=device)
            # We learn log_alpha instead of alpha to ensure exp(log_alpha)=alpha>0
            self.log_alpha = torch.tensor(np.log(alpha), dtype=torch.float, requires_grad=True, device=device)
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=c_lr)

        self.device = device

    def select_action(self, state, deterministic=False, with_logprob=False):
        # Select action based on the current policy
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            action, log_prob = self.actor(state, deterministic, with_logprob)
        return action.cpu().numpy().flatten()

    def train(self, replay_buffer):
        s, a, r, s_prime, dead_mask = replay_buffer.sample(self.batch_size)

        # ----------------------------- ↓↓↓↓↓ Update Q Net ↓↓↓↓↓ ------------------------------#
        with torch.no_grad():
            a_prime, log_pi_a_prime = self.actor(s_prime)
            target_Q1, target_Q2 = self.q_critic_target(s_prime, a_prime)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = r + (1 - dead_mask) * self.gamma * (
                    target_Q - self.alpha * log_pi_a_prime)  # Dead or Done is tackled by Randombuffer

        # Get current Q estimates
        current_Q1, current_Q2 = self.q_critic(s, a)

        q_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        self.q_critic_optimizer.zero_grad()
        q_loss.backward()
        self.q_critic_optimizer.step()

        # ----------------------------- ↓↓↓↓↓ Update Actor Net ↓↓↓↓↓ ------------------------------#
        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for params in self.q_critic.parameters():
            params.requires_grad = False

        a, log_pi_a = self.actor(s)
        current_Q1, current_Q2 = self.q_critic(s, a)
        Q = torch.min(current_Q1, current_Q2)

        a_loss = (self.alpha * log_pi_a - Q).mean()
        self.actor_optimizer.zero_grad()
        a_loss.backward()
        self.actor_optimizer.step()

        for params in self.q_critic.parameters():
            params.requires_grad = True
        # ----------------------------- ↓↓↓↓↓ Update alpha ↓↓↓↓↓ ------------------------------#
        if self.adaptive_alpha:
            # we optimize log_alpha instead of aplha, which is aimed to force alpha = exp(log_alpha)> 0
            # if we optimize aplpha directly, alpha might be < 0, which will lead to minimun entropy.
            alpha_loss = -(self.log_alpha * (log_pi_a + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp()

        # ----------------------------- ↓↓↓↓↓ Update Target Net ↓↓↓↓↓ ------------------------------#
        for param, target_param in zip(self.q_critic.parameters(), self.q_critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, save_dir, episode):
        assert os.path.exists(save_dir) and os.path.isdir(save_dir), f'Model saving directory "{save_dir}" error!'
        torch.save(self.actor.state_dict(), os.path.join(save_dir, f"ppo_actor_{episode}.pth"))
        torch.save(self.q_critic.state_dict(), os.path.join(save_dir, f"ppo_critic_{episode}.pth"))

    def load(self, load_dir, episode):
        assert os.path.exists(load_dir) and os.path.isdir(load_dir), f'Model loading directory "{load_dir}" error!'
        self.actor.load_state_dict(torch.load(os.path.join(load_dir, f"ppo_actor_{episode}.pth")))
        self.q_critic.load_state_dict(torch.load(os.path.join(load_dir, f"ppo_critic_{episode}.pth")))
