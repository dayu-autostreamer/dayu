from .base_switch import BaseSwitch
import time
import threading
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import namedtuple, deque
import os

# Define experience tuples
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'log_prob', 'value'))

class LSTMActorCritic(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMActorCritic, self).__init__()
        
        # Separate LSTM layers: one for the Actor and one for the Critic
        self.actor_lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.critic_lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        
        # Actor Network (Strategy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic Network (Value)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Hidden state initialization
        self.actor_hidden = None
        self.critic_hidden = None
        
    def reset_hidden(self, batch_size=1):
        """Reset the hidden state of LSTM"""
        device = next(self.parameters()).device
        self.actor_hidden = (
            torch.zeros(1, batch_size, self.actor_lstm.hidden_size).to(device),
            torch.zeros(1, batch_size, self.actor_lstm.hidden_size).to(device)
        )
        self.critic_hidden = (
            torch.zeros(1, batch_size, self.critic_lstm.hidden_size).to(device),
            torch.zeros(1, batch_size, self.critic_lstm.hidden_size).to(device)
        )
        
    def forward(self, x, reset_hidden=False):
        """Forward propagation"""
        batch_size = x.size(0)
        
        # Reset hidden states (if needed)
        if reset_hidden or self.actor_hidden is None or self.critic_hidden is None:
            self.reset_hidden(batch_size)
        
        # Ensure hidden-state batch size matches the input batch size
        if self.actor_hidden[0].size(1) != batch_size:
            self.reset_hidden(batch_size)
            
        # Actor LSTM forward pass
        actor_lstm_out, self.actor_hidden = self.actor_lstm(x, self.actor_hidden)
        actor_features = actor_lstm_out[:, -1, :]  # Use the last time step output

        # Critic LSTM forward pass
        critic_lstm_out, self.critic_hidden = self.critic_lstm(x, self.critic_hidden)
        critic_features = critic_lstm_out[:, -1, :]  # Use the last time step output

        # Compute action probabilities and state value
        action_probs = self.actor(actor_features)
        state_value = self.critic(critic_features)
        
        return action_probs, state_value
    
    def act(self, state, exploration_rate=0.0):
        """Select an action based on the current state."""
        # Make sure hidden states match the current input batch size
        self.reset_hidden(batch_size=state.size(0))
        
        # Forward pass to get action probabilities and the state value
        action_probs, state_value = self(state)
        
        print(f"Action probs: {action_probs}")
        
        # Apply exploration - sometimes choose a random action
        if np.random.random() < exploration_rate:
            action_idx = torch.randint(0, action_probs.size(-1), (1,)).item()
            log_prob = torch.log(action_probs[0, action_idx])
            return action_idx, log_prob, state_value
        
        # Create a categorical distribution and sample from it
        dist = Categorical(action_probs)
        action = dist.sample()
        
        # Return action index, log probability, and state value
        return action.item(), dist.log_prob(action), state_value


class ACSwitch(BaseSwitch):
    def __init__(self, decision_interval: int, 
                 detector_instance: object,
                 queue_low_threshold_length: int = 10,
                 state_history_length: int = 5,
                 hidden_dim: int = 32,
                 *args, **kwargs):
        """
        LSTM Actor-Critic model switcher.

        Args:
            decision_interval: Time interval (seconds) between model switch decisions.
            detector_instance: Detector instance.
            queue_low_threshold_length: Queue length threshold used for reward calculation.
            state_history_length: Time steps for the LSTM.
            hidden_dim: Hidden dimension size of the LSTM.

        """
        # Basic attribute initialization
        self.models_num = detector_instance.get_models_num()
        self.decision_interval = decision_interval
        self.detector_instance = detector_instance
        self.queue_low_threshold_length = queue_low_threshold_length
        self.state_history_length = state_history_length
        
        # Queue threshold parameters
        self.queue_high_threshold_length = 2 * self.queue_low_threshold_length
        
        # Currently used model
        self.current_model_index = None
        
        # Feature-related
        self.input_dim = 11  # Number of features in StatsEntry excluding timestamp

        # Decision-making and learning related parameters
        self.hidden_dim = hidden_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Exploration parameters
        self.exploration_rate = 0.9  # Initial exploration rate
        self.min_exploration_rate = 0.1  # Minimum exploration rate
        self.exploration_decay = 0.999  # Exploration decay factor

        # Initialize LSTM Actor-Critic network
        self.network = LSTMActorCritic(
            input_dim=self.input_dim, 
            hidden_dim=self.hidden_dim, 
            output_dim=self.models_num
        ).to(self.device)
        
        # Use separate optimizers
        self.actor_optimizer = optim.Adam([p for n, p in self.network.named_parameters()
                                          if 'actor' in n or 'actor_lstm' in n], lr=0.005)
        self.critic_optimizer = optim.Adam([p for n, p in self.network.named_parameters() 
                                           if 'critic' in n or 'critic_lstm' in n], lr=0.001)
        
        # Track the last chosen action and state
        self.previous_action = None
        self.previous_state = None
        self.previous_log_prob = None
        self.previous_value = None
        self.last_switch_time = time.time()
        
        # Experience replay
        self.replay_buffer = deque(maxlen=100)
        self.min_samples_before_update = 8
        self.batch_size = 8
        self.update_frequency = 1
        self.update_counter = 0
        
        # Actor-Critic hyperparameters
        self.gamma = 0.5  # Discount factor
        self.entropy_beta = 0.1  # Entropy regularization coefficient
        self.critic_loss_coef = 0.5  # Value function loss coefficient

        # Statistics
        self.episodes = 0
        self.training_steps = 0

        # Model save path
        self.model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
        os.makedirs(self.model_dir, exist_ok=True)
        self.model_path = os.path.join(self.model_dir, "lstm_ac_switch_model.pt")
        
        # Try to load pretrained model
        self.load_model(self.model_path)
        
        # Start the decision thread
        self.switch_thread = threading.Thread(target=self._switch_loop)
        self.switch_thread.daemon = True
        self.switch_thread.start()

        print(f"LSTM Actor-Critic Switch initialized with {self.models_num} models, device: {self.device}")

    def _switch_loop(self):
        """Main loop for model switch decision making"""
        while True:
            try:
                current_time = time.time()
                
                # If the decision interval time has been reached
                if current_time - self.last_switch_time > self.decision_interval:
                    # Get interval statistics for LSTM input
                    stats_list = self.get_detector_interval_stats(nums=self.state_history_length, interval=1.0)
                    
                    # Get current statistics to calculate reward
                    current_stats = self.get_detector_stats()
                    
                    # Check for emergency response
                    emergency_mode = False
                    if current_stats and current_stats[0]:
                        stats_dict = self.stats_entry_to_dict(current_stats[0])
                        if stats_dict.get('queue_length', 0) >= self.queue_high_threshold_length:
                            emergency_mode = True
                            # Select the lightest model (index 0)
                            self.handle_emergency()
                            time.sleep(self.decision_interval)
                            continue
                    
                    if stats_list and len(stats_list) > 0:
                        # Preprocess the statistics sequence for network input
                        current_state = self.preprocess_stats(stats_list)
                        
                        # If there is a previous action and state, calculate reward and learn
                        if self.previous_action is not None and self.previous_state is not None and current_stats and current_stats[0]:
                            # Calculate reward
                            stats_dict = self.stats_entry_to_dict(current_stats[0])
                            reward = self.calculate_reward(stats_dict)
                            
                            # Create experience tuple
                            transition = Transition(
                                state=self.previous_state,
                                action=self.previous_action,
                                next_state=current_state,
                                reward=reward,
                                log_prob=self.previous_log_prob,
                                value=self.previous_value
                            )
                            
                            # Add to experience replay buffer
                            self.replay_buffer.append(transition)
                            
                            # Increment counter and update network when reaching update frequency
                            self.update_counter += 1
                            if self.update_counter >= self.update_frequency and len(self.replay_buffer) >= self.min_samples_before_update:
                                self.update_network(self.batch_size)
                                self.update_counter = 0
                        
                        # Select new action
                        if current_state is not None:
                            action_idx, log_prob, value = self.select_action(current_state)
                            
                            # Execute action (switch model)
                            if self.current_model_index is None or action_idx != self.current_model_index:
                                self.switch_model(action_idx)
                                self.current_model_index = action_idx
                                print(f'LSTM Actor-Critic switched model to {action_idx}')
                            else:
                                print(f'Keeping current model: {action_idx}')
                            
                            # Save current state and action for next time
                            self.previous_state = current_state
                            self.previous_action = action_idx
                            self.previous_log_prob = log_prob
                            self.previous_value = value
                            self.last_switch_time = current_time
                            
                            # Increase episode counter
                            self.episodes += 1
                            
                            # Save the model periodically
                            if self.episodes % 50 == 0:
                                self.save_model(self.model_path)
                        else:
                            print("Invalid state, skipping this decision")
                    else:
                        print("No valid state sequence available")
                
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Error in decision loop: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(1)  # Wait longer before retrying on errors

    def handle_emergency(self):
        """Handle emergencies, e.g., when queue length exceeds the max threshold"""
        # In emergencies choose the lightest model (index 0)
        selected_model = 0
        self.switch_model(selected_model)
        self.current_model_index = selected_model
        print(f"EMERGENCY: Queue length exceeded threshold. Selecting lightest model: {selected_model}")
        return selected_model

    def stats_entry_to_dict(self, stats):
        """Convert StatsEntry to a dict"""
        if stats is None:
            return {}
        
        return {
            'timestamp': stats.timestamp,
            'queue_length': stats.queue_length,
            'cur_model_index': stats.cur_model_index,
            'cur_model_accuracy': stats.cur_model_accuracy,
            'processing_latency': stats.processing_latency,
            'target_nums': stats.target_nums,
            'avg_confidence': stats.avg_confidence,
            'std_confidence': stats.std_confidence,
            'avg_size': stats.avg_size,
            'std_size': stats.std_size,
            'brightness': stats.brightness,
            'contrast': stats.contrast
        }
    
    def preprocess_stats(self, stats_list):
        """Preprocess stats for network input"""
        if not stats_list:
            return None
            
        # Extract features
        features = []
        for stats in stats_list:
            if stats is None:
                continue
                
            stats_dict = self.stats_entry_to_dict(stats)
            
            # Extract features from StatsEntry (excluding timestamp)
            feature = [
                float(stats_dict.get('queue_length', 0)) / self.queue_high_threshold_length,  # Normalized queue length
                float(stats_dict.get('cur_model_index', 0)),  # Current model index
                float(stats_dict.get('cur_model_accuracy', 0)) / 100.0,  # Normalized current model accuracy
                float(stats_dict.get('processing_latency', 0)),  # Processing latency
                float(stats_dict.get('target_nums', 0)) / 10.0,  # Normalized target count (assume avg <= 10)
                float(stats_dict.get('avg_confidence', 0)),  # Avg confidence (already in [0, 1])
                float(stats_dict.get('std_confidence', 0)),  # Std of confidence
                float(stats_dict.get('avg_size', 0)),  # Avg size
                float(stats_dict.get('std_size', 0)),  # Std of size
                float(stats_dict.get('brightness', 0)) / 255.0,  # Normalized brightness
                float(stats_dict.get('contrast', 0)) / 255.0  # Normalized contrast
            ]
            features.append(feature)
            
        # If there are no valid features, return None
        if len(features) == 0:
            print("Warning: no valid features")
            return None
        
        # Convert features to numpy array [sequence_length, features]
        features = np.array(features, dtype=np.float32)
        
        # Ensure correct shape (batch_size, sequence_length, features)
        # For LSTM, we need [batch_size=1, sequence_length, features]
        features = np.expand_dims(features, axis=0)
            
        # Convert to PyTorch tensor
        state_tensor = torch.FloatTensor(features).to(self.device)
        
        return state_tensor

    def calculate_reward(self, stats):
        """Compute reward"""
        queue_ratio = stats['queue_length'] / self.queue_low_threshold_length
        
        # Weight calculation
        w1 = max(1 - queue_ratio, 0)  # Accuracy weight
        w2 = queue_ratio  # Latency weight

        # Reward calculation
        reward = w1 * (stats['cur_model_accuracy']/100.0 + stats['avg_confidence']) - \
                 w2 * (stats['processing_latency'])
                 
        print(f"Reward: {reward:.4f} (w1={w1:.2f}, w2={w2:.2f})")
        return reward

    def select_action(self, state):
        """Select an action (model)"""
        with torch.no_grad():
            action_idx, log_prob, state_value = self.network.act(state, exploration_rate=self.exploration_rate)
        
        # Decay exploration rate
        self.exploration_rate = max(self.exploration_rate * self.exploration_decay,
                                    self.min_exploration_rate)
        
        print(f"Selected action: {action_idx}, exploration rate: {self.exploration_rate:.4f}")
        return action_idx, log_prob, state_value

    def sample_from_replay_buffer(self, batch_size):
        """Sample transitions from replay buffer"""
        if len(self.replay_buffer) < batch_size:
            return list(self.replay_buffer)
        
        # Random sampling without replacement
        indices = np.random.choice(len(self.replay_buffer), batch_size, replace=False)
        return [self.replay_buffer[i] for i in indices]

    def update_network(self, batch_size=32):
        """Batch-update the LSTM Actor-Critic network using samples from the replay buffer."""
        try:
            # If there are not enough samples in the buffer, skip the update
            if len(self.replay_buffer) < batch_size:
                print(f"Skip update: Not enough samples in replay buffer ({len(self.replay_buffer)}/{batch_size})")
                return

            # Randomly sample a batch of experiences from the buffer
            transitions = self.sample_from_replay_buffer(batch_size)

            # Validate samples and filter invalid ones
            valid_transitions = []
            for t in transitions:
                if t.log_prob is not None and isinstance(t.log_prob, torch.Tensor):
                    valid_transitions.append(t)

            if not valid_transitions:
                print("No valid transitions to update network")
                return

            # Extract batch data
            batch_size = len(valid_transitions)

            # Ensure all states have consistent sequence length
            seq_lengths = set()
            for t in valid_transitions:
                if t.state is not None and len(t.state.shape) >= 2:
                    seq_lengths.add(t.state.shape[1])

            if len(seq_lengths) > 1:
                print(f"Warning: Inconsistent sequence lengths {seq_lengths}, skipping update")
                return

            states = torch.cat([t.state for t in valid_transitions], dim=0)
            actions = torch.tensor([t.action for t in valid_transitions], dtype=torch.long).to(self.device)
            rewards = torch.tensor([t.reward for t in valid_transitions], dtype=torch.float32).view(-1, 1).to(self.device)

            # Handle next_states (some may be None)
            next_states_list = []
            masks = []
            for t in valid_transitions:
                if t.next_state is not None:
                    next_states_list.append(t.next_state)
                    masks.append(1.0)
                else:
                    # If next_state is None, use a zero tensor instead
                    next_states_list.append(torch.zeros_like(t.state))
                    masks.append(0.0)

            next_states = torch.cat(next_states_list, dim=0)
            masks = torch.tensor(masks, dtype=torch.float32).view(-1, 1).to(self.device)

            # Update Critic network
            self.critic_optimizer.zero_grad()

            # Compute current state value
            self.network.reset_hidden(batch_size)
            _, current_values = self.network(states)

            # Compute next state value
            self.network.reset_hidden(batch_size)
            _, next_values = self.network(next_states)
            next_values = next_values.detach()  # Stop gradient

            # Compute target and critic loss
            target_values = rewards + self.gamma * next_values * masks
            critic_loss = F.mse_loss(current_values, target_values)

            # Update critic
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
            self.critic_optimizer.step()

            # Update Actor network
            self.actor_optimizer.zero_grad()

            # Recompute policy
            self.network.reset_hidden(batch_size)
            action_probs, values = self.network(states)

            # Compute advantage (using current network value)
            advantages = (target_values - values).detach()

            print(f"Advantages stats: mean={advantages.mean().item():.6f}, std={advantages.std().item():.6f}, min={advantages.min().item():.6f}, max={advantages.max().item():.6f}")

            # Create a categorical distribution
            dist = Categorical(action_probs)

            # Compute log probability of selected actions
            action_log_probs = dist.log_prob(actions)

            # Check NaN or Inf in log_probs and advantages before update
            if torch.isnan(action_log_probs).any() or torch.isinf(action_log_probs).any():
                print("Warning: NaN or Inf detected in log_probs")

            if torch.isnan(advantages).any() or torch.isinf(advantages).any():
                print("Warning: NaN or Inf detected in advantages")

            # Compute actor loss
            actor_loss = -(action_log_probs * advantages.squeeze()).mean()

            # Add entropy regularization
            entropy_loss = dist.entropy().mean()
            actor_loss -= self.entropy_beta * entropy_loss

            # === Special loss for extremely small probabilities ===
            # Set minimum acceptable probability threshold
            min_prob_threshold = 5e-2  # 5%

            # Check the minimum probability in each batch sample
            min_probs_per_batch = action_probs.min(dim=1)[0]  # Minimum prob per batch sample

            # Count how many batch samples contain extremely small probabilities
            extreme_prob_batches = (min_probs_per_batch < min_prob_threshold).float()
            num_extreme_batches = extreme_prob_batches.sum().item()

            # Only add penalty when extremely small probabilities exist
            if num_extreme_batches > 0:
                # Minimum probability for each action across the batch
                min_action_probs = action_probs.min(dim=0)[0]

                # Actions below threshold
                small_prob_actions = (min_action_probs < min_prob_threshold).float()
                num_small_actions = small_prob_actions.sum().item()

                if num_small_actions > 0:
                    # Log penalty: -log(p) grows very large as p approaches 0
                    prob_penalty = -torch.log(min_action_probs + 1e-10) * small_prob_actions

                    # Only penalize actions below threshold
                    extreme_prob_penalty = prob_penalty.sum() / max(1.0, num_small_actions)

                    # Dynamically adjust penalty coefficient: smaller probabilities -> larger penalty
                    penalty_coef = 0.1 * (1.0 - min_action_probs.min().item() / min_prob_threshold)
                    penalty_coef = min(penalty_coef, 0.5)  # Cap max coefficient

                    # Add to actor loss
                    actor_loss = actor_loss + penalty_coef * extreme_prob_penalty

                    print(
                        f"Applied probability penalty: {penalty_coef:.4f} * {extreme_prob_penalty.item():.4f} "
                        f"for {num_small_actions} actions below threshold"
                    )

            # Update actor
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
            self.actor_optimizer.step()

            # Reset LSTM hidden state for next prediction
            self.network.reset_hidden(batch_size=1)

            # Update statistics
            self.training_steps += 1

            print(f"Network updated - Actor Loss: {actor_loss.item():.4f}, Critic Loss: {critic_loss.item():.4f}")
        except Exception as e:
            print(f"Error during network update: {e}")
            import traceback
            traceback.print_exc()

    def save_model(self, path):
        """Save model to file"""
        try:
            torch.save({
                'model_state_dict': self.network.state_dict(),
                'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
                'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
                'exploration_rate': self.exploration_rate,
                'episodes': self.episodes,
                'training_steps': self.training_steps
            }, path)
            print(f"Model saved to {path}")
        except Exception as e:
            print(f"Error saving model: {e}")
    
    def load_model(self, path):
        """Load model from file"""
        try:
            if not os.path.exists(path):
                print(f"Model file not found: {path}")
                return False
                
            checkpoint = torch.load(path, map_location=self.device)
            self.network.load_state_dict(checkpoint['model_state_dict'])
            
            if 'actor_optimizer_state_dict' in checkpoint and 'critic_optimizer_state_dict' in checkpoint:
                self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
                self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
                
            self.exploration_rate = checkpoint.get('exploration_rate', self.exploration_rate)
            self.episodes = checkpoint.get('episodes', 0)
            self.training_steps = checkpoint.get('training_steps', 0)
            print(f"Model loaded from {path}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def switch_model(self, index: int):
        """Switch to the model at the given index"""
        self.detector_instance.switch_model(index)

    def get_detector_stats(self):
        """Get the latest detector stats"""
        stats = self.detector_instance.stats_manager.get_latest_stats()
        return stats
    
    def get_detector_interval_stats(self, nums: int = 5, interval: float = 1.0):
        """Get detector stats at a given interval"""
        stats = self.detector_instance.stats_manager.get_interval_stats(nums, interval)
        return stats
