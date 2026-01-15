from .base_switch import BaseSwitch
import time
import threading
import numpy as np
from scipy import linalg
import numpy as np

class DummyStrategy:
    """Placeholder strategy class. It will be replaced by a concrete strategy implementation."""

    def __init__(self, models_num, queue_low_threshold_length):
        self.models_num = models_num
        self.queue_low_threshold_length = queue_low_threshold_length
        
    def select_model(self, stats, current_model_index):
        """Return a placeholder model decision."""
        return current_model_index


class QueueLengthStrategy:
    """Adaptive queue-length strategy based on a cooling mechanism."""

    def __init__(self, models_num, queue_low_threshold_length):
        self.models_num = models_num
        
        # Queue threshold
        self.queue_threshold = queue_low_threshold_length * 2.0
        
        # Stability-related parameters
        self.stability_counter = 0
        self.base_stability_threshold = 3
        
        # Cooling mechanism parameters
        self.cooling_factor = 1
        self.max_cooling_factor = 5
        self.cooling_recovery_rate = 1
    
    def select_model(self, stats, current_model_index):
        """Make a model switch decision based on queue length."""
        queue_length = stats.get('queue_length', 0)
        
        # Update system stability state
        is_currently_stable = queue_length <= self.queue_threshold
        
        # If the system is unstable (queue exceeds the threshold), consider downgrading immediately
        if not is_currently_stable:
            self.stability_counter = 0  # Reset stability counter

            # If a lighter model is available, downgrade
            if current_model_index > 0:
                next_model = current_model_index - 1
                print(
                    f"Queue length ({queue_length}) exceeded threshold ({self.queue_threshold}). "
                    f"Downgrading from model {current_model_index} to {next_model}."
                )
                return next_model
            else:
                print(
                    f"Queue length ({queue_length}) exceeded threshold, but the lightest model is already in use."
                )
                return current_model_index
        
        # If the system is stable (queue below threshold), consider upgrading
        else:
            # Increase stability counter
            self.stability_counter += 1
            print(
                f"System stable: stability_counter={self.stability_counter}/"
                f"{self.base_stability_threshold * self.cooling_factor}"
            )

            # Decrease cooling factor
            if self.cooling_factor > 1 and self.stability_counter % 2 == 0:
                self.cooling_factor = max(1, self.cooling_factor - self.cooling_recovery_rate)
                print(f"Cooling factor decreased to: {self.cooling_factor}")

            # Upgrade only when stable for long enough
            current_stability_threshold = int(self.base_stability_threshold * self.cooling_factor)
            if self.stability_counter >= current_stability_threshold:
                # If a higher-quality model is available, upgrade
                if current_model_index < self.models_num - 1:
                    next_model = current_model_index + 1
                    print(
                        f"System has been stable ({self.stability_counter}/{current_stability_threshold}). "
                        f"Upgrading from model {current_model_index} to {next_model}."
                    )
                    self.stability_counter = 0
                    return next_model
                else:
                    print("System is stable, but the highest-level model is already in use.")
                    self.stability_counter = 0
            
        # Keep current model
        return current_model_index


class ProbabilisticStrategy:
    """Probability-based adaptive strategy.

    Make model switch decisions with different probabilities based on queue length:
    1. If queue length is below the low threshold, upgrade with high probability.
    2. If queue length is between low and high thresholds, decide proportionally.
    3. If queue length is above the high threshold, downgrade with high probability.
    """
    
    def __init__(self, models_num, queue_low_threshold_length):
        self.models_num = models_num
        
        # Define queue thresholds
        self.queue_low = queue_low_threshold_length
        self.queue_high = queue_low_threshold_length * 2
        
        # Probability parameters
        self.upgrade_prob_base = 0.7  # Base upgrade probability
        self.downgrade_prob_base = 0.8  # Base downgrade probability

        # Stability counter for model stability
        self.stability_counter = 0
        self.min_stability_period = 3  # Minimum stability period

    def select_model(self, stats, current_model_index):
        """Make a model switch decision based on queue length and probabilities."""
        # Get current queue length
        queue_length = stats.get('queue_length', 0)
        
        # Print current state
        print(
            f"Probabilistic strategy analysis - queue_length: {queue_length}, low: {self.queue_low}, high: {self.queue_high}"
        )
        print(f"Current model: {current_model_index}, total models: {self.models_num}")

        # Increase stability counter
        self.stability_counter += 1
        
        # Check whether we should consider switching (do not switch during stability period)
        if self.stability_counter < self.min_stability_period:
            print(
                f"In stability period ({self.stability_counter}/{self.min_stability_period}), keep current model"
            )
            return current_model_index
        
        # Compute base upgrade/downgrade probabilities
        if queue_length <= self.queue_low:
            # Low load: consider upgrading
            upgrade_probability = self.upgrade_prob_base
            downgrade_probability = 0.0
            print(
                f"Low queue load - upgrade_probability: {upgrade_probability:.2f}, downgrade_probability: {downgrade_probability:.2f}"
            )

        elif queue_length >= self.queue_high:
            # High load: consider downgrading
            upgrade_probability = 0.0
            downgrade_probability = self.downgrade_prob_base
            print(
                f"High queue load - upgrade_probability: {upgrade_probability:.2f}, downgrade_probability: {downgrade_probability:.2f}"
            )

        else:
            # Medium load: adjust probabilities based on load ratio
            load_ratio = (queue_length - self.queue_low) / (self.queue_high - self.queue_low)
            upgrade_probability = self.upgrade_prob_base * (1 - load_ratio)
            downgrade_probability = self.downgrade_prob_base * load_ratio
            print(
                f"Medium queue load (ratio: {load_ratio:.2f}) - upgrade_probability: {upgrade_probability:.2f}, downgrade_probability: {downgrade_probability:.2f}"
            )

        # Adjust probabilities by current model level (lowest/highest cannot downgrade/upgrade)
        if current_model_index == 0:  # Lowest level model
            downgrade_probability = 0.0
        if current_model_index == self.models_num - 1:  # Highest level model
            upgrade_probability = 0.0
        
        # Generate a random value and make decision
        random_value = np.random.random()
        
        if random_value < upgrade_probability:
            # Trigger upgrade
            if current_model_index < self.models_num - 1:
                next_model = current_model_index + 1
                print(
                    f"Probabilistic strategy - random_value ({random_value:.2f}) < upgrade_probability ({upgrade_probability:.2f}), upgrading to model {next_model}"
                )
                self.stability_counter = 0  # Reset stability counter
                return next_model
            
        elif random_value < upgrade_probability + downgrade_probability:
            # Trigger downgrade
            if current_model_index > 0:
                next_model = current_model_index - 1
                print(
                    f"Probabilistic strategy - random_value ({random_value:.2f}) < downgrade_probability ({downgrade_probability:.2f}), downgrading to model {next_model}"
                )
                self.stability_counter = 0  # Reset stability counter
                return next_model
        
        # Default: keep current model
        print(f"Probabilistic strategy - keep current model {current_model_index}")
        return current_model_index


class DistributionBasedStrategy:
    """Adaptive strategy based on target count and size distributions."""

    def __init__(self, models_num, queue_low_threshold_length):
        self.models_num = models_num
        self.queue_low_threshold = queue_low_threshold_length
        
        # Statistics window size
        self.window_size = 100
        
        # Initialize statistics window
        self.target_nums_history = []  # Target count history
        self.target_size_history = []  # Target size history

        # Statistics
        self.target_nums_mean = None
        self.target_nums_std = None
        self.target_size_mean = None
        self.target_size_std = None
        
        # Threshold for standard deviation deviation
        self.std_threshold = 1.0
        
        # Stability counters
        self.upgrade_counter = 0
        self.downgrade_counter = 0
        self.stability_threshold = 3
    
    def _update_statistics(self, target_nums, target_size):
        """Update statistics."""
        # Update historical data
        self.target_nums_history.append(target_nums)
        self.target_size_history.append(target_size)
        
        # Keep window size
        if len(self.target_nums_history) > self.window_size:
            self.target_nums_history.pop(0)
            self.target_size_history.pop(0)
        
        # Compute stats only when enough samples are collected
        if len(self.target_nums_history) >= 10:
            self.target_nums_mean = np.mean(self.target_nums_history)
            self.target_nums_std = np.std(self.target_nums_history) or 1.0  # Avoid division by zero

            self.target_size_mean = np.mean(self.target_size_history)
            self.target_size_std = np.std(self.target_size_history) or 1.0  # Avoid division by zero

    def select_model(self, stats, current_model_index):
        """Select model based on the distribution of target count and size."""
        target_nums = stats.get('target_nums', 0)
        avg_size = stats.get('avg_size', 0)
        
        # Update statistics
        self._update_statistics(target_nums, avg_size)
        
        # If statistics are not initialized yet, keep current model
        if self.target_nums_mean is None:
            print("Statistics are not initialized yet, keep current model")
            return current_model_index
        
        # Compute deviation from mean (in units of std)
        nums_deviation = (target_nums - self.target_nums_mean) / self.target_nums_std
        size_deviation = (avg_size - self.target_size_mean) / self.target_size_std
        
        print(f"Target count: {target_nums} (mean: {self.target_nums_mean:.2f}, deviation: {nums_deviation:.2f}σ)")
        print(f"Target size: {avg_size:.2f} (mean: {self.target_size_mean:.2f}, deviation: {size_deviation:.2f}σ)")

        # Check whether we need to upgrade the model
        need_upgrade = (
            nums_deviation > self.std_threshold or  # Target count above mean + std
            size_deviation < -self.std_threshold    # Target size below mean - std
        )
        
        # Check whether we need to downgrade the model
        need_downgrade = (
            nums_deviation < -self.std_threshold or  # Target count below mean - std
            size_deviation > self.std_threshold      # Target size above mean + std
        )
        
        # Decision logic
        if need_upgrade:
            self.upgrade_counter += 1
            self.downgrade_counter = 0
            print(f"Upgrade may be needed: {self.upgrade_counter}/{self.stability_threshold}")

            if self.upgrade_counter >= self.stability_threshold:
                if current_model_index < self.models_num - 1:
                    next_model = current_model_index + 1
                    print(f"Upgrade needed consistently, switching to model {next_model}")
                    self.upgrade_counter = 0
                    return next_model
                else:
                    print("Upgrade needed, but the highest-level model is already in use")
                    self.upgrade_counter = 0
        
        elif need_downgrade:
            self.downgrade_counter += 1
            self.upgrade_counter = 0
            print(f"Downgrade may be needed: {self.downgrade_counter}/{self.stability_threshold}")

            if self.downgrade_counter >= self.stability_threshold:
                if current_model_index > 0:
                    next_model = current_model_index - 1
                    print(f"Downgrade needed consistently, switching to model {next_model}")
                    self.downgrade_counter = 0
                    return next_model
                else:
                    print("Downgrade needed, but the lightest model is already in use")
                    self.downgrade_counter = 0
        
        # If there is no clear upgrade/downgrade signal, keep current model
        return current_model_index
    
class MetaSwitch(BaseSwitch):
    def __init__(self, decision_interval: int, 
                 detector_instance: object,
                 queue_low_threshold_length: int = 10,
                 lambda_reg: float = 1.0,
                 noise_variance: float = 0.1,
                 exploration_rate: float = 0.1,
                 *args, **kwargs):
        """
        Meta-strategy switcher that uses Thompson sampling to select among multiple base strategies.

        Args:
            decision_interval: Time interval (seconds) between model switch decisions.
            detector_instance: Detector instance.
            queue_low_threshold_length: Queue length threshold used for reward calculation.
            lambda_reg: Regularization parameter.
            noise_variance: Observation noise variance.
            exploration_rate: Probability of random exploration.
        """
        self.models_num = detector_instance.get_models_num()
        self.decision_interval = decision_interval
        self.detector_instance = detector_instance
        self.queue_low_threshold_length = queue_low_threshold_length
        
        # Feature dimension and Bayesian linear regression parameters
        self.context_dimension = 10  # Fixed to 10 features (9 actual features + 1 bias term)
        self.lambda_reg = lambda_reg
        self.noise_variance = noise_variance
        
        # Exploration parameters
        self.exploration_rate = exploration_rate
        self.exploration_counter = 0
        self.force_exploration_count = 15  # Force exploration once every 15 decisions

        # Initialize base strategies
        self.strategies = self._initialize_strategies()
        
        # Meta-strategy Thompson Sampling parameters
        self.meta_models = {}
        self.init_meta_thompson_sampling()
        
        # Track last selected strategy/model and switch time
        self.last_selected_strategy = None
        self.last_selected_arm = None
        self.last_switch_time = time.time()
        self.last_context = None
        
        # Start decision thread
        self.switch_thread = threading.Thread(target=self._switch_loop)
        self.switch_thread.daemon = True
        self.switch_thread.start()
        
        print(f"MetaSwitch initialized with {self.models_num} models and {len(self.strategies)} strategies")

    def _initialize_strategies(self):
        """Initialize the base strategy set."""
        # Add strategies here if needed. Currently we use three strategies.
        strategies = {
            'strategy1': QueueLengthStrategy(self.models_num, self.queue_low_threshold_length),
            'strategy2': ProbabilisticStrategy(self.models_num, self.queue_low_threshold_length),
            'strategy3': DistributionBasedStrategy(self.models_num, self.queue_low_threshold_length)
        }
        return strategies

    def init_meta_thompson_sampling(self):
        """Initialize Thompson Sampling parameters for the meta-strategy."""
        for strategy_name in self.strategies.keys():
            self.meta_models[strategy_name] = {
                # Parameter mean vector (zeros)
                'mu': np.zeros(self.context_dimension),
                
                # Parameter covariance matrix (identity scaled by inverse regularization)
                'Sigma': np.eye(self.context_dimension) / self.lambda_reg,
                
                # Sufficient statistic: X^T X
                'precision': self.lambda_reg * np.eye(self.context_dimension),
                
                # Sufficient statistic: X^T y
                'precision_mean': np.zeros(self.context_dimension),
                
                # Observation count
                'count': 0,
                
                # Sum of observed rewards
                'sum_reward': 0,
                
                # Reward variance estimate
                'reward_variance': 1.0
            }

    def _switch_loop(self):
        """Main loop for model switch decisions."""
        while True:
            current_time = time.time()
            
            # If the decision interval time has been reached
            if current_time - self.last_switch_time > self.decision_interval:
                # Get current statistics to calculate reward
                current_stats = self.get_detector_stats()
                
                if current_stats and self.last_selected_strategy is not None and self.last_context is not None and self.last_selected_arm is not None:
                    # Convert StatsEntry to dict for reward calculation
                    stats_dict = self.stats_entry_to_dict(current_stats[0])
                    
                    # Compute reward for the previous decision
                    reward = self.calculate_reward(stats_dict)
                    
                    # Update the last selected strategy parameters
                    self.update_meta_thompson_sampling(self.last_selected_strategy, self.last_context, reward)
                
                # Extract current context features
                if current_stats:
                    context = self.extract_features(current_stats[0])
                    
                    # Select a strategy using meta Thompson sampling
                    selected_strategy = self.select_meta_strategy(context)
                    
                    # Get model recommendation from the selected strategy
                    stats_dict = self.stats_entry_to_dict(current_stats[0])
                    current_model_idx = stats_dict['cur_model_index']
                    selected_arm = self.strategies[selected_strategy].select_model(stats_dict, current_model_idx)
                    
                    # Switch to the selected model
                    self.switch_model(selected_arm)
                    print(f'MetaSwitch: strategy {selected_strategy} switched model to {selected_arm}')
                    
                    # Update tracking variables
                    self.last_selected_strategy = selected_strategy
                    self.last_selected_arm = selected_arm
                    self.last_context = context
                    self.last_switch_time = current_time
                    
                    # Print meta-strategy stats
                    self.print_meta_stats()
            
            time.sleep(0.1)

    def extract_features(self, stats_entry):
        """Extract a feature vector from StatsEntry."""
        stats = self.stats_entry_to_dict(stats_entry)
        
        # Extract features; normalize only the needed ones.
        # Fixed feature count is 10 (9 features + 1 bias term).
        features = np.zeros(10)
        
        # Normalize only queue_length, brightness and contrast
        features[0] = float(stats['queue_length']) / self.queue_low_threshold_length  # Normalized queue length
        features[1] = float(stats['processing_latency'])  # Not normalized
        features[2] = float(stats['target_nums'])  # Not normalized
        features[3] = float(stats['avg_confidence'])  # Not normalized
        features[4] = float(stats['std_confidence'])  # Not normalized
        features[5] = float(stats['avg_size'])  # Not normalized
        features[6] = float(stats['std_size'])  # Not normalized
        features[7] = float(stats['brightness']) / 255.0  # Normalized brightness
        features[8] = float(stats['contrast']) / 255.0  # Normalized contrast

        # Add a constant feature as bias term
        features[9] = 1.0
            
        # Print extracted features
        print("Extracted features:")
        for i, value in enumerate(features):
            print(f"  feature_{i}: {value:.4f}")
            
        return features

    def sample_meta_parameter(self, strategy_name):
        """Sample a parameter vector from the meta-strategy posterior."""
        model_data = self.meta_models[strategy_name]
        
        try:
            # Sample from a multivariate normal via Cholesky for numerical stability
            L = linalg.cholesky(model_data['Sigma'], lower=True)
            
            # Sample from standard normal
            standard_normal = np.random.standard_normal(self.context_dimension)
            
            # Transform to target distribution
            theta_sample = model_data['mu'] + L @ standard_normal
            
            return theta_sample
        except (np.linalg.LinAlgError, ValueError) as e:
            print(f"Warning: sampling failed for strategy {strategy_name}: {e}")
            print("Falling back to mean vector")
            # If sampling fails, fall back to the mean vector
            return model_data['mu']

    def select_meta_strategy(self, context):
        """Select a base strategy using meta Thompson Sampling."""
        # Check whether this is a forced-exploration round
        self.exploration_counter += 1
        force_exploration = self.exploration_counter >= self.force_exploration_count
        
        # If forced exploration, reset counter and choose randomly
        if force_exploration:
            self.exploration_counter = 0
            selected_strategy = np.random.choice(list(self.strategies.keys()))
            print(f"Forced exploration: randomly selected strategy {selected_strategy}")
            return selected_strategy
        
        # ε-greedy random exploration
        if np.random.random() < self.exploration_rate:
            selected_strategy = np.random.choice(list(self.strategies.keys()))
            print(f"Random exploration: selected strategy {selected_strategy}")
            return selected_strategy
        
        # Meta Thompson Sampling decision
        expected_rewards = {}
        sampled_params = {}
        
        # Sample parameters and compute expected reward for each strategy
        for strategy_name in self.strategies.keys():
            theta = self.sample_meta_parameter(strategy_name)
            sampled_params[strategy_name] = theta
            
            expected_reward = np.dot(theta, context)
            expected_rewards[strategy_name] = float(expected_reward)

        # Select the strategy with the highest expected reward
        selected_strategy = max(expected_rewards, key=expected_rewards.get)
        
        print(
            f"Meta Thompson sampling selected strategy: {selected_strategy} "
            f"(expected_reward={expected_rewards[selected_strategy]:.4f})"
        )

        # Print expected reward for all strategies
        for strategy, reward in expected_rewards.items():
            print(
                f"  strategy {strategy}: expected_reward={reward:.4f}, "
                f"param_norm={np.linalg.norm(sampled_params[strategy]):.4f}"
            )

        return selected_strategy

    def update_meta_thompson_sampling(self, strategy_name, context, reward):
        """Update the meta Thompson Sampling model parameters."""
        model_data = self.meta_models[strategy_name]
        
        # Accumulate count and reward
        model_data['count'] += 1
        model_data['sum_reward'] += reward
        
        # Update covariance and precision matrices
        context_2d = context.reshape(-1, 1)  # Column vector

        # Update precision matrix (X^T X)
        model_data['precision'] += context_2d @ context_2d.T
        
        # Update precision mean (X^T y)
        model_data['precision_mean'] += context * reward
        
        # Recompute mean vector and covariance matrix
        try:
            model_data['Sigma'] = np.linalg.inv(model_data['precision'])
            model_data['mu'] = model_data['Sigma'] @ model_data['precision_mean']
            
            # Update reward variance estimate
            if model_data['count'] > 1:
                _avg_reward = model_data['sum_reward'] / model_data['count']
                model_data['reward_variance'] = max(0.1, self.noise_variance)
            
            print(f"Updated meta Thompson Sampling parameters for strategy {strategy_name}:")
            print(f"  count={model_data['count']}, avg_reward={model_data['sum_reward']/model_data['count']:.4f}")
            print(f"  mu_norm={np.linalg.norm(model_data['mu']):.4f}, var={model_data['reward_variance']:.4f}")
        except np.linalg.LinAlgError:
            print(
                f"Warning: failed to invert precision matrix for strategy {strategy_name}. Using previous values."
            )

    def calculate_reward(self, stats):
        """Compute model reward."""
        queue_ratio = stats['queue_length'] / self.queue_low_threshold_length
        
        # Weight calculation
        w1 = max(1 - queue_ratio, 0)  # Accuracy weight
        w2 = queue_ratio  # Latency weight

        # Reward calculation
        raw_reward = w1 * (stats['cur_model_accuracy']/100.0 + stats['avg_confidence']) - \
                w2 * (stats['processing_latency'])
        
        print(f"Reward: {raw_reward:.4f} (w1={w1:.2f}, w2={w2:.2f})")

        return raw_reward

    def switch_model(self, index: int):
        """Switch to the model at the given index."""
        self.detector_instance.switch_model(index)

    def get_detector_stats(self):
        """Get the latest detector stats."""
        stats = self.detector_instance.stats_manager.get_latest_stats()
        return stats
    
    def get_detector_interval_stats(self, nums: int = 5, interval: float = 1.0):
        """Get detector stats at a given interval."""
        stats = self.detector_instance.stats_manager.get_interval_stats(nums, interval)
        return stats
    
    def stats_entry_to_dict(self, stats_entry):
        """Convert a StatsEntry object to a dict."""
        if stats_entry is None:
            return {}
        
        return {
            'timestamp': stats_entry.timestamp,
            'queue_length': stats_entry.queue_length,
            'cur_model_index': stats_entry.cur_model_index,
            'cur_model_accuracy': stats_entry.cur_model_accuracy,
            'processing_latency': stats_entry.processing_latency,
            'target_nums': stats_entry.target_nums,
            'avg_confidence': stats_entry.avg_confidence,
            'std_confidence': stats_entry.std_confidence,
            'avg_size': stats_entry.avg_size,
            'std_size': stats_entry.std_size,
            'brightness': stats_entry.brightness,
            'contrast': stats_entry.contrast
        }
    
    def print_meta_stats(self):
        """Print current statistics for all meta strategies."""
        print("\nMeta Thompson Sampling strategy parameters:")
        print("------------------------")
        for strategy_name in self.strategies.keys():
            model_data = self.meta_models[strategy_name]
            if model_data['count'] > 0:
                avg_reward = model_data['sum_reward'] / model_data['count']
            else:
                avg_reward = 0.0
            print(f"strategy {strategy_name}: count={model_data['count']}, avg_reward={avg_reward:.4f}, "
                  f"param_norm={np.linalg.norm(model_data['mu']):.4f}")
        print("------------------------\n")