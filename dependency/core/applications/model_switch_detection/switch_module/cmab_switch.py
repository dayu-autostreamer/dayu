from .base_switch import BaseSwitch
import time
import threading
import numpy as np
from scipy import linalg


class CMABSwitch(BaseSwitch):
    def __init__(self, decision_interval: int, 
                 detector_instance: object,
                 queue_low_threshold_length: int = 10,
                 lambda_reg: float = 1.0,
                 noise_variance: float = 0.1,
                 exploration_rate: float = 0.1,
                 *args, **kwargs):
        """
        Linear contextual Thompson Sampling (CMAB) model switcher.

        Args:
            decision_interval: Time interval (seconds) between model switch decisions.
            detector_instance: Detector instance.
            queue_low_threshold_length: Queue length threshold used for reward calculation.
            context_dimension: Feature vector dimension.
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

        # Parameters for each model
        self.models = {}
        self.init_thompson_sampling_models()
        
        # Track last selected arm and switch time
        self.last_selected_arm = None
        self.last_switch_time = time.time()
        self.last_context = None
        
        # Start decision thread
        self.switch_thread = threading.Thread(target=self._switch_loop)
        self.switch_thread.daemon = True
        self.switch_thread.start()
        
        print(f"CMAB Linear Thompson Sampling initialized with {self.models_num} models")

    def init_thompson_sampling_models(self):
        """Initialize linear Thompson Sampling parameters for each model."""
        for i in range(self.models_num):
            self.models[i] = {
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
                
                if current_stats and self.last_selected_arm is not None and self.last_context is not None:
                    # Convert StatsEntry to dict for reward calculation
                    stats_dict = self.stats_entry_to_dict(current_stats[0])
                    
                    # Compute reward for the previously selected arm
                    reward = self.calculate_reward(stats_dict)
                    
                    # Update parameters for the previously selected arm
                    self.update_thompson_sampling(self.last_selected_arm, self.last_context, reward)
                
                # Extract current context features
                if current_stats:
                    context = self.extract_features(current_stats[0])
                    
                    # Select a new arm using Thompson sampling
                    selected_arm = self.select_model_thompson_sampling(context)
                    
                    # Switch to the selected model
                    self.switch_model(selected_arm)
                    print(f'Thompson Sampling switched model to {selected_arm}')
                    
                    # Update tracking variables
                    self.last_selected_arm = selected_arm
                    self.last_context = context
                    self.last_switch_time = current_time
                    
                    # Print per-arm statistics
                    self.print_arm_stats()
            
            time.sleep(0.1)

    def extract_features(self, stats_entry):
        """Extract feature vector from StatsEntry."""
        stats = self.stats_entry_to_dict(stats_entry)
        
        # Extract features, normalizing only the required ones
        # Fixed number of features is 10 (9 features + 1 bias term)
        features = np.zeros(10)
        
        # Normalize only queue_length, brightness, and contrast
        features[0] = float(stats['queue_length']) / self.queue_low_threshold_length  # Normalize queue length
        features[1] = float(stats['processing_latency'])  # No normalization
        features[2] = float(stats['target_nums'])  # No normalization
        features[3] = float(stats['avg_confidence'])  # No normalization
        features[4] = float(stats['std_confidence'])  # No normalization
        features[5] = float(stats['avg_size'])  # No normalization
        features[6] = float(stats['std_size'])  # No normalization
        features[7] = float(stats['brightness']) / 255.0  # Normalize brightness
        features[8] = float(stats['contrast']) / 255.0  # Normalize contrast

        # Add a constant feature as a bias term
        features[9] = 1.0
            
        # Print extracted features
        print("Extracted features:")
        for i, value in enumerate(features):
            print(f"  feature_{i}: {value:.4f}")
            
        return features

    def sample_parameter(self, arm):
        """Sample parameter vector from the model's posterior distribution."""
        model_data = self.models[arm]
        
        try:
            # Use Cholesky decomposition to sample from multivariate normal distribution for numerical stability
            L = linalg.cholesky(model_data['Sigma'], lower=True)
            
            # Sample standard normal distribution
            standard_normal = np.random.standard_normal(self.context_dimension)
            
            # Transform to target distribution
            theta_sample = model_data['mu'] + L @ standard_normal
            
            return theta_sample
        except (np.linalg.LinAlgError, ValueError) as e:
            print(f"Warning: Model {arm} sampling error: {e}")
            print("Using mean vector as a substitute for sampling")
            # If sampling fails, return the mean vector
            return model_data['mu']

    def select_model_thompson_sampling(self, context):
        """Select model using Thompson Sampling strategy."""
        # Check if it's a forced exploration round
        self.exploration_counter += 1
        force_exploration = self.exploration_counter >= self.force_exploration_count
        
        # If it's a forced exploration round, reset counter and select randomly
        if force_exploration:
            self.exploration_counter = 0
            # Randomly select a model
            selected_arm = np.random.randint(0, self.models_num)
            print(f"Forced exploration: Randomly selected model {selected_arm}")
            return selected_arm
        
        # ε-greedy random exploration
        if np.random.random() < self.exploration_rate:
            selected_arm = np.random.randint(0, self.models_num)
            print(f"Random exploration: Selected model {selected_arm}")
            return selected_arm
        
        # Thompson Sampling decision
        expected_rewards = {}
        sampled_params = {}
        
        # Sample parameters for each model and compute expected rewards
        for arm in range(self.models_num):
            # Sample parameter vector from posterior distribution
            theta = self.sample_parameter(arm)
            sampled_params[arm] = theta
            
            # Compute expected reward
            expected_reward = np.dot(theta, context)
            expected_rewards[arm] = float(expected_reward)  # Ensure it's a Python float

        # Select the model with the highest expected reward
        selected_arm = max(expected_rewards, key=expected_rewards.get)
        
        print(f"Thompson sampling selected model: {selected_arm} (expected reward={expected_rewards[selected_arm]:.4f})")

        # Print expected rewards for all models
        for arm, reward in expected_rewards.items():
            print(f"  Model {arm}: Expected reward={reward:.4f}, "
                  f"Parameter norm={np.linalg.norm(sampled_params[arm]):.4f}")

        return selected_arm

    def update_thompson_sampling(self, arm, context, reward):
        """Update the parameters of the Thompson Sampling model."""
        model_data = self.models[arm]
        
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
            # Compute covariance matrix (Sigma)
            model_data['Sigma'] = np.linalg.inv(model_data['precision'])
            
            # Compute mean vector (mu = Sigma * precision_mean)
            model_data['mu'] = model_data['Sigma'] @ model_data['precision_mean']
            
            # Update reward variance estimate
            if model_data['count'] > 1:
                avg_reward = model_data['sum_reward'] / model_data['count']
                # Simple variance estimate
                model_data['reward_variance'] = max(0.1, self.noise_variance)
            
            print(f"Updated Thompson Sampling parameters for model {arm}:")
            print(f"  count={model_data['count']}, avg_reward={model_data['sum_reward']/model_data['count']:.4f}")
            print(f"  mu_norm={np.linalg.norm(model_data['mu']):.4f}, var={model_data['reward_variance']:.4f}")
        except np.linalg.LinAlgError:
            print(f"Warning: Unable to compute inverse of precision matrix for model {arm}. Using previous values.")

    def calculate_reward(self, stats):
        """Calculate the reward value for the model."""
        queue_ratio = stats['queue_length'] / self.queue_low_threshold_length
        
        # Weight calculations
        w1 = max(1 - queue_ratio, 0)  # Accuracy weight
        w2 = queue_ratio  # Latency weight

        # Reward calculation
        raw_reward = w1 * (stats['cur_model_accuracy']/100.0 + stats['avg_confidence']) - \
                w2 * (stats['processing_latency'])
        
        print(f"Calculated reward: {raw_reward:.4f} (w1={w1:.2f}, w2={w2:.2f})")

        return raw_reward

    def switch_model(self, index: int):
        """Switch to the model at the specified index."""
        self.detector_instance.switch_model(index)

    def get_detector_stats(self):
        """Get the latest statistics from the detector."""
        stats = self.detector_instance.stats_manager.get_latest_stats()
        return stats
    
    def get_detector_interval_stats(self, nums: int = 5, interval: float = 1.0):
        """Get statistics from the detector over intervals."""
        stats = self.detector_instance.stats_manager.get_interval_stats(nums, interval)
        return stats
    
    def stats_entry_to_dict(self, stats_entry):
        """Convert StatsEntry object to dictionary."""
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
    
    def print_arm_stats(self):
        """Print the current statistics for all arms."""
        print("\nLinear Thompson Sampling Model Parameters:")
        print("------------------------")
        for arm in range(self.models_num):
            model_data = self.models[arm]
            if model_data['count'] > 0:
                avg_reward = model_data['sum_reward'] / model_data['count']
            else:
                avg_reward = 0.0
            print(f"Model {arm}: count={model_data['count']}, avg_reward={avg_reward:.4f}, "
                  f"param_norm={np.linalg.norm(model_data['mu']):.4f}")
        print("------------------------\n")