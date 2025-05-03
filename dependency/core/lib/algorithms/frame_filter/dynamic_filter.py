import abc
from core.lib.common import ClassFactory, ClassType
from .base_filter import BaseFilter
import time
import random
import math
__all__ = ('DynamicFilter',)


@ClassFactory.register(ClassType.GEN_FILTER, alias='dynamic')
class DynamicFilter(BaseFilter, abc.ABC):
    def __init__(self, min_fps=1, max_fps=5, min_duration=120, max_duration=600, transition_ratio=0.1):
        self.min_fps = min_fps
        self.max_fps = max_fps
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.transition_ratio = transition_ratio

        # Initialize first cycle
        self.current_fps_range = self._generate_fps_range()
        self.current_cycle_duration = random.uniform(min_duration, max_duration)
        print(f'Starting cycle: {self.current_fps_range}, {self.current_cycle_duration}')
        self.cycle_start_time = time.time()
        
        # Transition state
        self.is_transitioning = False
        self.transition_start_time = None
        self.transition_duration = 0
        self.transition_progress = 0.0
        self.next_fps_range = None
        self.next_cycle_duration = None
        
        # Frame processing state
        self.last_processed_time = None

    def _generate_fps_range(self):
        """Generate random FPS range within configured bounds"""
        min_val = random.uniform(self.min_fps, (self.max_fps-self.min_fps)/2)
        max_val = random.uniform(min_val, self.max_fps)
        return {'min_fps': min_val, 'max_fps': max_val}

    def _update_cycle_state(self, current_time):
        """Update cycle state and handle transitions"""
        elapsed_time = current_time - self.cycle_start_time
        
        if self.is_transitioning:
            # Update transition progress
            transition_elapsed = current_time - self.transition_start_time
            self.transition_progress = transition_elapsed / self.transition_duration
            
            if self.transition_progress >= 1.0:
                # Complete transition
                self.current_fps_range = self.next_fps_range
                self.current_cycle_duration = self.next_cycle_duration
                print(f'Switched to new cycle: {self.current_fps_range}, {self.current_cycle_duration}')
                self.cycle_start_time = current_time - (transition_elapsed - self.transition_duration)
                self.is_transitioning = False
                self.next_fps_range = None
                self.next_cycle_duration = None
                self.transition_progress = 0.0
        else:
            # Check if should start transition
            remaining_time = self.current_cycle_duration - elapsed_time
            if remaining_time <= self.current_cycle_duration * self.transition_ratio:
                # Start transition to new cycle
                self.is_transitioning = True
                self.transition_start_time = current_time
                self.next_fps_range = self._generate_fps_range()
                self.next_cycle_duration = random.uniform(self.min_duration, self.max_duration)
                self.transition_duration = remaining_time

    def _calculate_target_fps(self, current_time):
        """Calculate target FPS based on current cycle state"""
        elapsed_time = current_time - self.cycle_start_time
        
        if self.is_transitioning:
            # Current cycle calculation
            current_phase = (elapsed_time % self.current_cycle_duration) * (2 * math.pi / self.current_cycle_duration)
            current_sine = math.sin(current_phase)
            current_fps = (current_sine + 1)/2 * (self.current_fps_range['max_fps'] - self.current_fps_range['min_fps']) + self.current_fps_range['min_fps']
            
            # Next cycle calculation
            next_cycle_start = self.cycle_start_time + self.current_cycle_duration - self.transition_duration
            next_elapsed = current_time - next_cycle_start
            next_phase = (next_elapsed % self.next_cycle_duration) * (2 * math.pi / self.next_cycle_duration)
            next_sine = math.sin(next_phase)
            next_fps = (next_sine + 1)/2 * (self.next_fps_range['max_fps'] - self.next_fps_range['min_fps']) + self.next_fps_range['min_fps']
            
            # Linear interpolation between cycles
            return current_fps * (1 - self.transition_progress) + next_fps * self.transition_progress
        else:
            # Normal cycle calculation
            phase = (elapsed_time % self.current_cycle_duration) * (2 * math.pi / self.current_cycle_duration)
            sine_val = math.sin(phase)
            return (sine_val + 1)/2 * (self.current_fps_range['max_fps'] - self.current_fps_range['min_fps']) + self.current_fps_range['min_fps']

    def __call__(self, system, frame):
        """Determine if current frame should be processed"""
        current_time = time.time()
        self._update_cycle_state(current_time)
        
        # Calculate target FPS and required interval
        target_fps = self._calculate_target_fps(current_time)
        required_interval = 1.0 / target_fps
        
        # First frame always processed
        if self.last_processed_time is None:
            self.last_processed_time = current_time
            return True
        
        # Check if enough time has passed since last processed frame
        elapsed = current_time - self.last_processed_time
        if elapsed >= required_interval:
            self.last_processed_time = current_time
            return True
        return False
    
# test code
if __name__ == '__main__':
    filter = DynamicFilter(1, 10, 20, 60)
    while True:
        if filter():
            print('Processing frame')
        else:
            print('Skipping frame')
        time.sleep(0.03)