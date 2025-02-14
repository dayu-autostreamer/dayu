from dataclasses import dataclass

@dataclass
class StatsEntry:
    # The timestamp of the inference
    timestamp: float = 0.0
    # The queue length at the time of the inference
    queue_length: int = 0
    # The model index of the inference
    cur_model_index: int = 0
    # The accuracy of the model
    cur_model_accuracy: float = 0.0
    # The processing latency of the inference
    processing_latency: int = 0
    # The number of detected targets
    target_nums: int = 0
    # The average confidence of the detected targets
    avg_confidence: float = 0.0
    # The standard deviation of the confidence of the detected targets
    std_confidence: float = 0.0
    # The standard deviation of the confidence of the detected targets
    avg_size: float = 0.0
    # The standard deviation of the size of the detected targets
    std_size: float = 0.0
    # The brightness of the image
    brightness: float = 0.0
    # The contrast of the image
    contrast: float = 0.0

    def __str__(self) -> str:
        return (
            f"Stats Entry:\n"
            f"  Timestamp: {self.timestamp:.2f}\n"
            f"  Queue Length: {self.queue_length}\n"
            f"  Model Index: {self.cur_model_index}\n"
            f"  Model Accuracy: {self.cur_model_accuracy:.2f}\n"
            f"  Processing Latency: {self.processing_latency:.4f}s\n"
            f"  Targets: {self.target_nums}\n"
            f"  Confidence: {self.avg_confidence:.2f}±{self.std_confidence:.2f}\n"
            f"  Size: {self.avg_size:.2f}±{self.std_size:.2f}\n"
            f"  Image Stats: brightness={self.brightness:.1f}, contrast={self.contrast:.1f}"
        )
    
class StatsManager:
    def __init__(self, time_window: int = 10.0):
        # initialize a deque to store the stats
        from collections import deque
        self.stats = deque()
        self.time_window = time_window

    def update_stats(self, entry: StatsEntry):
        '''
        Update the stats.
        Called when a new inference is done.
        Remove the outdated stats and append the new stats.
        '''
        # append the new stats
        self.stats.append(entry)
        # remove the outdated stats
        while self.stats and entry.timestamp - self.stats[0].timestamp > self.time_window:
            self.stats.popleft()
