import os

from .time_estimation import Timer
from core.lib.common import FileOps, Context


class OverheadEstimator:
    def __init__(self, method_name, save_dir):

        self.timer = Timer(f'Runtime Overhead of {method_name}')
        self.overhead_file = Context.get_file_path(os.path.join(save_dir, f'{method_name}_Overhead.txt'))
        self.latest_overhead = 0

        self.clear()

    def __enter__(self):
        self.timer.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.timer.__exit__(exc_type, exc_val, exc_tb)
        self.latest_overhead = self.timer.get_elapsed_time()
        self.write_overhead(self.latest_overhead)

    def get_latest_overhead(self):
        return self.latest_overhead

    def get_average_overhead(self):
        with open(self.overhead_file, 'r') as f:
            lines = f.readlines()
            overhead_list = [float(line.strip()) for line in lines]
            return sum(overhead_list) / len(overhead_list)

    def write_overhead(self, overhead):
        with open(self.overhead_file, 'a') as f:
            f.write(f'{overhead}\n')

    def clear(self):
        self.latest_overhead = 0
        FileOps.remove_file(self.overhead_file)
