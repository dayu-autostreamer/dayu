from .utils import singleton

@singleton
class Counter:

    def __init__(self):
        self.counts = {}

    def get_count(self, name='default'):
        if name in self.counts:
            self.counts[name] += 1
        else:
            self.counts[name] = 0
        return self.counts[name]

    def reset_count(self, name='default'):
        if name in self.counts:
            del self.counts[name]

    def reset_all_counts(self):
        self.counts = {}

    def get_all_counts(self):
        return self.counts
