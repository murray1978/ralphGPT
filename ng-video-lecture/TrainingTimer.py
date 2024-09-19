import time


class TrainingTimer:
    def __init__(self, max_iters, eval_interval):
        self.max_iters = max_iters
        self.eval_interval = eval_interval
        self.start_time = time.time()
        self.iterations_completed = 0
        self.time_per_interval = []

    def update(self):
        elapsed_time = time.time() - self.start_time
        self.time_per_interval.append(elapsed_time)
        self.start_time = time.time()  # Reset start time for the next interval
        self.iterations_completed += self.eval_interval

        avg_time_per_interval = sum(self.time_per_interval) / len(self.time_per_interval)
        remaining_iters = self.max_iters - self.iterations_completed
        remaining_time = remaining_iters / self.eval_interval * avg_time_per_interval

        return remaining_time

    def format_time(self, seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        return f"{hours}h {minutes}m {seconds}s"