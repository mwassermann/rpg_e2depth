import torch
import time
import numpy as np
import atexit

cuda_timers = {}
mps_cpu_timers = {}

class CudaTimer:
    def __init__(self, timer_name=''):
        self.timer_name = timer_name
        if self.timer_name not in cuda_timers:
            cuda_timers[self.timer_name] = []

        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        self.start.record()
        return self

    def __exit__(self, *args):
        self.end.record()
        torch.cuda.synchronize()
        cuda_timers[self.timer_name].append(self.start.elapsed_time(self.end))




class Timer:
    def __init__(self, timer_name=''):
        self.timer_name = timer_name
        if self.timer_name not in mps_cpu_timers:
            mps_cpu_timers[self.timer_name] = []

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start  # measured in seconds
        self.interval *= 1000.0  # convert to milliseconds
        mps_cpu_timers[self.timer_name].append(self.interval)


def print_timing_info():
    print('== Timing statistics ==')
    for timer_name, timing_values in [*cuda_timers.items(), *mps_cpu_timers.items()]:
        timing_value = np.mean(np.array(timing_values))
        if timing_value < 1000.0:
            print('{}: {:.2f} ms'.format(timer_name, timing_value))
        else:
            print('{}: {:.2f} s'.format(timer_name, timing_value / 1000.0))

class AutoTimer:
    def __init__(self, timer_name, device):
        self.device = device
        if self.device.type == 'cuda':
            self.timer = CudaTimer(timer_name)
        else:
            self.timer = Timer(timer_name)
        

    def __enter__(self):
        return self.timer.__enter__()

    def __exit__(self, *args):
        self.timer.__exit__(*args)


# this will print all the timer values upon termination of any program that imported this file
atexit.register(print_timing_info)