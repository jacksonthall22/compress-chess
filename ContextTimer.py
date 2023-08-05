# https://stackoverflow.com/a/69156219/7304977

from time import perf_counter


class ContextTimer:

    def __init__(self, precision: int = 3):
        self.precision = precision

    def __enter__(self):
        self.time = perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.time = perf_counter() - self.time
        self.readout = f'Time: {self.time:.{self.precision}f} seconds'
        print(self.readout)
