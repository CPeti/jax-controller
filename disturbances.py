import numpy as np
from interfaces import IDisturbance


class NormalDisturbance(IDisturbance):
    def __init__(self, mean: float = 0.0, std_dev: float = 1.0):
        self.mean = mean
        self.std_dev = std_dev
    
    def __call__(self) -> float:
        return self.mean + self.std_dev * np.random.randn()
    
class UniformDisturbance(IDisturbance):
    def __init__(self, low: float = 0.0, high: float = 1.0):
        self.low = low
        self.high = high
    
    def __call__(self) -> float:
        return np.random.uniform(self.low, self.high)

class CustomDisturbance(IDisturbance):
    def __init__(self, function: callable):
        self.function = function

    def __call__(self) -> float:
        return self.function()