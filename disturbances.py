import numpy as np
from interfaces import IDisturbance


class AdditiveDisturbance(IDisturbance):
    def __init__(self, mean: float = 0.0, std_dev: float = 1.0):
        self.mean = mean
        self.std_dev = std_dev
    
    def disturb(self, control_value: float):
        return control_value + self.mean + self.std_dev * np.random.randn()
    
class MultiplicativeDisturbance(IDisturbance):
    def __init__(self, mean: float = 1.0, std_dev: float = 0.1):
        self.mean = mean
        self.std_dev = std_dev
    
    def disturb(self, control_value: float):
        return control_value * (self.mean + self.std_dev * np.random.randn())
    
class UniformDisturbance(IDisturbance):
    def __init__(self, low: float = 0.0, high: float = 1.0):
        self.low = low
        self.high = high
    
    def disturb(self, control_value: float):
        return control_value + np.random.uniform(self.low, self.high)

class CustomDisturbance(IDisturbance):
    def __init__(self, function: callable):
        self.function = function

    def disturb(self, control_value: float):
        return self.function(control_value)