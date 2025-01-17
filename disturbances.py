import numpy as np
from interfaces import IDisturbance


class AdditiveDisturbance(IDisturbance):
    """
    Add gaussian noise. 
    """
    
    def __init__(self, mean: float = 0.0, std_dev: float = 1.0):
        """
        Initialize the disturbance with a mean and standard deviation
        """
        self.mean = mean
        self.std_dev = std_dev
    
    def disturb(self, control_value: float):
        return control_value + self.mean + self.std_dev * np.random.randn()
    
class MultiplicativeDisturbance(IDisturbance):
    """
    Add multiplicative noise. 
    """
    def __init__(self, mean: float = 1.0, std_dev: float = 0.1):
        """
        Initialize the disturbance with a mean and standard deviation
        """
        self.mean = mean
        self.std_dev = std_dev
    
    def disturb(self, control_value: float):
        return control_value * (self.mean + self.std_dev * np.random.randn())

class CustomDisturbance(IDisturbance):
    def __init__(self, function: callable):
        self.function = function

    def disturb(self, control_value: float):
        return self.function(control_value)