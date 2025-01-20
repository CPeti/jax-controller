from abc import ABC, abstractmethod
from numpy import isnan

class IDisturbance(ABC):    
    @abstractmethod
    def disturb(self, control_value: float):
        pass

class IPlant(ABC):

    def __init__(self, state: float, disturbance: IDisturbance):
        """
        Initialize the plant state and behavior
        """
        self.state = state
        self.disturbance = disturbance

    @abstractmethod
    def update(self, control_value):
        """
        Update the plant behavior based on the control value
        """
        pass

    @abstractmethod
    def reset(self):
        """
        Reset the plant state
        """
        pass

class IController(ABC):
    def __init__(self):
        self.derivative = 0.0
        self.integral = 0.0
        self.prev_error = 0.0
        self.MSE = 0.0
        self.steps = 0

    @abstractmethod
    def set_weights(self, weights):
        """
        Set the weights of the controller
        """
        pass

    @abstractmethod
    def get_init_weights(self):
        """
        Get the initial weights of the controller
        """
        pass

    @abstractmethod
    def control(self, error) -> float:
        """
        Update the controller based on the error
        """
        pass

    @abstractmethod
    def log_data(self, history: dict):
        """
        Log data to the history dictionary
        """
        pass

    def step(self, error, dt):
        # check for overflow
        self.derivative = (error - self.prev_error) / dt
        self.integral = self.integral + error * dt
        self.prev_error = error
        self.MSE = self.MSE + (error ** 2)
        self.steps += 1

    def get_MSE(self):
        return self.MSE / self.steps
    
    def reset(self):
        self.derivative = 0.0
        self.integral = 0.0
        self.prev_error = 0.0
        self.MSE = 0.0
        self.steps = 0

    def update_weights(self, weights, gradients, d: int = -1, lr: float = 0.01):
        # check if gradients are nan
        assert not any(isnan(g).any() for g in gradients), "Gradients are NaN"

        for i in range(len(weights)):
            weights[i] = weights[i] + d * lr * gradients[i]
        self.set_weights(weights)
        return weights


