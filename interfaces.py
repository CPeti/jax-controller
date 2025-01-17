from abc import ABC, abstractmethod

class IDisturbance(ABC):
    """
    Class to represent a disturbance in the system. 
    """
    
    @abstractmethod
    def disturb(self, control_value: float):
        pass

class IPlant(ABC):
    """
    Interface for a plant
    """

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
    """
    Interface for a controller
    """
    @abstractmethod
    def control(self, error) -> float:
        """
        Update the controller based on the error
        """
        pass

    @abstractmethod
    def reset(self):
        """
        Reset the controller state
        """
        pass



