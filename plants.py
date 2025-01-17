from interfaces import IPlant, IDisturbance

class Bathtub(IPlant):
    """
    Class for a bathtub plant
    """

    def __init__(self, state: float, disturbance: IDisturbance, area: float = 1.0, exit_area: float = 0.01):
        """
        Initialize the bathtub plant with a state and disturbance
        """
        # system state is the water level
        super().__init__(state, disturbance)

        # constants
        self.initial_state = state
        self.g = 9.81
        self.area = area
        self.exit_area = exit_area

    def reset(self):
        """
        Reset the state of the bathtub
        """
        self.state = self.initial_state

    def get_exit_velocity(self):
        """
        Get the water velocity based on the state
        """
        return (2 * self.g * self.state) ** 0.5
    
    def get_flow_rate(self):
        """
        Get the flow rate based on the state
        """
        return self.get_exit_velocity() * self.exit_area

    def update(self, control_value: float, dt: float = 1.0):
        """
        Update the bathtub plant based on the control value
        """
        delta_volume = self.disturbance.disturb(control_value) - self.get_flow_rate() * dt
        self.state = max(self.state + delta_volume / self.area, 0)