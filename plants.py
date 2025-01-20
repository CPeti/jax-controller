from interfaces import IPlant, IDisturbance
import jax.numpy as jnp

class Bathtub(IPlant):
    def __init__(self, state: float, disturbance: IDisturbance, area: float = 1.0, exit_area: float = 0.01):
        # system state is the water level
        super().__init__(state, disturbance)

        # constants
        self.initial_state = state
        self.g = 9.81
        self.area = area
        self.exit_area = exit_area

    def reset(self):
        self.state = self.initial_state

    def get_exit_velocity(self):
        return jnp.sqrt(2 * self.g * self.state)
    
    def get_flow_rate(self):
        return self.get_exit_velocity() * self.exit_area

    def update(self, control_value: float, dt: float = 1.0):
        delta_volume = self.disturbance.disturb(control_value) - self.get_flow_rate() * dt
        #self.state = max(self.state + delta_volume / self.area, 0)
        self.state = self.state + delta_volume / self.area
        self.state = jnp.clip(self.state, 0)


class Cournot(IPlant):

    def __init__(self, state: float, disturbance: IDisturbance, p_max: float = 100):
        super().__init__(state, disturbance)

        self.p_max = p_max
        self.initial_state = state