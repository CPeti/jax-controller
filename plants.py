from interfaces import IPlant, IDisturbance
import jax.numpy as jnp
from collections import deque

class Bathtub(IPlant):
    def __init__(self, state: float, disturbance: IDisturbance, area: float = 1.0, exit_area: float = 0.01):
        # system state is the water level
        super().__init__(state, disturbance)

        # constants
        self.initial_state = state
        self.g = 9.81
        self.area = area
        self.exit_area = exit_area
        self.delta_volume = 0

    def reset(self):
        self.state = self.initial_state
        self.delta_volume = 0

    def get_exit_velocity(self):
        return jnp.sqrt(2 * self.g * self.state) # sqrt(2gh)
    
    def get_flow_rate(self):
        return self.get_exit_velocity() * self.exit_area

    def update(self, control_value: float, dt: float = 1.0):
        self.delta_volume = control_value + self.disturbance() - self.get_flow_rate() * dt
        self.state = self.state + self.delta_volume / self.area
        self.state = jnp.minimum(jnp.maximum(self.state, 0), 1e4)


class Cournot(IPlant):

    def __init__(self, state: float, disturbance: IDisturbance, p_max: float, cm1: float = 0.5, cm2: float = 0.5, q1: float = 0.0, q2: float = 0.0):
        super().__init__(state, disturbance)

        self.p_max = p_max
        self.p1 = self.Producer(cm1, q1)
        self.p2 = self.Producer(cm2, q1)
        q = self.p1.q + self.p2.q
        p = self.p_max - q
        self.state = self.p1.get_profit(p)
        self.cm1 = cm1
        self.cm2 = cm2
        self.q1 = q1
        self.q2 = q2

    def reset(self):
        self.p1 = self.Producer(self.cm1, self.q1)
        self.p2 = self.Producer(self.cm2, self.q1)
        q = self.p1.q + self.p2.q
        p = self.p_max - q
        self.state = self.p1.get_profit(p)

    class Producer():
        def __init__(self, cm: float, q: float = 0):
            self.cm = cm
            self.q = q

        def update(self, control_value: float):
            self.q += control_value
            self.q = jnp.minimum(jnp.maximum(self.q, 0), 1)

        def get_profit(self, p: float):
            return (p - self.cm) * self.q

    def update(self, control_value, _: float = 1.0):
        self.p1.update(control_value)
        self.p2.update(self.disturbance())

        q = self.p1.q + self.p2.q
        p = self.p_max - q
        
        self.state = self.p1.get_profit(p) # assume p1 is the only one that matters - state is p1's profit

class TemperaturePlant(IPlant):
    def __init__(self, state, disturbance, alpha: float = 0.1, beta: float = 0.5, T_env: float = 10.0):
        super().__init__(state, disturbance)
        self.initial_state = state
        self.alpha = alpha
        self.beta = beta
        self.T_env = T_env


    def update(self, control_value: float, dt: float = 1.0):
        """
        Update the temperature state based on control input and disturbance
        """
        self.state += (-self.alpha * (self.state - self.T_env) + self.beta * control_value + self.disturbance()) * dt

    def reset(self):
        """
        Reset the plant state to initial condition
        """
        self.state = self.initial_state