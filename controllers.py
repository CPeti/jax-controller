from interfaces import IController

class NaiveController(IController):
    def __init__(self, value: float = 1.0):
        self.value = value

    def control(self, error: float):
        return error * self.value
    
    def reset(self):
        pass
    
class PIDController(IController):
    def __init__(self, kp: float = 1.0, ki: float = 0.0, kd: float = 0.0):
        self.initial_values = (kp, ki, kd)

        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.prev_error = 0.0

    def control(self, error: float, dt: float = 1.0):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative
    
    def reset(self):
        self.kp, self.ki, self.kd = self.initial_values
        self.integral = 0.0
        self.prev_error = 0.0