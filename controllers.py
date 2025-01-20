from interfaces import IController
import jax.numpy as jnp
import numpy as np

class NaiveController(IController):
    def __init__(self, value: float = 1.0):
        super().__init__()
        self.value = value

    def control(self, error: float):
        return error * self.value
    
    def get_weights(self):
        return [self.value]
    
    def reset(self):
        pass
    
class PIDController(IController):
    def __init__(self, kp: float = 1.0, ki: float = 0.0, kd: float = 0.0):
        super().__init__()
        self.initial_weights = [kp, ki, kd]
        self.kp = kp
        self.ki = ki
        self.kd = kd

    def set_weights(self, weights):
        self.kp, self.ki, self.kd = weights

    def get_init_weights(self):
        return self.initial_weights

    def control(self, error: float, dt: float = 1.0):
        self.step(error, dt)
        return self.kp * error + self.ki * self.integral + self.kd * self.derivative
    
    def log_data(self, history: dict):
        if "Kp" not in history:
            history["Kp"] = []
            history["Ki"] = []
            history["Kd"] = []
        history["Kp"].append(self.kp)
        history["Ki"].append(self.ki)
        history["Kd"].append(self.kd)

        return history       

class NeuralController(IController):
    def __init__(self, layer_sizes: list, activation: str = 'relu'):
        super().__init__()
        # assert that the input layer size is 3 and the output layer size is 1
        assert layer_sizes[0] == 3, "Input layer size must be 3"
        assert layer_sizes[-1] == 1, "Output layer size must be 1"
        self.layer_sizes = layer_sizes
        self.weights = self.get_init_weights()
        
        if activation == 'relu':
            self.activation = lambda x: jnp.maximum(0, x)
        elif activation == 'tanh':
            self.activation = jnp.tanh
        elif activation == 'sigmoid':
            self.activation = lambda x: 1 / (1 + jnp.exp(-x))
        elif activation == 'linear':
            self.activation = lambda x: x
        else:
            raise ValueError("Invalid activation function")

    def forward(self, inp):
        x = inp
        for w in self.weights[:-1]:
            x = jnp.append(x, 1)
            x = self.activation(w @ x)
        x = self.weights[-1] @ jnp.append(x, 1)
        return x

    def set_weights(self, weights):
        self.weights = weights

    def get_init_weights(self):
        return [jnp.concatenate([np.random.uniform(-1, 1, (m, n)), np.random.uniform(-1, 1, (m, 1))], axis=1) for m, n in zip(self.layer_sizes[1:], self.layer_sizes[:-1])]

    def control(self, error: float, dt: float = 1.0):
        self.step(error, dt)
        return self.forward(jnp.array([error, self.integral, self.derivative]))[0]
    
    def log_data(self, history: dict):
        pass
    