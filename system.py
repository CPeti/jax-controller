from interfaces import IPlant, IController
import jax
import numpy as np
import sys
class Consys():
    """
    Class for connecting the control system to the simulator
    """

    def __init__(self, plant: IPlant, controller: IController, target: float, dt: float = 1.0, lr: float = 0.01, steps: int = 100, jit: bool = True):
        """
        Initialize the control system with a plant and controller
        """
        self.plant = plant
        self.controller = controller
        self.target = target
        self.dt = dt
        self.lr = lr
        self.steps_per_epoch = steps

        self.gradfunc = jax.value_and_grad(self.run_one_epoch, argnums=0)
        if jit:
            self.gradfunc = jax.jit(self.gradfunc)
        
        self.history = {
            "MSEs": []
        }

    def run_one_epoch(self, weights):
        self.plant.reset()
        self.controller.reset()
        self.controller.set_weights(weights)
        error = 0

        for _ in range(self.steps_per_epoch):
            control = self.controller.control(error, self.dt)
            self.plant.update(control, self.dt)
            error = self.target - self.plant.state
        return self.controller.get_MSE()
    
    def run(self, epochs, reset: bool = True):
        """
        Run the control system for a number of epochs
        """
        if reset:
            self.reset_history()

        weights = self.controller.get_init_weights()
        i = 0
        while i < epochs:
            mse, gradients = self.gradfunc(weights)
            if gradient_invalid(gradients):
                weights = self.controller.get_init_weights() # reset weights if all zero
                self.reset_history()
                i = 0
                #print("Invalid gradient, resetting weights")
                continue
            weights = self.controller.update_weights(weights, gradients, lr=self.lr)
            self.controller.log_data(self.history)
            self.history["MSEs"].append(mse)
            i += 1

    def reset_history(self):
        self.history = {
            "MSEs": []
        }

def gradient_invalid(matrices):
    if any(np.isnan(g).any() for g in matrices):
        return True
    if np.all(matrices == 0):
        return True
    for _, matrix in enumerate(matrices):
        total_elements = matrix.size  # Total number of elements in the matrix
        zero_count = np.count_nonzero(matrix == 0)  # Count of zero elements
        
        # Check if zero count equals total elements
        if zero_count >= total_elements * 2/3 :
            return True
    return False

