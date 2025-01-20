from interfaces import IPlant, IController
import jax

class Consys():
    """
    Class for connecting the control system to the simulator
    """

    def __init__(self, plant: IPlant, controller: IController, target: float, dt: float = 1.0, lr: float = 0.01, jit: bool = True):
        """
        Initialize the control system with a plant and controller
        """
        self.plant = plant
        self.controller = controller
        self.target = target
        self.dt = dt
        self.lr = lr

        self.gradfunc = jax.value_and_grad(self.run_one_epoch, argnums=0)
        if jit:
            self.gradfunc = jax.jit(self.gradfunc)
        
        self.history = {
            "MSEs": []
        }

    def run_one_epoch(self, weights, steps: int = 100):
        self.plant.reset()
        self.controller.reset()
        self.controller.set_weights(weights)
        control = 0
        for _ in range(steps):
            self.plant.update(control, self.dt)
            error = self.target - self.plant.state
            print(error)            
            control = self.controller.control(error, self.dt)
        return self.controller.get_MSE()
    
    def run(self, epochs: int = 100, reset: bool = True):
        """
        Run the control system for a number of epochs
        """
        if reset:
            self.reset_history()

        weights = self.controller.get_init_weights()

        for i in range(epochs):
            mse, gradients = self.gradfunc(weights)
            weights = self.controller.update_weights(weights, gradients, lr=self.lr)
            self.controller.log_data(self.history)
            self.history["MSEs"].append(mse)

    def reset_history(self):
        self.history = {
            "MSEs": []
        }