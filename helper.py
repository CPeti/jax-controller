import matplotlib.pyplot as plt

from plants import Bathtub, Cournot, TemperaturePlant
from disturbances import NormalDisturbance, UniformDisturbance
from controllers import PIDController, NeuralController
from system import Consys

import config as cfg


class Helper():
    def __init__(self, config):
        self.config = config
        if config.plant == 'bathtub':
            self.disturbance = UniformDisturbance(-config.bathtub['DISTURBANCE_RANGE'], config.bathtub['DISTURBANCE_RANGE'])
            self.plant = Bathtub(config.bathtub['INITIAL_STATE'], self.disturbance, config.bathtub['AREA'], config.bathtub['EXIT_AREA'])
            self.target = config.bathtub['TARGET']
        elif config.plant == 'cournot':
            self.disturbance = UniformDisturbance(-config.cournot['DISTURBANCE_RANGE'], config.cournot['DISTURBANCE_RANGE'])
            self.plant = Cournot(config.cournot['INITIAL_PROFIT'], self.disturbance, config.cournot['P_MAX'], config.cournot['COST_1'], config.cournot['COST_2'], config.cournot['Q1'], config.cournot['Q2'])
            self.target = config.cournot['TARGET']
        elif config.plant == 'temperature':
            self.disturbance = UniformDisturbance(-config.cournot['DISTURBANCE_RANGE'], config.cournot['DISTURBANCE_RANGE'])
            self.plant = TemperaturePlant(config.temperature['T_INITIAL'], self.disturbance, config.temperature['ALPHA'], config.temperature['BETA'], config.temperature['T_ENV'])
            self.target = config.temperature['TARGET']
        else:
            raise ValueError("Invalid plant")

        if config.controller == 'pid':
            self.controller = PIDController(config.pid['KP'], config.pid['KI'], config.pid['KD'])
        elif config.controller == 'neural':
            self.controller = NeuralController(config.neural['LAYER_SIZES'], config.neural['ACTIVATION'], layer_init=config.neural['layer_init'], param_range=config.neural['param_range'])
        else:
            raise ValueError("Invalid controller")

        self.consys = Consys(self.plant, self.controller, self.target, config.consys['DT'], config.consys['LR'], config.consys['STEPS_PER_EPOCH'], config.consys['JIT'])

    def run(self):
        self.consys.run(self.config.consys['EPOCHS'])

    def plot_MSE(self):
        plt.plot(self.consys.history["MSEs"])
        plt.xlabel("Epoch")
        plt.ylabel("MSE")
        plt.title(f"Mean Squared Error - {self.config.controller} controller with {self.config.plant} plant")
        plt.show()

    def plot_k_values(self):
        if self.config.controller == 'pid':
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            ax[0].plot(self.consys.history['Kp'])
            ax[0].set_title('kp')
            ax[1].plot(self.consys.history['Ki'])
            ax[1].set_title('ki')
            ax[2].plot(self.consys.history['Kd'])
            ax[2].set_title('kd')
            plt.xlabel("Epoch")
            plt.ylabel("Value")
            plt.suptitle(f"PID controller values - {self.config.plant} plant")
            plt.show()
        else:
            return
    
    def simulate(self):
        self.plant.reset()
        self.controller.reset()

        states = [self.plant.state]
        error = 0
        for i in range(300):
            control = self.controller.control(error, self.config.consys['DT'])
            self.plant.update(control, self.config.consys['DT'])
            error = self.target - self.plant.state
            states.append(self.plant.state)

        plt.plot(states)
        plt.plot([self.target] * len(states), linestyle='--', color='black')
        plt.xlabel('Time')
        plt.ylabel('State')
        plt.title(f"Simulation - {self.config.controller} controller with {self.config.plant} plant")
        plt.show()

    def demo(self):
        self.run()
        self.plot_MSE()
        self.plot_k_values()
        self.simulate()

if __name__ == "__main__":
    helper = Helper(cfg)
    helper.demo()


