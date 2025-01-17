from interfaces import IPlant, IController

class Consys():
    """
    Class for connecting the control system to the simulator
    """

    def __init__(self, plant: IPlant, controller: IController, target: float):
        """
        Initialize the control system with a plant and controller
        """
        self.plant = plant
        self.controller = controller
        self.target = target

    def run(self, steps):
        """
        Run the control system for a given amount of steps
        """
        pass