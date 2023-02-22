# imports framework
import sys

sys.path.insert(0, 'evoman')
import os
from controller import Controller
from z_evoman_ea import EvomanEA
from z_evoman_simulation import simulate, make_controller_config, make_simulation_config, make_simulate
import random
import pickle


class EvomanSimple(EvomanEA):
    """
    Easiest class the implement the interface EvomanEA
    """

    def __init__(self, train_controllers=['player', 'enemy']):
        self.train_controllers = train_controllers

    def controllers(self):
        """
        Method to use in order to override the controllers
        Returns:
            Dictionary with the used controllers

        """
        return {
            'player': player_controller(),
            'enemy': enemy_controller(),
        }

    def do_train(self, simulate):
        """
        Method use for training the controllers and use self.set_best_controller_data to save the best agents

        For this simple implementation there is no actual training
        Each controller will be making the same actions everytime.
        The training phase will choose those actions at random.

        Args:
            simulate: function to call in order to make the simulation

        Returns:

        """

        controller_data_dict = {}

        for name in self.train_controllers:
            if name == 'player':
                controller_data_dict['player'] = random.randint(0, 10)
            if name == 'enemy':
                controller_data_dict['enemy'] = random.randint(0, 10)

        self.set_best_controller_data_dict(controller_data_dict)

    # Override
    def train(self, simulate):
        self.do_train(simulate)

    # Override
    def run(self, simulate):
        controller_data_dict_dict = {
            'simulation_1': self.get_best_controller_data_dict()
        }
        results = self.run_simulate(simulate, controller_data_dict_dict)
        return results['simulation_1']

    def run_simulate(self, simulate, controller_data_dict_dict):
        """
        Method use in order to prepare and run the simulation with the propper params
        Args:
            simulate: Function to call in order to make the simulation
            controller_data_dict_dict: Dictionary containing the controller data to use in the simulation

        Returns:
            Results of the simulation
        """
        controllers = self.controllers()
        simulations_dict = {}
        for simulationId, controller_data_dict in controller_data_dict_dict.items():
            # Calling controllers here is important as it creates new controller for each simulation
            controllers = self.controllers()
            simulationConfigParams = {}
            for key, data in controller_data_dict.items():
                simulationConfigParams[key] = make_controller_config(
                    controller=controllers[key],
                    controller_data=data,
                )
            simulations_dict[simulationId] = make_simulation_config(**simulationConfigParams)

        return simulate(simulations_dict)

    def get_best_controller_data_dict(self):
        """
        Method to return the best controller data
        Returns:
            Dictionary containing the best controller data
        """
        return self.best_controller_data_dict

    def set_best_controller_data_dict(self, controller_data_dict):
        """
        Method to set the best controller data
        Returns:
            Dictionary containing the best controller data
        """
        self.best_controller_data_dict = controller_data_dict

    # Override
    def save(self, filename):
        filename = filename + '.pickle'
        data = self.get_best_controller_data_dict()
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    # Override
    def load(self, filename):
        filename = filename + ".pickle"
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            self.set_best_controller_data_dict(data)


class player_controller(Controller):
    """
    Class to use in order to controller the player.
    """

    def control(self, inputs, controller_data):
        rng = random.Random(controller_data)
        left = rng.randint(0, 1)
        right = rng.randint(0, 1)
        jump = rng.randint(0, 1)
        shoot = rng.randint(0, 1)
        release = rng.randint(0, 1)
        return [left, right, jump, shoot, release]


class enemy_controller(Controller):
    """
    Class to use in order to controller the enemy.
    """

    def control(self, inputs, controller_data):
        rng = random.Random(controller_data)
        attack1 = rng.randint(0, 1)
        attack2 = rng.randint(0, 1)
        attack3 = rng.randint(0, 1)
        attack4 = rng.randint(0, 1)
        return [attack1, attack2, attack3, attack4]


if __name__ == "__main__":
    run_mode = "test"
    filename = "data/evoman_simple"

    if len(sys.argv) > 1:
        run_mode = sys.argv[1]

    simulate_enemy2 = make_simulate({'enemies': [2]})

    if run_mode == "train":
        ea = EvomanSimple()
        ea.train(simulate_enemy2)
        ea.save(filename)

    if run_mode == "test":
        ea = EvomanSimple()
        ea.load(filename)
        ea.run(simulate_enemy2)
