import statistics
import sys

sys.path.insert(0, 'evoman')
import os
from environment import Environment
from multiprocessing import Pool
import multiprocessing

import numpy as np
import random
import time


def make_controller_config(controller, controller_data):
    """
    This function make the controller configuration in preparation for the simulation
    Args:
        controller: Should inherit from evoman/Controller
        controller_data: This is the data the controller need to act.

    Returns:
        dict containing the controller configuration
    """

    return {
        'controller': controller,
        'controller_data': controller_data,
    }


def make_simulation_config(player=None, enemy=None, enemies=[1], speed='fastest', hidden=False):
    """
    Args:
        player: optional - A ControllerConfig. Use MakeControllerConfig to create one.
        enemy: optional - A ControllerConfig. Use MakeControllerConfig to create one.
        enemies: enemies to fight in the simulation
        speed: speed to use in the simulation
        hidden: hidden layer to use in the simulation (Not used)

    Returns:
        dict containing the configuration for the simulation
    """
    simulation = {
        'enemies': enemies,
        'speed': speed,
        'hidden': False,
    }
    if player is not None:
        simulation['player'] = player
    if enemy is not None:
        simulation['enemy'] = enemy
    return simulation


def simulate_base(simulations_dict, commonSimulationConfig):
    """
    This function is the base for the simulation.
    It is able to run simulations in parallel depending on the parameters
    Args:
        simulations_dict: simulation dictionary
        commonSimulationConfig: common simulation config

    Returns:
        results of the simulation
    """
    results = {}

    if 'parallel' in commonSimulationConfig and commonSimulationConfig['parallel']:
        inputIds = []
        inputs = []
        for simulationId, simulationCfg in simulations_dict.items():
            inputIds.append(simulationId)
            inputs.append(simulationCfg)

        os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "true"

        with Pool(int(2 + multiprocessing.cpu_count() / 2)) as p:
            outputs = p.map(simulate_one, inputs)
            for simulationId, result in zip(inputIds, outputs):
                results[simulationId] = result
    else:
        # Sequential version
        for simulationId, simulationCfg in simulations_dict.items():
            results[simulationId] = simulate_one(simulationCfg)

    return results


def simulate(simulations_dict):
    """
    Function to run the simulation. As default it runs in parallel.
    Args:
        simulations_dict: A list of SimulationConfig. Use MakeSimulationConfig to create them.

    Returns:
    """
    return simulate_base(simulations_dict, {'parallel': True})


def simulate_one(simulationCfg):
    """
    Function to run a single simulation
    Args:
        simulationCfg: configuration to use in the simulation
    Returns:
        results of the simulation
    """
    envParams = {}
    playParams = {}
    for name, controllerCfg in simulationCfg.items():
        if name == 'player':
            envParams['player_controller'] = controllerCfg['controller']
            playParams['pcont'] = controllerCfg['controller_data']
        if name == 'enemy':
            envParams['enemy_controller'] = controllerCfg['controller']
            playParams['econt'] = controllerCfg['controller_data']

    if 'player' not in simulationCfg:
        raise Exception('A player controller MUST be specified')

    envParams['enemymode'] = 'static'
    if 'enemy' in simulationCfg:
        envParams['enemymode'] = 'ai'
    if 'randomini' in simulationCfg:
        envParams['randomini'] = simulationCfg['randomini']

    if simulationCfg['hidden']:
        os.putenv('SDL_VIDEODRIVER', 'fbcon')
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    # Seed with microseconds
    np.random.seed(int(time.time() * 1000000) % 1000000)
    random.seed(int(time.time() * 1000000) % 1000000)

    enemies = simulationCfg['enemies']
    if len(enemies) == 1:
        env = Environment(
            playermode="ai",
            logs="off",
            enemies=enemies,
            speed=simulationCfg['speed'],
            **envParams,
        )
        f, p, e, t = env.play(**playParams)
        return f, p, e, t
    else:
        results = {}
        for enemy in enemies:
            env = Environment(
                playermode="ai",
                logs="off",
                enemies=[enemy],
                speed=simulationCfg['speed'],
                **envParams,
            )
            result = env.play(**playParams)
            results[enemy] = result
        return results


def make_simulate(extraSimulationCfg):
    """
    Function to make a simulation
    Args:
        extraSimulationCfg: extra configuration to set for running a simulation

    Returns:
        results of the simulation
    """

    def new_simulate(simulations_dict):
        # Use extraSimulationCfg to transform simulations_dict into new_simulation_dict
        new_simulations_dict = dict(simulations_dict)
        for simulationId, simulationCfg in new_simulations_dict.items():
            for key, value in extraSimulationCfg.items():
                simulationCfg[key] = value

        # call simulate with new_simulations_dict
        return simulate_base(new_simulations_dict, extraSimulationCfg)

    return new_simulate


def get_effective_fitness_of_individual_restuls(individual_results):
    """
    Return the fitness of multiple simulation on different enemies.
    The formula applied is equal ot the original one: mean(fitness) - std(fitness)
    Args:
        individual_results: Results of multiple enemies simulation

    Returns:
        Fitness value
    """
    if isinstance(individual_results, dict):
        total_fitness = []
        for enemy, result in individual_results.items():
            f, p, e, t = result
            total_fitness.append(f)
        return statistics.mean(total_fitness) - statistics.stdev(total_fitness)
    else:
        return individual_results[0]
