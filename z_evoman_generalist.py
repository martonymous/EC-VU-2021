# imports framework
import sys
import time

import numpy as np

from demo_controller import player_controller
from z_evoman_simple import EvomanSimple
from z_evoman_simulation import make_simulate, get_effective_fitness_of_individual_restuls


class EvomanGeneralist(EvomanSimple):
    """
    Basic implementation of an Evoman for generalist tasks
    """

    def __init__(self, train_controllers=['player'], population_size=100, num_generations=100,
                 statistics_dir='./stats/', data_collector=None):
        super().__init__(train_controllers=train_controllers)
        self.num_generations = num_generations
        self.statistics_dir = statistics_dir
        self.population_size = population_size
        self.num_hidden = 10
        self.num_inputs = 20
        self.num_outputs = 5
        self.num_params = (self.num_inputs + 1) * self.num_hidden + (self.num_hidden + 1) * self.num_outputs
        self.rng = np.random.default_rng(int(time.time()))
        self.evaluator = None
        self.result_cache = {}
        self.__id_counter = int(1)
        self.data_collector = data_collector

    # Override
    def controllers(self):
        return {
            'player': player_controller(self.num_hidden),
            # 'enemy': enemy_controller(),
        }

    # Override
    def do_train(self, simulate):
        if self.train_controllers != ["player"]:
            raise Exception("Only player controller training is implemented")

        self.evaluator = self.single_evaluator(simulate)

        # evaluator is used as follows
        # fitness_array = evaluator(weights_array)
        # where:
        # weights_array.shape = (num_genomes, self.num_params)
        # fitness_array.shape = (num_genomes)

        # Initialize population
        print(f"Train generation number: 0")
        gen = 0
        population = self.rng.uniform(-1, 1, (self.population_size, self.num_params))
        fitness, ids = self.evaluator(population)
        self.report(gen, ids)
        print(
            f"The avg fitness of the population is {np.mean(fitness)}, the max fitness of the population is {np.max(fitness)} for {len(population)} individuals")

        # Train population
        for gen in range(1, self.num_generations):
            print(f"Train generation number: {gen}")
            population, fitness, ids = self.evolve_population(population, fitness, ids)
            print(
                f"The avg fitness of the population is {np.mean(fitness)}, the max fitness of the population is {np.max(fitness)} for {len(population)} individuals")
            self.report(gen, ids)

        # Select best agent
        winner_index = np.argmax(fitness)
        winner_weights = population[winner_index]

        # Save the best agent
        self.set_best_controller_data_dict({
            # Whatever data needs to be saved for the 'player'
            # This should be the same as controller_data
            # which gets passed to the .control() function
            'player': winner_weights
        })

    def single_evaluator(self, simulate):
        """
        This method evaluate the genomes running simulation
        Args:
            simulate: Function to call for running the simulation

        Returns:
            Results of the evaluation
        """

        def evaluate_this(weights_array):
            controller_data_dict_dict = {}
            for index, weights in enumerate(weights_array):
                controller_data_dict_dict[index] = {
                    'player': weights
                }

            results = self.run_simulate(simulate, controller_data_dict_dict)

            num_genomes = np.array(weights_array).shape[0]
            fitness = np.zeros(num_genomes)
            ids = np.zeros(num_genomes, dtype=int)
            for index in range(num_genomes):
                genome_results = results[index]
                fitness[index] = get_effective_fitness_of_individual_restuls(genome_results)
                individual_id = self.__get_next_id()
                ids[index] = individual_id
                self.result_cache[individual_id] = genome_results

            return fitness, ids

        return evaluate_this

    def __get_next_id(self):
        """
        Protected method to crate new ids
        Returns:
            next id
        """
        temp = self.__id_counter
        self.__id_counter = int(self.__id_counter + 1)
        return temp

    def report(self, gen, ids):
        """
        Method to collect the results after each generation
        Args:
            gen: id of the generation
            ids: ids of the genes

        Returns:
            None
        """
        if self.data_collector is not None:
            gen_results = {id: self.result_cache[id] for id in ids}
            self.data_collector.collect(gen, gen_results)

    def evolve_population(self, population, fitness, ids):
        """
        Override this method to apply:
        - crossover
        - mutation
        - selection
        and return the new population.

        Args:
            population: population array
            fitness:  fitness array
            ids:  ids array

        Returns:
            population, fitness, ids
        """

        return population, fitness, ids

    class _Individual(object):
        def __init__(self, attr_float, fitness, id):
            self.attr_float = attr_float
            self.fitness = fitness
            self.id = id


if __name__ == "__main__":
    run_mode = "test"
    base_dir = "data/evoman_generalist"
    result_filename = base_dir + '/result'

    if len(sys.argv) > 1:
        run_mode = sys.argv[1]

    simulate_normal = make_simulate({'speed': 'normal'})
    simulate_train = make_simulate({'hidden': True, 'parallel': True})

    if run_mode == "train":
        ea = EvomanGeneralist(population_size=20, num_generations=10, statistics_dir=base_dir + '/stats')
        ea.train(simulate_train)
        ea.save(result_filename)

    if run_mode == "test":
        ea = EvomanGeneralist()
        ea.load(result_filename)
        ea.run(simulate_normal)
