# imports framework
import os
import sys
import time

from neat import StatisticsReporter

sys.path.insert(0, '../evoman')
from controller import Controller
from z_evoman_ann_neat import EvomanAnnNeat
import neat
import numpy as np
from z_evoman_simulation import simulate, make_simulate


class EvomanAnnNeatDynamic(EvomanAnnNeat):
    """
    This class extend EvomanAnnNeat and change the fitness function in order to make it dynamic
    """

    def __init__(self, num_swings=10, **kwargs):
        super().__init__(**kwargs)
        full_range = 0.8
        self.step_size = min(0.1, full_range / (self.num_generations / num_swings))
        self.playerLifeImportance = 0
        self.decrement_player_life_importance()

    # Override
    def single_evaluator(self, key, simulate):
        def evaluate_this(genomes, config):
            controller_data_dict_dict = {}
            for genome_id, genome in genomes:
                controller_data_dict_dict[genome_id] = {
                    key: {'genome': genome, 'config': config}
                }

            results = self.run_simulate(simulate, controller_data_dict_dict)
            results_copy = results.copy()

            for genome_id, genome in genomes:
                base_fitness, player_life, enemy_life, time = results[genome_id]
                genome.fitness = self.calc_genome_fitness(base_fitness, player_life, enemy_life, time)

                # Collect data - Custom fitness value for individual and player importance values
                results_copy[genome_id] = results_copy[genome_id] + (genome.fitness, self.playerLifeImportance,)

            self.results_history.append(results_copy)
            self.check_change_x(results)

        return evaluate_this

    def calc_genome_fitness(self, base_fitness, player_life, enemy_life, time):
        """
        This method calculates the genome fitness

        f = k * player_life + (1-k)(100-enemy_life)
        k = 0.5+x
        x is the dynamic variable
        step = 0.1
        min x = -0.4
        max x = 0.4

        Args:
            base_fitness: base fitness from the default Evoman function
            player_life: life of the player
            enemy_life: life of the enemy
            time: time of the episode

        Returns:
            Fitness of the genome
        """

        k = 0.5 + self.playerLifeImportance
        return k * player_life + (1 - k) * (100 - enemy_life) + -np.log(time)

    def check_change_x(self, results):
        """
        This method change the importance of the player using winning the ratio threshold of 0.2
        Args:
            results: Results of the population

        Returns:
            None
        """
        winner_ratio = self.compute_winner_ratio(results)
        if winner_ratio > 0.2:
            self.increment_player_life_importance()
        else:
            self.decrement_player_life_importance()

    def compute_winner_ratio(self, results):
        """
        This method compute the winning ratio
        Args:
            results: Results of the population

        Returns:
            Winning ratio
        """
        num_winner_genomes = 0
        for genome_id, result in results.items():
            base_fitness, player_life, enemy_life, time = result
            if enemy_life == 0:
                num_winner_genomes += 1

        winner_ratio = num_winner_genomes / len(results)
        return winner_ratio

    def increment_player_life_importance(self):
        """
        This method increment the player life importance adding to it the step size
        Returns:
            None
        """
        playerLifeImportance = self.playerLifeImportance
        playerLifeImportance += self.step_size
        self.playerLifeImportance = min(playerLifeImportance, 0.4)

    def decrement_player_life_importance(self):
        """
        This method decrement the player life importance removing to it the step size
        Returns:
            None
        """
        playerLifeImportance = self.playerLifeImportance
        playerLifeImportance -= self.step_size
        self.playerLifeImportance = max(playerLifeImportance, -0.4)


if __name__ == "__main__":
    run_mode = "test"
    base_dir = "data/evoman_ann_neat_phases"
    result_filename = base_dir + '/result'

    if len(sys.argv) > 1:
        run_mode = sys.argv[1]

    simulate_normal = make_simulate({'speed': 'normal'})
    # simulate_train = make_simulate({"enemies": [enemy], "hidden": True, 'parallel': True})

    if run_mode == "train":
        ea = EvomanAnnNeatDynamic(num_generations=10, statistics_dir=base_dir + '/stats')
        ea.train(simulate)
        ea.save(result_filename)

    if run_mode == "test":
        ea = EvomanAnnNeatDynamic()
        ea.load(result_filename)
        ea.run(simulate_normal)
