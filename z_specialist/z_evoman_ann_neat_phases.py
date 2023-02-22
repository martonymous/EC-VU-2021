# imports framework
import os
import sys
import time

from neat import StatisticsReporter

sys.path.insert(0, '../evoman')
from controller import Controller
from z_evoman_ann_neat import EvomanAnnNeat
import neat
from z_evoman_simulation import simulate, make_simulate


class EvomanAnnNeatPhases(EvomanAnnNeat):
    """
    This class extend EvomanAnnNeat and change the fitness function in order to have different phases
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # Override
    def single_evaluator(self, key, simulate):
        # start at phase 1
        self.phase = 1
        self.gen_number = 0

        def evaluate_this(genomes, config):
            self.gen_number += 1
            controller_data_dict_dict = {}
            for genome_id, genome in genomes:
                controller_data_dict_dict[genome_id] = {
                    key: {'genome': genome, 'config': config}
                }

            results = self.run_simulate(simulate, controller_data_dict_dict)
            self.results_history.append(results)

            for genome_id, genome in genomes:
                base_fitness, player_life, enemy_life, time = results[genome_id]
                genome.fitness = self.calc_genome_fitness(self.phase, base_fitness, player_life, enemy_life, time)

            self.check_change_phase_criteria(results, self.gen_number)

        return evaluate_this

    def calc_genome_fitness(self, phase, base_fitness, player_life, enemy_life, time):
        """
        This method calculate the genome fitness
        Args:
            phase: phase of the training
            base_fitness: default fitness given by the Evoman repo
            player_life: life of the player
            enemy_life: life of the enemy
            time: time of the episode

        Returns:
            fitness value
        """
        # The following formulas are normalized to be able to use
        # 100 as the fitness_threshold in neat config
        if phase == 1:
            return 0.99 * (100 - enemy_life)
        elif phase == 2:
            return 0.5 * (player_life + 100 - enemy_life)
        elif phase == 3:
            return base_fitness

    def check_change_phase_criteria(self, results, gen_number):
        """
        This method changes the phases of the training s
        Args:
            results: results of the population from the previous gen
            gen_number: generation number

        Returns:
            None
        """
        if self.phase == 1:
            # Check if at least 10% of genomes can beat the enemy
            # or if the third of the num_generations have passed
            winner_ratio = self.compute_winner_ratio(results)
            if winner_ratio >= 0.1 or gen_number >= self.num_generations / 3:
                self.phase = 2
                print("Changing to phase 2: winner_ratio {}, gen_number {}".format(winner_ratio, gen_number))

        elif self.phase == 2:
            # Change to phase 3 if two thirds of the num_generations have passed
            if gen_number >= self.num_generations * 2 / 3:
                # Compute winner_ratio just for reporting
                winner_ratio = self.compute_winner_ratio(results)
                self.phase = 3
                print("Changing to phase 3: winner_ratio {}, gen_number {}".format(winner_ratio, gen_number))

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


if __name__ == "__main__":
    run_mode = "test"
    base_dir = "data/evoman_ann_neat_phases"
    result_filename = base_dir + '/result'

    if len(sys.argv) > 1:
        run_mode = sys.argv[1]

    simulate_normal = make_simulate({'speed': 'normal'})
    # simulate_train = make_simulate({"enemies": [enemy], "hidden": True, 'parallel': True})

    if run_mode == "train":
        ea = EvomanAnnNeatPhases(num_generations=10, statistics_dir=base_dir + '/stats')
        ea.train(simulate)
        ea.save(result_filename)

    if run_mode == "test":
        ea = EvomanAnnNeatPhases()
        ea.load(result_filename)
        ea.run(simulate_normal)
