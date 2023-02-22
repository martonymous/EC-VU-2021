# imports framework
import sys

sys.path.insert(0, '../evoman')
from z_evoman_ann_neat import EvomanAnnNeat
import numpy as np
from z_evoman_simulation import simulate, make_simulate


class EvomanAnnNeatStatic(EvomanAnnNeat):
    """
    This class extend EvomanAnnNeat and change the fitness function in order to make it static based on the report formula
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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
                results_copy[genome_id] = results_copy[genome_id] + (genome.fitness, -0.4,)

            self.results_history.append(results_copy)

        return evaluate_this

    def calc_genome_fitness(self, base_fitness, player_life, enemy_life, time):
        """
        This method calculate the genome fitness
        Args:
            base_fitness: default fitness given by the Evoman repo
            player_life: life of the player
            enemy_life: life of the enemy
            time: time of the episode

        Returns:
            fitness value
        """
        k = 0.1  # 0.5 + -0.4
        return k * player_life + (1 - k) * (100 - enemy_life) + -np.log(time)


if __name__ == "__main__":
    run_mode = "test"
    base_dir = "data/evoman_ann_neat_phases"
    result_filename = base_dir + '/result'

    if len(sys.argv) > 1:
        run_mode = sys.argv[1]

    simulate_normal = make_simulate({'speed': 'normal'})
    # simulate_train = make_simulate({"enemies": [enemy], "hidden": True, 'parallel': True})

    if run_mode == "train":
        ea = EvomanAnnNeatStatic(num_generations=10, statistics_dir=base_dir + '/stats')
        ea.train(simulate)
        ea.save(result_filename)

    if run_mode == "test":
        ea = EvomanAnnNeatStatic()
        ea.load(result_filename)
        ea.run(simulate_normal)
