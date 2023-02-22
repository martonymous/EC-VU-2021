# imports framework
import os
import sys

from neat import StatisticsReporter

sys.path.insert(0, '../evoman')
from controller import Controller
from z_evoman_simple import EvomanSimple
import neat
from z_evoman_simulation import simulate, make_simulate


class EvomanRnnNeat(EvomanSimple):
    """
    Class that implement the NEAT algorithm using an RNN architecture.
    """

    def __init__(self, train_controllers=['player'], num_generations=100, statistics_dir='./stats/'):
        super().__init__(train_controllers=train_controllers)
        self.num_generations = num_generations
        self.statistics_dir = statistics_dir

    # Override
    def controllers(self):
        return {
            'player': player_controller(),
            'enemy': enemy_controller(),
        }

    # Override
    def do_train(self, simulate):
        config = {}
        population = {}

        if "player" in self.train_controllers:
            config_file = "../z_configuration_files/z_evoman_rnn_neat_player.cfg"
            config['player'] = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                           neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                           config_file)
            population['player'] = neat.Population(config['player'])

        if "enemy" in self.train_controllers:
            config_file = "../z_configuration_files/z_evoman_rnn_neat_enemy.cfg"
            config['enemy'] = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                          neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                          config_file)
            population['enemy'] = neat.Population(config['enemy'])

        # Creation of the directory for the checkpoints
        # checkpoints_dir = 'z_neat_checkpoints'
        # if not os.path.exists(checkpoints_dir):
        # 	os.makedirs(checkpoints_dir)

        for k, p in population.items():
            p.add_reporter(neat.StdOutReporter(True))
            stats = neat.StatisticsReporter()
            p.add_reporter(stats)
        # p.add_reporter(neat.Checkpointer(generation_interval=5, filename_prefix=checkpoints_dir  + '/' + k + '-neat-checkpoint-'))

        if len(population) == 1:
            for k, p in population.items():
                thisKey = k
                thisPopulation = p
                break

            winner_genome = thisPopulation.run(self.single_evaluator(thisKey, simulate), self.num_generations)

            # Save best_controller_data_dict
            self.set_best_controller_data_dict({
                thisKey: {'genome': winner_genome, 'config': config[thisKey]}
            })

            # Save all the statistics from the the population
            for reporter in thisPopulation.reporters.reporters:
                if isinstance(reporter, StatisticsReporter):
                    if not os.path.exists(self.statistics_dir):
                        os.makedirs(self.statistics_dir)
                    reporter.save_genome_fitness(delimiter=';', filename=self.statistics_dir + '/fitness_history.csv')
                    reporter.save_species_fitness(delimiter=';', filename=self.statistics_dir + '/speciation.csv')
                    reporter.save_species_count(delimiter=';', filename=self.statistics_dir + '/species_fitness.csv')

        else:
            self.__coevolve(population)

    def __coevolve(self, population):
        """
        This method train the player and the enemy controller at the same time.
        Args:
            population:

        Returns:

        """
        raise Exception("Not implemented")

        # player_population = population['player']
        # enemy_population = population['enemy']

        # def coevaluator(key):
        # 	if key == 'player':
        # 		# Get best enemy
        # 	return self.evaluator(key)

        # def evaluate_enemy():
        # 	# Get best player
        # 	# simulate

        # for i in range(0, self.num_generations):
        # 	# TODO test Should_Evaluate
        # 	player_population.run(evaluate_player, 1)
        # 	enemy_population.run(evaluate_enemy, 1)

        # # Get best player and save it to self.player_controller_context
        # # Get best enemy and save it to self.enemy_controller_context

    def single_evaluator(self, key, simulate):
        """
        This method evaluate the genomes running simulation
        Args:
            key: Key to use fot the genome
            simulate: Function to call for running the simulation

        Returns:
            Results of the evaluation
        """
        def evaluate_this(genomes, config):
            controller_data_dict_dict = {}
            for genome_id, genome in genomes:
                controller_data_dict_dict[genome_id] = {
                    key: {'genome': genome, 'config': config}
                }

            results = self.run_simulate(simulate, controller_data_dict_dict)
            for genome_id, genome in genomes:
                genome.fitness = results[genome_id][0]

        return evaluate_this


class player_controller(Controller):

    def control(self, inputs, controller_data):
        config = controller_data['config']
        genome = controller_data['genome']

        if not hasattr(self, 'net'):
            self.net = neat.nn.RecurrentNetwork.create(genome, config)

        outputs = self.net.activate(inputs)

        left = outputs[0]
        right = outputs[1]
        jump = outputs[2]
        shoot = outputs[3]
        release = outputs[4]
        return [left, right, jump, shoot, release]


class enemy_controller(Controller):

    def control(self, inputs, controller_data):
        config = controller_data['config']
        genome = controller_data['genome']

        net = neat.nn.RecurrentNetwork.create(genome, config)
        outputs = net.activate(inputs)

        attack1 = outputs[0]
        attack2 = outputs[1]
        attack3 = outputs[2]
        attack4 = outputs[3]
        return [attack1, attack2, attack3, attack4]


if __name__ == "__main__":
    run_mode = "test"
    base_dir = "data/evoman_rnn_neat"
    result_filename = base_dir + '/result'

    if len(sys.argv) > 1:
        run_mode = sys.argv[1]

    simulate_normal = make_simulate({'speed': 'normal'})

    if run_mode == "train":
        ea = EvomanRnnNeat(num_generations=10, statistics_dir=base_dir + '/stats')
        ea.train(simulate)
        ea.save(result_filename)

    if run_mode == "test":
        ea = EvomanRnnNeat()
        ea.load(result_filename)
        ea.run(simulate_normal)
