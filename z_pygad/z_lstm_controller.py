import statistics

import keras
import numpy as np
import pygad.kerasga

import sys
import os
import time

from tensorflow.keras.optimizers import Adam

sys.path.insert(0, '../evoman')
from controller import Controller
from environment import Environment
from z_evoman_simple import EvomanSimple
from z_lstm_constructor import Constructor
from z_read_pygad_ini import read_config
from z_evoman_simulation import make_simulate


class EvomanPyGadLstm(EvomanSimple):

	def __init__(self,train_controllers=['player'],num_generations=100, statistics_dir=''):
		super().__init__(train_controllers=train_controllers)
		self.num_generations = num_generations

		model = keras.models.Sequential()
		model.add(keras.layers.LSTM(units=35, input_shape=(20, 1)))
		model.add(keras.layers.Dense(5))
		self.model = model

	def controllers(self):
		return {
			'player': player_controller(self.model),
			'enemy': enemy_controller(),
		}

	def do_train(self, simulate):

		num_solutions = 3
		config_file = '../z_configuration_files/z_default_pygad.ini'

		generator = pygad.kerasga.KerasGA(model=self.model, num_solutions=num_solutions)
		initial_population = generator.population_weights

		def pygad_fitness_func(solution, solution_idx):
			# The solution is the wieghts
			weights = solution

			controller_data_dict_dict = {
				solution_idx: {
					'player': {
						'weights': weights,
					}
				}
			}

			results = self.run_simulate(simulate, controller_data_dict_dict)

			# [0] is the fitness
			return results[solution_idx][0]
	
		ga_configs = read_config(config_file)
		ga = pygad.GA(
			fitness_func=pygad_fitness_func,
			initial_population=initial_population,
			num_generations=self.num_generations,
			**ga_configs,
			# on_generation=keep_track_generation
		)
		ga.run()
		ga.plot_fitness()
				
		# This function simulates all solutions again!
		best_solution, _, _ = ga.best_solution()
		controller_data_dict = {
			'player': {
				'weights': best_solution,
			}
		}
		self.set_best_controller_data_dict(controller_data_dict)


# statistics_to_save = []

# def keep_track_generation(ga_instance):
# 	print(f'Generation {ga_instance.generations_completed} completed...')
# 	print(f'Fitness = {ga_instance.best_solution()[1]}')

# 	values_to_save = {
# 		'pop_size': ga_instance.pop_size[0],
# 		'max': max(ga_instance.solutions_fitness),
# 		'avg': statistics.mean(ga_instance.solutions_fitness)
# 	}

# 	# This variable is declare at file level because this callback does not allow to pass parameters
# 	statistics_to_save.append(values_to_save)

class player_controller(Controller):
	def __init__(self, model):
		self.model = model

	def control(self, inputs, controller_data):
		if not hasattr(self, 'net'):
			weights = controller_data['weights']
			constructor = Constructor(self.model, weights)
			self.net = constructor.output_model

		# This method takes 30ms which is a lot

		# The following normalization is bad because it is different for each example!
		inputs = (inputs - min(inputs)) / float((max(inputs) - min(inputs)))
		inputs = inputs.reshape(1, 20, 1)

		outputs = self.net.predict([inputs])[0]

		outputs = np.where(outputs > 0.5, 1, 0)

		left = outputs[0]
		right = outputs[1]
		jump = outputs[2]
		shoot = outputs[3]
		release = outputs[4]
		return [left, right, jump, shoot, release]


class enemy_controller(Controller):

	def control(self, inputs, controller_data):
		raise "Not implemented"
		
		attack1 = outputs[0]
		attack2 = outputs[1]
		attack3 = outputs[2]
		attack4 = outputs[3]
		return [attack1, attack2, attack3, attack4]


if __name__ == '__main__':
	run_mode = "test"
	base_dir = "data/evoman_pygad_lstm"
	result_filename = base_dir + '/result'

	if len(sys.argv) > 1:
		run_mode = sys.argv[1]

	simulate = make_simulate({'parallel': False})
	simulate_normal = make_simulate({'speed': 'normal'})

	if run_mode == "train":
		ea = EvomanPyGadLstm(num_generations=2, statistics_dir=base_dir + '/stats')
		ea.train(simulate)
		ea.save(result_filename)

	if run_mode == "test":
		ea = EvomanPyGadLstm()
		ea.load(result_filename)
		ea.run(simulate_normal)
		
