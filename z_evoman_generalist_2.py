# imports framework
import random
import sys

import numpy as np

from z_evoman_generalist import EvomanGeneralist
from z_evoman_simulation import make_simulate


class EvomanGeneralist2(EvomanGeneralist):
    """
    Class to run a generalist agent using DE algorithm
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # Override
    def evolve_population(self, population, fitness, ids):

        # Creation of individuals to use with deap
        originals = []
        offsprings = []
        for idx, individual in enumerate(population):
            originals.append(self._Individual(attr_float=population[idx], fitness=fitness[idx], id=ids[idx]))
            offsprings.append(self._Individual(attr_float=population[idx], fitness=fitness[idx], id=ids[idx]))

        # Mutation nd Crossover
        for idx, off in enumerate(originals):
            offsprings[idx].attr_float = self.de_mutate(off, originals, 0.6, 1.0, -1.0, 0.9)
            del offsprings[idx].fitness

        # Evaluate the offsprings
        originals_fitness, originals_ids = [x.fitness for x in originals], [x.id for x in originals]
        # The evaluator evaluates only a numpy list of weights and not the class individuals
        offspring_fitness, offspring_ids = self.evaluator([x.attr_float for x in offsprings])

        # Combine offspring to the population for individuals better than original
        for idx, individual in enumerate(offsprings):
            if originals_fitness[idx] < offspring_fitness[idx]:
                originals[idx] = individual
                originals[idx].fitness = offspring_fitness[idx]
                originals[idx].id = offspring_ids[idx]

        final_pop = originals

        return np.array([x.attr_float for x in final_pop]), np.array([x.fitness for x in final_pop]), np.array([x.id for x in final_pop])

    def de_mutate(self, individual, population, F: float, upper_b, lower_b, CXpb: float):
        """
        Method for applying mutation and crossover using the differential evolution paradigm
        Args:
            individual: Individual to mutate
            population: Rest of the population
            F: constant of DE formula
            upper_b: Upper bound for random number
            lower_b:Lower bound for random number
            CXpb: Probability of crossover

        Returns:
            Individual mutated
        """
        candidates = population.copy()
        trial_individual = individual.attr_float.copy()
        candidates.remove(individual)

        # simple mutation
        ind1, ind2, ind3 = random.sample(candidates)
        mutation_vector = ind1.attr_float + (F * (ind2.attr_float + ind3.attr_float))

        # check if value is within upper and lower bound
        for i in range(len(mutation_vector)):
            if mutation_vector[i] > upper_b:
                mutation_vector[i] = upper_b
            elif mutation_vector[i] < lower_b:
                mutation_vector[i] = lower_b

        # if crossover with a target takes place (for every point separate)...
        for j in range(len(individual.attr_float)):
            if random.random() < CXpb:
                trial_individual[j] = mutation_vector[j]

        return trial_individual


if __name__ == "__main__":
    run_mode = "train"
    base_dir = "data/evoman_generalist"
    result_filename = base_dir + '/result'

    if len(sys.argv) > 1:
        run_mode = sys.argv[1]

    simulate_normal = make_simulate({'speed': 'normal'})
    simulate_train = make_simulate({'hidden': True, 'parallel': True})

    if run_mode == "train":
        ea = EvomanGeneralist2(population_size=10, num_generations=50, statistics_dir=base_dir + '/stats')
        ea.train(simulate_train)
        ea.save(result_filename)

    if run_mode == "test":
        ea = EvomanGeneralist2()
        ea.load(result_filename)
        ea.run(simulate_normal)
