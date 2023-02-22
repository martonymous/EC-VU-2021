# imports framework
import random
import sys

import numpy as np
from deap import tools

from z_evoman_generalist import EvomanGeneralist
from z_evoman_simulation import make_simulate


class EvomanGeneralist1(EvomanGeneralist):
    """
    Class to run a generalist agent using GA algorithm
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def uniform_mutation(self, individual_weights, lower_b=-1, upper_b=1, indpb=0.2):
        """
        Method for apply a  uniform mutation to an individual.
        Args:
            individual_weights: weights of the individual
            lower_b: lower bound for the random uniform number
            upper_b: upper bound for the random uniform number
            indpb: probably of mutation for each weight

        Returns:
            weights mutated
        """
        for idx, weight in enumerate(individual_weights[:]):
            if random.random() < indpb:
                # clamp value (-1, 1)
                individual_weights[idx] = np.random.uniform(lower_b, upper_b, 1)[0]
        return individual_weights

    # Override
    def evolve_population(self, population, fitness, ids):
        # population.shape == (population_size, num_params)
        # fitness.shape == (population_size)

        # Creation of individual to use with deap
        individuals = []
        for idx, individual in enumerate(population):
            individuals.append(self._Individual(attr_float=population[idx], fitness=fitness[idx], id=ids[idx]))

        # Select the size od the offspring and do the crossover
        offspring_size = int(population.shape[0] / 2)
        offsprings = []
        for off in range(offspring_size):
            parents = tools.selTournament(individuals, 2, 5)
            ch_one, ch_two = tools.cxOnePoint(ind1=np.copy(parents[0].attr_float), ind2=np.copy(parents[1].attr_float))
            offsprings.extend([ch_one, ch_two])

        # Mutation
        for idx, off in enumerate(offsprings):
            offsprings[idx] = self.uniform_mutation(individual_weights=off, upper_b=1, lower_b=-1, indpb=0.2)
            #offsprings[idx] = tools.mutShuffleIndexes(individual=off, indpb=0.2)[0]

        # Evaluate the offspring
        offsprint_fitness, offspring_ids = self.evaluator(offsprings)

        # Combine offspring to the population
        for idx, individual in enumerate(offsprings):
            individuals.append(self._Individual(attr_float=offsprings[idx], fitness=offsprint_fitness[idx], id=offspring_ids[idx]))

        # Elitism
        n_elite = int(population.shape[0] * 0.10)
        best_individuals = tools.selBest(individuals, n_elite)
        individuals = list(set(individuals) - set(best_individuals))

        # Delete worst
        n_worst = int(population.shape[0] * 0.10)
        worst_individuals = tools.selWorst(individuals, n_worst)
        individuals = list(set(individuals) - set(worst_individuals))

        # Selection with probabilities without replacement
        fitnesse_for_prop = np.array([x.fitness for x in individuals])
        fitnesse_for_prop = fitnesse_for_prop + np.absolute(np.min(fitnesse_for_prop))

        # Apply Sigma scaling for solve well known problems in selection based on fitness
        c, mean_fi, std_fi = 2, np.mean(fitnesse_for_prop), np.std(fitnesse_for_prop)
        scaled_sigma = np.vectorize(lambda x: max(x - (mean_fi - c * std_fi), 0))
        scaled_fitness = scaled_sigma(fitnesse_for_prop)

        # use the roulette mechanism
        probabilities = scaled_fitness / sum(scaled_fitness)
        # Create a list of ids for selecting without replacement
        id_selected = np.random.choice(a=np.array([x.id for x in individuals]),
                                       size=population.shape[0] - n_elite,
                                       replace=False,
                                       p=probabilities)

        final_pop = [x for x in individuals if x.id in id_selected]
        final_pop.extend(best_individuals)

        return np.array([x.attr_float for x in final_pop]), np.array([x.fitness for x in final_pop]), np.array([x.id for x in final_pop])


if __name__ == "__main__":
    run_mode = "test"
    base_dir = "data/evoman_generalist"
    result_filename = base_dir + '/result'

    if len(sys.argv) > 1:
        run_mode = sys.argv[1]

    simulate_normal = make_simulate({'speed': 'normal'})
    simulate_train = make_simulate({'hidden': True, 'parallel': True})

    if run_mode == "train":
        ea = EvomanGeneralist1(population_size=20, num_generations=10, statistics_dir=base_dir + '/stats')
        ea.train(simulate_train)
        ea.save(result_filename)

    if run_mode == "test":
        ea = EvomanGeneralist1()
        ea.load(result_filename)
        ea.run(simulate_normal)
