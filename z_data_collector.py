import pandas as pd
from z_evoman_simulation import get_effective_fitness_of_individual_restuls


class RunDataCollector(object):
    """
    This class collects data during the training of the generalist agent
    """

    def __init__(self):
        self.results = {}

    def collect(self, gen, gen_results):
        """
        The method adds the results of the generation to the global results
        Args:
            gen: id of the generation
            gen_results: generation results

        Returns:
            None
        """
        self.results[gen] = gen_results

    def get_results(self):
        """
        The method return the global result
        Returns:
            results
        """
        return self.results


class MultienemyDataCollector(object):
    """
    This class is specific for recording the result of a multi-enemy agent
    """

    def __init__(self):
        # values for the run
        self.run_list = []
        self.method_list = []
        self.enemy_group_list = []

        # values for the generation
        self.gen_list = []

        # values for the individual
        self.individual_id_list = []
        self.enemy_list = []
        self.fitness_list = []
        self.player_life_list = []
        self.enemy_life_list = []
        self.time_list = []
        self.avg_fitness_list = []

    def collect(self, run, method, enemy_group, run_results):
        """
        Collect the data
        Args:
            run: id of the run
            method: id of the method
            enemy_group: enemy group
            run_results: results to collect

        Returns:
            None
        """
        for gen, gen_results in run_results.items():
            for individual_id, individual_results in gen_results.items():

                avg_fitness = get_effective_fitness_of_individual_restuls(individual_results)

                if isinstance(individual_results, dict):
                    for enemy, result in individual_results.items():
                        fitness, player_life, enemy_life, time = result
                        self.__record(
                            run=run,
                            method=method,
                            enemy_group=enemy_group,
                            gen=gen,
                            individual_id=individual_id,
                            enemy=enemy,
                            fitness=fitness,
                            player_life=player_life,
                            enemy_life=enemy_life,
                            time=time,
                            avg_fitness=avg_fitness,
                        )
                else:
                    fitness, player_life, enemy_life, time = individual_results
                    enemy = int(enemy_group)
                    self.__record(
                        run=run,
                        method=method,
                        enemy_group=enemy_group,
                        gen=gen,
                        individual_id=individual_id,
                        enemy=enemy,
                        fitness=fitness,
                        player_life=player_life,
                        enemy_life=enemy_life,
                        time=time,
                        avg_fitness=avg_fitness,
                    )

    def __record(
            self,
            run,
            method,
            enemy_group,
            gen,
            individual_id,
            avg_fitness,
            enemy,
            fitness,
            player_life,
            enemy_life,
            time
    ):
        # values for the run
        self.run_list.append(run)
        self.method_list.append(method)
        self.enemy_group_list.append(enemy_group)

        # values for the run
        self.gen_list.append(gen)

        # values for the individual
        self.individual_id_list.append(individual_id)
        self.enemy_list.append(enemy)
        self.fitness_list.append(fitness)
        self.player_life_list.append(player_life)
        self.enemy_life_list.append(enemy_life)
        self.time_list.append(time)
        self.avg_fitness_list.append(avg_fitness)

    def get_data_frame(self):
        """
        Return the dataframe using the data collected
        Returns:
            Dataframe with data
        """
        return pd.DataFrame({
            'method': self.method_list,  # Method used
            'run': self.run_list,  # Run number
            'enemy_group': self.enemy_group_list,  # enemy group name

            'gen': self.gen_list,  # Number of the Generation

            'individual_id': self.individual_id_list,  # Id of the individual
            'enemy': self.enemy_list,
            'fitness': self.fitness_list,  # Fitness obtained from the individual,
            'player_life': self.player_life_list,  # Player life
            'enemy_life': self.enemy_life_list,  # enemy life
            'time': self.time_list,
            'avg_fitness': self.avg_fitness_list,
        })

    def save(self, filename):
        """
        Save the dataframe locally
        Args:
            filename: name of the file

        Returns:
            None
        """
        df = self.get_data_frame()
        df.to_csv(filename, index=False)
