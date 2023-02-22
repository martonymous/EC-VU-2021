import statistics

import seaborn as sns
import pandas as pd
import os
import numpy as np

from matplotlib import pyplot as plt


def create_test_df(base_dir, results):
    """
    This function create a df with the results of the tests.
    Args:
        base_dir: base directory where save the dataframe
        results: Results obtained with the tests

    Returns:
        Dataframe with the tests results
    """
    # Check if the merge df is already present
    df_results_test = base_dir + '/results_test.csv'

    if results is None:
        raise Exception('You should run the test and pass the result in order to create the df')

    method_list = []
    run_list = []
    test_id_list = []
    enemy_list = []
    fitness_list = []
    player_life_list = []
    enemy_life_list = []
    time_list = []
    gain_list = []

    for result in results:
        method_list.append(result['method'])
        run_list.append(result['run'])
        test_id_list.append(result['test_id'])
        enemy_list.append(result['enemy'])
        fitness_list.append(result['fitness'])
        player_life_list.append(result['player_life'])
        enemy_life_list.append(result['enemy_life'])
        time_list.append(result['time'])
        gain_list.append(result['player_life'] - result['enemy_life'] + 100)

    df = pd.DataFrame({
        'method': method_list,
        'run': run_list,
        'test_id': test_id_list,
        'enemy': enemy_list,
        'fitness': fitness_list,
        'player_life': player_life_list,
        'enemy_life': enemy_life_list,
        'time': time_list,
        'gain': gain_list
    })
    df.to_csv(path_or_buf=df_results_test, sep=';')

    return df


def create_train_df_for_enemy(dir, results):
    """
    This function create a dataframe with the results return from the training.
    The difference between this function and create_average_results is that here we record every individual of the population
    for every generation
    Args:
        dir: Base directory
        results: History of the results from the training

    Returns:
        Df with complete results
    """
    df_results = dir + '/results_training.csv'

    if results is None:
        raise Exception('You should run the test and pass the result in order to create the df')

    if not os.path.exists(dir):
        os.makedirs(dir)

    # Generic values for the run
    method_list = []
    run_list = []
    enemy_list = []
    gen_list = []

    # Sub-values in results
    individual_id_list = []
    fitness_list = []
    player_life_list = []
    enemy_life_list = []
    time_list = []
    custom_fitness_list = []
    player_life_importance_list = []
    gain_list = []

    for gen, values in enumerate(results['results_history']):
        for id_individual, attributes in values.items():
            method_list.append(results['method'])
            run_list.append(results['run'])
            enemy_list.append(results['enemy'])
            gen_list.append(gen)
            individual_id_list.append(id_individual)
            fitness_list.append(attributes[0])
            player_life_list.append(attributes[1])
            enemy_life_list.append(attributes[2])
            time_list.append(attributes[3])
            custom_fitness_list.append(attributes[4])
            player_life_importance_list.append(attributes[5])
            gain_list.append(attributes[1] - attributes[2] + 100)

    df = pd.DataFrame({
        'method': method_list,  # Method used (e.g. NeatAnn)
        'run': run_list,  # Number of the run (1-10)
        'enemy': enemy_list,  # Id of the enemy
        'gen': gen_list,  # Number of the Generation
        'individual_id': individual_id_list,  # Id of the individual
        'fitness': fitness_list,  # Fitness obtained from the individual,
        'custom_fitness': custom_fitness_list,  # Custom fitness function
        'player_life': player_life_list,  # Player life
        'enemy_life': enemy_life_list,  # enemy life
        'player_life_importance': player_life_importance_list,  # x of the fitness function
        'time': time_list,
        'gain': gain_list  # gain (0-100)
    })
    df.to_csv(path_or_buf=df_results, sep=';')
    return df


def plot_test_results(df, base_dir, save):
    """
    This function plot a boxplot for each enemy
    Args:
        df: Dataframe to use in order to create the boxplots
        save: If true save the plots, otherwise show them

    Returns:
        None
    """
    plots_dir = base_dir + '/plots'
    df = df.groupby(['method', 'run', 'enemy']).mean().reset_index()
    df = df.drop(['test_id'], axis=1)

    for enemy in df['enemy'].drop_duplicates().sort_values():
        plt.figure(figsize=(10, 5))
        sns.boxplot(x="method", y="gain", data=df[df['enemy'] == enemy])
        sns.swarmplot(x="method", y="gain", data=df[df['enemy'] == enemy], color=".25")
        plt.title('Test results for enemy: ' + str(enemy))
        plt.xlabel('Method used')
        plt.ylabel('Gain')

        if save:
            if not os.path.exists(plots_dir):
                os.makedirs(plots_dir)
            plt.savefig(plots_dir + '/box_plot_enemy_' + str(enemy))
        else:
            plt.show()
