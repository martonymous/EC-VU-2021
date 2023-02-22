import os
import subprocess
import sys
import time
import pandas as pd
import numpy as np

import z_visualization_generalist as visualization

sys.path.insert(0, 'evoman')
from z_evoman_generalist_1 import EvomanGeneralist1
from z_evoman_generalist_2 import EvomanGeneralist2
from z_evoman_simulation import make_simulate

from z_data_collector import RunDataCollector, MultienemyDataCollector


def train_generalist(enemy_group, EvomanEAMethod, config):
    """
    Function to train the generalist agent
    Args:
        enemy_group: Enemy group to use
        EvomanEAMethod: Evoman to use (GA, DE)
        config: Configuration to use in the training phase

    Returns:
        Results of the training
    """
    data_collector = RunDataCollector()
    ea = EvomanEAMethod(
        num_generations=config["num_generations"],
        population_size=config['population_size'],
        data_collector=data_collector,
    )
    ea.train(
        make_simulate({"enemies": enemy_group, "hidden": True, 'parallel': True, 'randomini': config['randomini']}))
    ea.save(config['output_filename'])
    return data_collector.get_results()


def test_generalist(enemy_group, EvomanEAMethod, config):
    """
    Function to test a generalist agent
    Args:
        enemy_group: Enemy group to use
        EvomanEAMethod: Evoman to use (GA, DE)
        config: Configuration to use in the training phase

    Returns:
        Results of the test
    """
    ea = EvomanEAMethod(num_generations=config["num_generations"], population_size=config['population_size'])
    ea.load(config['output_filename'])
    result = ea.run(
        make_simulate({"enemies": enemy_group, "hidden": True, 'parallel': True, 'randomini': config['randomini']}))
    data_collector = RunDataCollector()
    data_collector.collect(
        0,  # gen = 0
        {
            '1':  # Only one individual. Set id = 1
                result
        }
    )
    return data_collector.get_results()


def train(**params):
    """
    Function to run the training phase
    Args:
        **params: parameters need to run the phase

    Returns:
        None
    """
    methods_dict = params['methods_dict']
    enemy_groups = params['enemy_groups']
    base_dir = params['base_dir']
    config = params['config']
    n_train_runs = params['n_train_runs']
    train_runs_start_at = params['train_runs_start_at']

    start_time = time.time()
    for method_name, method in methods_dict.items():
        for run in range(train_runs_start_at, n_train_runs):
            for enemy_group in enemy_groups:
                this_start = time.time()
                enemy_group_name = '{}'.format('_'.join(str(x) for x in sorted(enemy_group)))
                print('##########\n### Method {} - Run {} - Enemy Group {}\n##########'.format(method_name, run,
                                                                                               enemy_group_name))

                run_dir = '{}/{}_run_{}'.format(base_dir, method_name, run)

                results_filename = '{}/train_results_enemy_{}.csv'.format(run_dir, enemy_group_name)
                output_filename = '{}/best_enemy_{}'.format(run_dir, enemy_group_name)

                run_results = train_generalist(enemy_group, method, {
                    **config,
                    'output_filename': output_filename,
                })
                data_collector = MultienemyDataCollector()
                data_collector.collect(run, method_name, enemy_group_name, run_results)
                data_collector.save(results_filename)

                print("This one took: {} seconds".format(time.time() - this_start))
                print("Total time: {} seconds".format(time.time() - start_time))


def test(**params):
    """
    Function to run the test phase
    Args:
        **params: parameters need to run the phase

    Returns:
        None
    """
    methods_dict = params['methods_dict']
    enemy_groups = params['enemy_groups']
    base_dir = params['base_dir']
    config = params['config']
    n_train_runs = params['n_train_runs']
    n_test_runs = params['n_test_runs']

    start_time = time.time()
    for method_name, method in methods_dict.items():
        for run in range(n_train_runs):
            for enemy_group in enemy_groups:
                this_start = time.time()
                enemy_group_name = '{}'.format('_'.join(str(x) for x in sorted(enemy_group)))
                print('##########\n### Method {} - Run {} - Enemy Group {}\n##########'.format(method_name, run,
                                                                                               enemy_group_name))

                run_dir = '{}/{}_run_{}'.format(base_dir, method_name, run)
                output_filename = '{}/best_enemy_{}'.format(run_dir, enemy_group_name)

                data_collector = MultienemyDataCollector()
                for test_run in range(n_test_runs):
                    test_run_results = test_generalist(enemy_group, method, {
                        **config,
                        'output_filename': output_filename,
                    })
                    data_collector.collect(test_run, method_name, enemy_group_name, test_run_results)

                test_results_filename = '{}/test_results_enemy_{}.csv'.format(run_dir, enemy_group_name)
                data_collector.save(test_results_filename)


def test_all(**params):
    """
    Function to test the agent against all the enemies
    Args:
        **params: parameters need to run the phase

    Returns:
        None
    """
    methods_dict = params['methods_dict']
    enemy_groups = params['enemy_groups']
    base_dir = params['base_dir']
    config = params['config']
    n_train_runs = params['n_train_runs']
    n_test_runs = params['n_test_runs']

    all_enemies = [1, 2, 3, 4, 5, 6, 7, 8]

    start_time = time.time()
    for method_name, method in methods_dict.items():
        for run in range(n_train_runs):
            for enemy_group in enemy_groups:
                this_start = time.time()
                enemy_group_name = '{}'.format('_'.join(str(x) for x in sorted(enemy_group)))
                print('##########\n### Method {} - Run {} - Enemy Group {}\n##########'.format(method_name, run,
                                                                                               enemy_group_name))

                run_dir = '{}/{}_run_{}'.format(base_dir, method_name, run)
                output_filename = '{}/best_enemy_{}'.format(run_dir, enemy_group_name)

                data_collector = MultienemyDataCollector()
                for test_run in range(n_test_runs):
                    test_run_results = test_generalist(all_enemies, method, {
                        **config,
                        'output_filename': output_filename,
                    })
                    data_collector.collect(test_run, method_name, enemy_group_name, test_run_results)

                test_results_filename = '{}/test_all_enemies_results_agent_{}.csv'.format(run_dir, enemy_group_name)
                data_collector.save(test_results_filename)


def train_results(**params):
    """
    Function to generate a unique csv file containing the results from the training phase
    Args:
        **params: parameters need to run the phase

    Returns:
        None
    """
    # Read all train results files
    methods_dict = params['methods_dict']
    enemy_groups = params['enemy_groups']
    n_train_runs = params['n_train_runs']
    base_dir = params['base_dir']

    dataframes = []

    start_time = time.time()
    for method_name, method in methods_dict.items():
        for run in range(n_train_runs):
            for enemy_group in enemy_groups:
                enemy_group_name = '{}'.format('_'.join(str(x) for x in sorted(enemy_group)))
                run_dir = '{}/{}_run_{}'.format(base_dir, method_name, run)
                results_filename = '{}/train_results_enemy_{}.csv'.format(run_dir, enemy_group_name)
                df = pd.read_csv(results_filename)
                dataframes.append(df)

    # Combine them into one file
    all_train_results = pd.concat(dataframes)
    all_train_results_filename = '{}/train_results.csv'.format(base_dir)
    all_train_results.to_csv(all_train_results_filename, index=False)


def test_results(**params):
    """
    Function to generate a unique csv file containing the results from the test phase
    Args:
        **params: parameters need to run the phase

    Returns:
        None
    """
    # Read all test results files
    methods_dict = params['methods_dict']
    enemy_groups = params['enemy_groups']
    n_train_runs = params['n_train_runs']
    base_dir = params['base_dir']

    dataframes = []

    start_time = time.time()
    for method_name, method in methods_dict.items():
        for run in range(n_train_runs):
            for enemy_group in enemy_groups:
                enemy_group_name = '{}'.format('_'.join(str(x) for x in sorted(enemy_group)))
                run_dir = '{}/{}_run_{}'.format(base_dir, method_name, run)
                test_results_filename = '{}/test_results_enemy_{}.csv'.format(run_dir, enemy_group_name)
                df = pd.read_csv(test_results_filename)
                df['test_run'] = df['run']
                df['run'] = run
                dataframes.append(df)

    # Combine them into one file
    all_test_results = pd.concat(dataframes)
    all_test_results.drop('gen', axis=1, inplace=True)
    all_test_results.drop('individual_id', axis=1, inplace=True)
    cols = all_test_results.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    all_test_results = all_test_results[cols]
    all_test_results_filename = '{}/test_results.csv'.format(base_dir)
    all_test_results.to_csv(all_test_results_filename, index=False)


def test_all_results(**params):
    """
    Function to generate a unique csv file containing the results from the test all phase
    Args:
        **params: parameters need to run the phase

    Returns:
        None
    """

    # Read all test results files
    methods_dict = params['methods_dict']
    enemy_groups = params['enemy_groups']
    n_train_runs = params['n_train_runs']
    base_dir = params['base_dir']

    dataframes = []

    start_time = time.time()
    for method_name, method in methods_dict.items():
        for run in range(n_train_runs):
            for enemy_group in enemy_groups:
                enemy_group_name = '{}'.format('_'.join(str(x) for x in sorted(enemy_group)))
                run_dir = '{}/{}_run_{}'.format(base_dir, method_name, run)
                test_results_filename = '{}/test_all_enemies_results_agent_{}.csv'.format(run_dir, enemy_group_name)
                df = pd.read_csv(test_results_filename)
                df['test_run'] = df['run']
                df['run'] = run
                dataframes.append(df)

    # Combine them into one file
    all_test_results = pd.concat(dataframes)
    all_test_results.drop('gen', axis=1, inplace=True)
    all_test_results.drop('individual_id', axis=1, inplace=True)
    cols = all_test_results.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    all_test_results = all_test_results[cols]
    all_test_results_filename = '{}/test_all_enemies_results.csv'.format(base_dir)
    all_test_results.to_csv(all_test_results_filename, index=False)


def best(**params):
    """
    Function to generate a file containing the weights of the best of the best
    Args:
        **params: parameters need to run the phase

    Returns:
        None
    """
    # I believe it is best to run each individual we have against all enemies 5 times
    # and see which one has the best average performance

    # Read combined test results for all enemies
    base_dir = params['base_dir']
    all_test_results_filename = '{}/test_all_enemies_results.csv'.format(base_dir)
    all_test_results = pd.read_csv(all_test_results_filename)

    # Decide which is the best
    # We have 10 agents
    # Each agent was tested 5 times
    # Let's choose the agent with the best average performance across the 5 tests

    #avg_fitness_across_tests = all_test_results.groupby(['method', 'run', 'enemy_group'])['avg_fitness'].mean()
    #best_index = avg_fitness_across_tests.idxmax()
    #print(avg_fitness_across_tests)

    # At the end we take the one with a higher number of wins
    all_test_results['win'] = all_test_results.enemy_life.apply(lambda x: 1 if x == 0 else 0)
    best_index = all_test_results.groupby(['method', 'run', 'enemy_group']).sum()['win'].idxmax()
    print("The best is: {}".format(best_index))

    # output the best params in the right way
    method_name = best_index[0]
    run = best_index[1]
    enemy_group_name = best_index[2]
    run_dir = '{}/{}_run_{}'.format(base_dir, method_name, run)
    output_filename = '{}/best_enemy_{}'.format(run_dir, enemy_group_name)
    methods_dict = params['methods_dict']
    EAMethod = methods_dict[method_name]
    method_instance = EAMethod()
    method_instance.load(output_filename)
    controller_data_dict = method_instance.get_best_controller_data_dict()
    weights = controller_data_dict['player']
    assert (isinstance(weights, np.ndarray))
    print(weights.shape)

    best_filename = '{}/best.txt'.format(base_dir)
    np.savetxt(best_filename, weights)


def visuals(**params):
    """
    Function to generate plots needed for the report
    Args:
        **params: parameters needed

    Returns:
        None
    """
    # read the combined train results file
    # create train visuals
    visualization.plot_training_results(base_dir=params['base_dir'])

    # read the combined test results file
    # create test visuals
    visualization.plot_test_results(base_dir=params['base_dir'])


if __name__ == '__main__':
    if 'darwin' in sys.platform:
        print('Running \'caffeinate\' on MacOSX to prevent the system from sleeping')
        subprocess.Popen('caffeinate')

    run_mode = "train"
    if len(sys.argv) > 1:
        run_mode = sys.argv[1]

    base_dir = 'results/generalist_final_1'
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    enemy_groups = [
        [7, 8],
        [3, 7, 8],
    ]

    # Official number of run for the final experiment is 10
    config = {
        'population_size': 100,
        'num_generations': 100,
        'randomini': 'yes',
    }

    methods_dict = {
        'EvomanGeneralist_GA': EvomanGeneralist1,
        'EvomanGeneralist_DE': EvomanGeneralist2,
    }

    train_runs_start_at = 0
    n_train_runs = 10
    n_test_runs = 10

    params = {
        'base_dir': base_dir,
        'train_runs_start_at': train_runs_start_at,
        'n_train_runs': n_train_runs,
        'n_test_runs': n_test_runs,
        'methods_dict': methods_dict,
        'config': config,
        'enemy_groups': enemy_groups,
    }

    mode_func = {
        'train': train,
        'test': test,
        'test_all': test_all,
        'train_results': train_results,
        'test_results': test_results,
        'test_all_results': test_all_results,
        'best': best,
        'visuals': visuals,
    }
    mode_func[run_mode](**params)
