import os
import subprocess
import sys
import time

import z_visualization_specialist
import z_visualization_specialist as visualization

sys.path.insert(0, '../evoman')
from z_evoman_ann_neat_static import EvomanAnnNeatStatic
from z_evoman_ann_neat_dynamic import EvomanAnnNeatDynamic
from z_evoman_simulation import make_simulate


def experiment_filename(base_dir, enemy):
    """
    Function that returns the experiment filename
    Args:
        base_dir: base directory
        enemy: enemy ID

    Returns:
        name of the experiment
    """
    return '{}/specialist_enemy{}'.format(base_dir, enemy)


def train_specialist(enemy, EvomanEAMethod, base_dir, config):
    """
    Function that train the specialist agent
    Args:
        enemy: ID of the enemy
        EvomanEAMethod: Evoman class to use in order to train the specialist
        base_dir: base directory
        config: configuration to use in order to train the specialist

    Returns:
        list containing the results for every genome in every population in every generation
    """
    ea = EvomanEAMethod(num_generations=config["num_generations"], statistics_dir=config['statistics_dir'])
    ea.train(make_simulate({"enemies": [enemy], "hidden": True, 'parallel': True}))
    ea.save(experiment_filename(base_dir, enemy))
    return ea.results_history


def test_specialist(enemy, EvomanEAMethod, base_dir, config):
    """
    Function that test the agent
    Args:
        enemy: ID of the enemy
        EvomanEAMethod: Evoman class to use in order to train the specialist
        base_dir: base directory
        config: configuration to use in order to test the specialist

    Returns:
        results of the agent fot the episode
    """
    ea = EvomanEAMethod(num_generations=config["num_generations"], statistics_dir=config['statistics_dir'])
    ea.load(experiment_filename(base_dir, enemy))
    result = ea.run(make_simulate({"enemies": [enemy]}))
    return result


if __name__ == '__main__':
    if 'darwin' in sys.platform:
        print('Running \'caffeinate\' on MacOSX to prevent the system from sleeping')
        subprocess.Popen('caffeinate')

    run_mode = "train"
    run_results = []

    base_dir = '../results/specialist'
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    if len(sys.argv) > 1:
        run_mode = sys.argv[1]

    # Would be nice perform the experiment with all these enemies
    # 6-7-8 should be very hard to fight
    enemies = [1, 2, 3, 6, 7]
    # Official number of run for the final experiment is 10
    n_runs = 10
    config = {
        'num_generations': 100,
    }

    methods_dict = {
        'EvomanAnnNeatDynamic': EvomanAnnNeatDynamic,
        'EvomanAnnNeatStatic': EvomanAnnNeatStatic,
    }

    mode_func = {
        'train': train_specialist,
        'test': test_specialist,
    }

    if run_mode == 'test':
        extra_run = 10
    else:
        extra_run = 1

    start_time = time.time()
    for name, method in methods_dict.items():
        for run in range(n_runs):
            for enemy in enemies:
                # If the script is running in test mode we have to run 5 times for each best model
                for extra in range(extra_run):
                    print('##########\n### Method {} - Enemy {} - Run {}\n##########'.format(name, enemy, run))
                    run_dir = base_dir + '/' + name + '_run_' + str(run)
                    statistics_dir = run_dir + '/enemy' + str(enemy)
                    config['statistics_dir'] = statistics_dir
                    results = mode_func[run_mode](enemy, method, run_dir, config)

                    if run_mode == 'test':
                        run_results.append({
                            'method': name,
                            'run': run,
                            'test_id': extra_run,
                            'enemy': enemy,
                            'fitness': results[0],
                            'player_life': results[1],
                            'enemy_life': results[2],
                            'time': results[3],
                        })
                    else:
                        # Create a df with the original fitness function that is used for both model
                        visualization.create_train_df_for_enemy(dir=statistics_dir,
                                                                results={
                                                                    'method': name,
                                                                    'run': run,
                                                                    'enemy': enemy,
                                                                    'results_history': results
                                                                })
                    print("Elapsed time: {} seconds".format(time.time() - start_time))

    if run_mode == 'test':
        df = z_visualization_specialist.create_test_df(base_dir=base_dir, results=run_results)
        z_visualization_specialist.plot_test_results(df=df, base_dir=base_dir, save=True)
