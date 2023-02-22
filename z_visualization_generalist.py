import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_feature_by_gen_with_std(df, feature, plots_dir, style=None):
    """
    Function to create a generic line plot with different style
    Args:
        df: dataframe to use in the line plot
        feature: feature to plot in the y axis (x is the generation axis)
        plots_dir: dir where save the plot
        style: style for seaborn line plot

    Returns:
        None
    """
    # Define the possible groups
    groups = df['enemy_group'].unique()

    fig, axes = plt.subplots(len(groups), 1, figsize=(10, 8))
    fig.suptitle('Training statistics', fontsize=22)

    if len(groups) > 1:

        for idx, enemy_group in enumerate(groups):
            sns.lineplot(ax=axes[idx],
                         data=df[df['enemy_group'] == enemy_group],
                         x="gen",
                         y=feature,
                         hue="method",
                         style=style,
                         ci='sd')

            axes[idx].set_title('Group: ' + str(enemy_group), fontsize=16)
            axes[idx].set_xlabel('Generation', fontsize=14)
            axes[idx].set_ylabel(feature, fontsize=14)

        # Remove single legend
        for ax in axes:
            ax.get_legend().remove()

    else:
        sns.lineplot(ax=axes,
                     data=df[df['enemy_group'] == groups[0]],
                     x="gen",
                     y=feature,
                     hue="method",
                     style=style,
                     ci='sd')

        axes.set_title('Group: ' + str(groups[0]), fontsize=16)
        axes.set_xlabel('Generation', fontsize=14)
        axes.set_ylabel(feature, fontsize=14)

    plt.legend(fontsize='x-large', title_fontsize='40')
    plt.tight_layout()
    plt.savefig(plots_dir + '/lineplot_' + feature)


def plot_max_avg_by_gen(df, plots_dir):
    """
    Function to plot the average against maximum lineplot
    Args:
        df: dataframe to use
        plots_dir: directory where save the plot

    Returns:
        None
    """
    # Max an avg plots
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # Aggregate individuals
    df_aggregated = df.groupby(['method', 'enemy_group', 'run', 'gen', 'individual_id']).mean().reset_index()

    # Creation of the max avg df
    df_mean = df_aggregated.groupby(['method', 'enemy_group', 'run', 'gen']).mean().reset_index().loc[:,
              ['method', 'enemy_group', 'run', 'gen', 'avg_fitness']]
    df_mean['stat'] = 'mean'
    df_max = df_aggregated.groupby(['method', 'enemy_group', 'run', 'gen']).max().reset_index().loc[:,
             ['method', 'enemy_group', 'run', 'gen', 'avg_fitness']]
    df_max['stat'] = 'max'

    df_mean_max = pd.concat([df_mean, df_max]).reset_index(drop=True)
    plot_feature_by_gen_with_std(df=df_mean_max, feature='avg_fitness', plots_dir=plots_dir, style='stat')


def plot_boxes_test(df, plots_dir):
    """
    Function to plot the boxplot with the gain values
    Args:
        df: dataframe to use
        plots_dir: directory where save the plot

    Returns:
        None
    """
    plt.figure(figsize=(10, 5))

    # Aggregate rows anc creating the mean of the gain from enemy1 to enemy9
    df_test_grouped = df.groupby(['method', 'enemy_group', 'run']).mean().reset_index()
    ax = sns.boxplot(x="enemy_group", y="gain", hue="method", data=df_test_grouped)
    ax = sns.swarmplot(x="enemy_group", y="gain", hue="method", data=df_test_grouped, color='0.25',dodge=True)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(fontsize='x-large', title_fontsize='40')
    ax.legend(handles[0:2], labels[0:2])

    plt.title('Gain for test dataset', fontsize=22)
    plt.xlabel('Enemy group', fontsize=18)
    plt.ylabel('Gain value', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig(plots_dir + '/box_plot_all_enemies')


def plot_training_results(base_dir):
    """
    Function to plot the training data
    Args:
        base_dir: base directory where the data are present

    Returns:
        None
    """
    train_results_file = 'train_results.csv'
    plots_dir = base_dir + '/plots'

    df_train_result = pd.read_csv(
        base_dir + '/' + train_results_file,
        delimiter=',',
        header=0)

    plot_max_avg_by_gen(df=df_train_result, plots_dir=plots_dir)


def plot_test_results(base_dir):
    """
    Function to plot the test data
    Args:
        base_dir: base directory where the data are present

    Returns:
        None
    """
    test_results_file = 'test_all_enemies_results.csv'
    plots_dir = base_dir + '/plots'

    df_test_result = pd.read_csv(
        base_dir + '/' + test_results_file,
        delimiter=',',
        header=0,
    )

    # Create the individual gain for each enemy
    df_test_result['gain'] = df_test_result['player_life'] - df_test_result['enemy_life']

    plot_boxes_test(df=df_test_result, plots_dir=plots_dir)
