"""
    This file contains methods that are repeatedly used for
    testing the augmentation methods using various Machine learning models.

    @author: Yaksh J. Haranwala
"""
import csv
import random

import pandas as pd
from ptrail.preprocessing.statistics import Statistics

from test.TestUtils.Keys import *
from src.augmentation.augment import Augmentation
from src.selection.select import Selection


class TestUtils:
    # ------------------------- Data Preparation Utils ----------------------------- #
    @staticmethod
    def get_test_train_data(dataset, seed_val, class_col, k=0.8):
        """
            Given the seed value and a proportion, split the
            data into training and testing set and return it.

            Parameters
            ----------
            dataset: pd.DataFrame or PTRAILDataFrame
                The dataframe that is to be split into test and train.
            seed_val: int
                The seed value to use to control the randomness
                while selecting the train-test split.
            class_col: str
                The column that is used as the Y value in classification tasks.
            k: float
                The percent of data to be used as training data.

            Returns
            -------
                tuple:
                    training, testing_x and testing_y
        """
        # Get all the Trajectory Ids and set the random state.
        dataset = dataset.reset_index()
        traj_ids = list(dataset['traj_id'].unique())
        random.seed(seed_val)

        # Select the ids to be used as training set and the calculate subsequent testing ids.
        train_size = int(len(traj_ids) * k)
        train_traj_ids = random.sample(traj_ids, train_size)
        test_traj_ids = [id_ for id_ in traj_ids if id_ not in train_traj_ids]

        # Split the data into training and testing sets.
        training = dataset.loc[dataset.traj_id.isin(train_traj_ids)]
        testing = dataset.loc[dataset.traj_id.isin(test_traj_ids)]

        pivoted_test = Statistics.pivot_stats_df(Statistics.generate_kinematic_stats(testing, class_col, False),
                                                 class_col).dropna()

        return training.dropna(), pivoted_test.drop(columns=[class_col]), pivoted_test[class_col]

    @staticmethod
    def get_iterable_map(dataset, seed_val, class_col):
        """
            Given the seed value, generate the dataset based on it,
            and return a dictionary that contains the following values:
                | 1. Training dataset
                | 2. Testing X data.
                | 3. Testing Y data.
                | 4. Randomly selected trajectories to augment.
                | 5. Proportionally selected trajectories to augment.
                | 6. Fewest selected trajectories to augment.
                | 7. Representative selected trajectories to augment.
                | 8. Balanced using ON select_strategy Dataset.
                | 9. Balanced using IN select_strategy Dataset.

            Parameters
            ----------
                dataset: pd.DataFrame or PTRAILDataFrame
                    The dataframe that is to be used.
                seed_val: int
                    The seed value to use to control the randomness
                    while selecting the train-test split.
                class_col: str
                    The column that is used as the Y value in classification tasks.

            Returns
            -------
                dict:
                    Dictionary with the aforementioned values.
        """
        training, test_x, test_y = TestUtils.get_test_train_data(dataset, seed_val, class_col)

        # -------------------------- The trajectory selection strategies --------------------------------- #
        # Random selection.
        random_selected_ids = Selection.select_randomly(training, seed=seed_val, k=0.2)

        # Proportional selection.
        proportional_selected = Selection.select_trajectories_proportionally(training, classification_col=class_col,
                                                                             seed=seed_val, k=0.2)

        # Fewest selection.
        fewest_selected = Selection.select_with_fewest_points(training, k=0.2)

        # Representative Selection
        rep_selected = Selection.select_representative_trajectories(training, class_col, closeness_cutoff=0.7,
                                                                    tolerance=10)

        # Balance the dataset.
        balanced_on = Augmentation.balance_dataset_with_augmentation(training, class_col,
                                                                     balance_method='random', circle='on')
        balanced_in = Augmentation.balance_dataset_with_augmentation(training, class_col,
                                                                     balance_method='random', circle='in')
        balanced_drop = Augmentation.balance_dataset_with_augmentation(training, class_col, balance_method='drop',
                                                                       drop_probability=0.5)
        balanced_stretch = Augmentation.balance_dataset_with_augmentation(training, class_col, balance_method='stretch',
                                                                          stretch_method='min_max_random',
                                                                          lat_stretch=1000, lon_stretch=1000)

        # ------------------------ Create the iterable map to be returned ---------------------------------- #
        return {
            TRAINING: training,
            TEST_X: test_x,
            TEST_Y: test_y,
            RANDOM_SELECTED: random_selected_ids,
            PROPORTIONAL_SELECTED: proportional_selected,
            FEWEST_SELECTED: fewest_selected,
            REPRESENTATIVE_SELECTED: rep_selected,
            BALANCED_ON: balanced_on,
            BALANCED_IN: balanced_in,
            BALANCED_DROP: balanced_drop,
            BALANCED_STRETCH: balanced_stretch
        }

    @staticmethod
    def select_correct_test_train_split(iter_map, select_strategy, augment_strategy,
                                        class_col, n_augmentations):
        """
            Given the iterable map and the select_strategy, select the correct

            Parameters
            ----------
                iter_map: dict
                    The map generated containing the selection lists and the
                    required datasets.
                select_strategy: str
                    The selection strategy.
                augment_strategy: str
                    The augmentation strategy
                class_col: str
                    The column that is used as the Y value in classification tasks.
                n_augmentations: int
                    The number of times augmentation is to be performed.

            Returns
            -------
                tuple: train_x, train_y
                    The train and testing dataframes.
        """
        x_train, y_train = None, None

        if augment_strategy == BASE and 'balanced' not in select_strategy:
            training = Statistics.pivot_stats_df(Statistics.generate_kinematic_stats(iter_map[TRAINING], class_col),
                                                 class_col).dropna()
            x_train = training.drop(columns=[class_col])
            y_train = training[class_col]

        elif 'balanced' not in select_strategy:
            if augment_strategy == ON or augment_strategy == IN:
                x_train, y_train = TestUtils.augment_trajectories_using_random_strategy(dataset=iter_map[TRAINING],
                                                                                        ids_to_augment=iter_map[select_strategy],
                                                                                        circle=augment_strategy,
                                                                                        class_col=class_col,
                                                                                        n_augmentations=n_augmentations)
            elif augment_strategy == DROP:
                x_train, y_train = TestUtils.augment_trajectories_by_dropping(dataset=iter_map[TRAINING],
                                                                              ids_to_augment=iter_map[select_strategy],
                                                                              class_col=class_col,
                                                                              n_augmentations=n_augmentations)

            elif augment_strategy == STRETCH:
                x_train, y_train = TestUtils.augment_trajectories_by_stretching(dataset=iter_map[TRAINING],
                                                                                ids_to_augment=iter_map[select_strategy],
                                                                                class_col=class_col)
        elif 'balanced' in select_strategy:
            training = iter_map[select_strategy]
            training = Statistics.pivot_stats_df(Statistics.generate_kinematic_stats(training, class_col),
                                                 class_col).dropna()
            x_train, y_train = training.drop(columns=[class_col]), training[class_col]

        return x_train, y_train

    # ----------------------------- Augmentation Utils ----------------------------- #
    @staticmethod
    def augment_trajectories_using_random_strategy(dataset, ids_to_augment, circle,
                                                   class_col, n_augmentations):
        """
            Given the dataset, ids to augment and the balance_method select_strategy, augment
            the data using the randomly generated point on/in balance_method select_strategy.

            Parameters
            ----------
                dataset: pd.DataFrame
                    The dataset that is to be augmented.
                ids_to_augment: list
                    The list containing trajectory Ids to be augmented.
                circle: str
                    The balance_method select_strategy to be used. Valid values are: on, in.
                class_col: str
                    The column that is used as the Y value in classification task.
                n_augmentations: int
                    The number of augmentations that are to be performed.

            Returns
            -------
                pd.Dataframe:
                    The dataframe that has original as well as augmented trajectories.
        """
        # First augmentation to prepare the data for further augmentations.
        dataset = Augmentation.augment_trajectories_with_randomly_generated_points(dataset,
                                                                                   ids_to_augment=ids_to_augment,
                                                                                   circle=circle)

        # subsequent augmentations.
        for i in range(1, n_augmentations):
            dataset = Augmentation.augment_trajectories_with_randomly_generated_points(dataset,
                                                                                       ids_to_augment=ids_to_augment,
                                                                                       circle=circle)

        # convert to segment based format and return.
        pivoted = Statistics.pivot_stats_df(
            dataframe=Statistics.generate_kinematic_stats(dataset, class_col), target_col_name=class_col).dropna()

        return pivoted.drop(columns=[class_col]), pivoted[class_col]

    @staticmethod
    def augment_trajectories_by_dropping(dataset, ids_to_augment, class_col,
                                         n_augmentations, drop_probability=0.3):
        """
            Given the dataset and ids to augment, augment the data using the random
            point dropping select_strategy.

            Parameters
            ----------
                dataset: pd.DataFrame
                    The dataset that is to be augmented.
                ids_to_augment: list
                    The list containing trajectory Ids to be augmented.
                class_col: str
                    The column that is used as the Y value in classification task.
                n_augmentations: int
                    The number of augmentations that are to be performed.
                drop_probability: float
                    The probability od dropping points: [0, 1.0]

            Returns
            -------
                pd.Dataframe:
                    The dataframe that has original as well as augmented trajectories.
        """
        # First augmentation to prepare the data for further augmentations.
        dataset = Augmentation.augment_trajectories_by_dropping_points(dataset, ids_to_augment, drop_probability)

        # subsequent augmentations.
        for i in range(1, n_augmentations):
            dataset = Augmentation.augment_trajectories_by_dropping_points(dataset, ids_to_augment, drop_probability)

        # convert to segment based format and return.
        pivoted = Statistics.pivot_stats_df(dataframe=Statistics.generate_kinematic_stats(dataset, class_col),
                                            target_col_name=class_col).dropna()
        return pivoted.drop(columns=[class_col]), pivoted[class_col]

    @staticmethod
    def augment_trajectories_by_stretching(dataset, ids_to_augment, class_col):
        """
            Given the dataset and the Ids to augment, augment the selected
            trajectories exactly 4 times, once using each stretch select_strategy.


            Parameters
            ----------
                dataset: pd.DataFrame
                    The dataset that is to be augmented.
                ids_to_augment: list
                    The list containing trajectory Ids to be augmented.
                class_col: str
                    The column that is used as the Y value in classification task.

            Returns
            -------
                pd.Dataframe:
                    The dataframe that has original as well as augmented trajectories.
        """
        stretch_methods = ['min', 'max', 'min_max_random', 'random']
        dataset = Augmentation.augment_by_stretching(dataset,
                                                     ids_to_augment=ids_to_augment,
                                                     lat_stretch=2000,
                                                     lon_stretch=2000,
                                                     stretch_method=stretch_methods[0])
        for i in range(1, 4):
            dataset = Augmentation.augment_by_stretching(dataset,
                                                         ids_to_augment=ids_to_augment,
                                                         lat_stretch=2000,
                                                         lon_stretch=2000,
                                                         stretch_method=stretch_methods[i])

        pivoted = Statistics.pivot_stats_df(dataframe=Statistics.generate_kinematic_stats(dataset, class_col),
                                            target_col_name=class_col).dropna()
        return pivoted.drop(columns=[class_col]), pivoted[class_col]

    # ------------------------------- CSV File writer --------------------------------- #
    @staticmethod
    def write_csv_file(file_path, final_results):
        """
            Write the csv file containing results.

            Parameters
            ----------
                file_path: str
                    The path where the file is to be stored.
                final_results: list
                    The list containing the contents to write to csv.

        """
        with open(file_path, mode="w", newline="") as file:
            writer = csv.writer(file)
            for item in final_results:
                writer.writerow(item.split(", "))
            print(f"File successfully written to: {file_path}")

    # ------------------------------ Results Summarizer ------------------------------- #
    @staticmethod
    def summarize_results(df: pd.DataFrame, title: str):
        # Finding values corresponding to the base strategy
        base_strategy_values = df[df['strategy'] == 'base'][['seed', 'model', 'accuracy', 'f1_score']]
        base_strategy_values = base_strategy_values.rename(
            columns={'accuracy': 'base_accuracy', 'f1_score': 'base_f1_score'})

        # Finding values corresponding to the maximum accuracy and f1_score for each seed and model
        max_accuracy_values = df.groupby(['seed', 'model'])[['accuracy', 'f1_score']].max().reset_index()
        max_accuracy_values = max_accuracy_values.rename(
            columns={'accuracy': 'max_accuracy', 'f1_score': 'max_f1_score'})

        # Combining the dataframes
        print(title)
        combined = base_strategy_values.merge(max_accuracy_values, on=['seed', 'model'], how='outer')
        combined['accuracy_delta'] = combined['max_accuracy'] - combined['base_accuracy']
        combined['f1_score_delta'] = combined['max_f1_score'] - combined['base_f1_score']
        combined = combined[['seed', 'model', 'base_accuracy', 'max_accuracy', 'accuracy_delta',
                             'base_f1_score', 'max_f1_score', 'f1_score_delta']]
        combined = combined.round(4)
        return combined
