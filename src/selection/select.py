"""
    The following module contains algorithms which are used to
    select trajectories that will be augmented.

    | Authors: Nicholas Jesperson, Yaksh J. Haranwala
"""
import pandas as pd
import numpy as np

from typing import Union
from ptrail.features.kinematic_features import KinematicFeatures
from ptrail.preprocessing.statistics import Statistics
from src.selection.helpers import SelectionHelpers
from src.utils.alter import Alter

import math


class Selection:
    @staticmethod
    def select_randomly(dataset: pd.DataFrame, k: float = .2):
        """
            Given the trajectories and the test splitting percentage, randomly
            select a percentage of trajectories that will be augmented.

            Parameters
            ----------
                dataset: pd.DataFrame
                    The dataframe containing the trajectory data
                k: float
                    The percentage of data that should be selected.

            Returns
            -------
                list:
                    List containing randomly selected trajectory Ids to augment.
        """
        # Get all the trajectory IDs from the dataset.
        unique_values = dataset['traj_id'].unique()

        # Get the number of Ids that we need to select.
        traj_to_select = max(math.floor(len(unique_values) * k), 1)

        # Randomly select the number of ids calculated above without replacement and return them.
        return np.random.choice(unique_values, traj_to_select, replace=False).tolist()

    @staticmethod
    def select_trajectories_proportionally(dataset: pd.DataFrame,
                                           classification_col: str,
                                           k: float = .2):
        """
            Given the trajectories and the test splitting percentage, randomly
            select a percentage of trajectories that will be augmented.

            Parameters
            ----------
                dataset: pd.DataFrame
                    The dataframe containing the trajectory data
                classification_col: str
                    The column that is used in classification tasks. Essentially the value
                    that identifies each trajectory to a specific class.
                k: float
                    The percentage of data that should be selected.

            Returns
            -------
                list:
                    List containing proportionally selected trajectory Ids to augment.
        """
        # List to store the select ids.
        selected_traj_ids = []

        # Get the number of trajectories that belong to each category.
        traj_id_and_class = dataset.groupby(classification_col)['traj_id'].unique().to_dict()

        # For each class that we got above, randomly select k% of total trajectories for
        # that class and add them to the selected list.
        for key, value in traj_id_and_class.items():
            num_traj_to_select = max(np.ceil(len(traj_id_and_class[key]) * k), 1)
            print(f"Class: {key}, Total trajectories in class: {len(traj_id_and_class[key])}, trajectories selected: {num_traj_to_select}")
            # Randomly select the above calculated number of trajectories.
            selected_traj_ids.extend(
                np.random.choice(traj_id_and_class[key], int(num_traj_to_select), replace=False).tolist()
            )

        return selected_traj_ids

    @staticmethod
    def select_fewest_class(dataset: pd.DataFrame, classify: str,
                            customRandom, test_split_per: float = .2):
        """
            Given the trajectories and the test splitting percentage, return a list of trajectories that have the least 
            represented class

            Parameters
            ----------
                dataset: pd.DataFrame
                    The dataframe containing the trajectory data
                test_split_per: float
                    The percentage of data that should be split as the testing dataset.
                classify: str
                    The header of the class column.
                customRandom: Random
                    A randomNumber generator with your preferred seed.

            Returns
            -------
                dict:
                    Dictionary containing the test and train partitions.
        """
        # TODO: Update this code to suit the framework, and hence return a list of selected ids to augment.
        # Get all the unique classification column values and set their counts in the dict to 0.
        uniqueValsDict = dict(dataset[classify].value_counts())
        for key in uniqueValsDict:
            uniqueValsDict[key] = []

        # Create a list with unique trajectory IDs and for each unique trajectory, find which
        # class it belongs to and then increase the count of that class.
        uniqueTrajIds = dataset['traj_id'].unique()
        for traj_id in uniqueTrajIds:
            key = dataset.reset_index().loc[dataset.reset_index()['traj_id'] == traj_id][classify].unique()
            uniqueValsDict[key[0]].append(traj_id)

        dictKeys = uniqueValsDict.keys()

        testValues = []
        trainValues = []
        for val in dictKeys:
            for i in range(math.floor(len(uniqueValsDict[val]) * test_split_per)):
                testValues.append(uniqueValsDict[val].pop(customRandom.randrange(len(uniqueValsDict[val]))))
            trainValues = trainValues + uniqueValsDict[val]

        # Return the dictionary containing the train test split.
        return {"test": testValues, "train": trainValues}

    @staticmethod
    def select_representative_trajectories(dataset: Union[PTRAILDataFrame, pd.DataFrame], target_col: str,
                                           tolerance=0.5):
        """
             Given a dataset, select the trajectories that are representative of
             the given dataset.

             Definition
             ----------
                Representative Trajectory:
                    - A trajectory is called a representative trajectory if its stats are
                      in a user specified range of the entire dataset's stats.
                    - We are using Kinematic stats for selecting the representative
                      trajectories from the dataset.

             Parameters
             ----------
                dataset: Union[PTRAILDataFrame, pd.DataFrame]
                    The dataframe from which the trajectories are to be selected.
                target_col: str
                    - The PTRAIL parameter of target column.
                    - It is the column which is the column header for classification result column.
                tolerance: float
                    The Tolerance multiplier for creating the range of comparison of stats.
                k: float
                    The percentage of data that should be split as the testing dataset.

             Returns
             -------
                list:
                    The list containing trajectory Ids that are representative of the dataset..
        """
        # Generate Kinematic stats for the given dataframe.
        kinematic_stats = Statistics.generate_kinematic_stats(dataset, target_col)

        # Generate kinematic features for the entire data given and take out the required
        # stats only.
        df_features = KinematicFeatures.generate_kinematic_features(dataset)
        full_df_stats = df_features[['Distance', 'Distance_from_start', 'Speed', 'Acceleration', 'Jerk',
                                     'Bearing', 'Bearing_Rate', 'Rate_of_bearing_rate']].describe(
            percentiles=[.1, .25, .5, .75, .9]).iloc[1:, :]

        # Select Trajectories to be augmented.
        traj_ids = df_features.reset_index()['traj_id'].unique().tolist()
        selected_traj = []
        for i in range(len(traj_ids)):
            # Get the required stats for a single trajectory and adjust
            # its dataframe to resemble the stats df of the entire df.
            single_traj = kinematic_stats.reset_index().loc[kinematic_stats.reset_index()['traj_id'] == traj_ids[i]]\
                                         .drop(columns=['traj_id', target_col]).set_index('Columns').transpose()

            if SelectionHelpers.include_or_not(full_df_stats, single_traj, tolerance):
                selected_traj.append(traj_ids[i] + "sel")

        return selected_traj
