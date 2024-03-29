"""
    The following module contains algorithms which are used to
    select trajectories that will be augmented.

    | Authors: Nicholas Jesperson, Yaksh J. Haranwala
"""
import math
from typing import Union, Text

import numpy as np
import pandas as pd
from ptrail.core.TrajectoryDF import PTRAILDataFrame
from ptrail.features.kinematic_features import KinematicFeatures
from ptrail.preprocessing.statistics import Statistics

from src.selection.helpers import SelectionHelpers


class Selection:
    @staticmethod
    def select_randomly(dataset: pd.DataFrame, seed: int, k: float = .2):
        """
            Given the trajectories and the test splitting percentage, randomly
            select a percentage of trajectories that will be augmented.

            Parameters
            ----------
                dataset: pd.DataFrame
                    The dataframe containing the trajectory data
                seed: int
                    The seed that is to be used for random number generation.
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
        if seed:
            np.random.seed(seed)
        return np.random.choice(unique_values, traj_to_select, replace=False).tolist()

    @staticmethod
    def select_trajectories_proportionally(dataset: pd.DataFrame,
                                           classification_col: Text,
                                           seed: int,
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
                seed: int
                    The seed that is to be used for random number generation.
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
            # Randomly select the above calculated number of trajectories.
            if seed:
                np.random.seed(seed)
            selected_traj_ids.extend(
                np.random.choice(traj_id_and_class[key], int(num_traj_to_select), replace=False).tolist()
            )

        return selected_traj_ids

    @staticmethod
    def select_with_fewest_points(dataset: pd.DataFrame, k: float = .2):
        """
            Given the trajectories and the test splitting percentage, return a list of trajectories that have the least
            number of points.

            Parameters
            ----------
                dataset: pd.DataFrame
                    The dataframe containing the trajectory data
                k: float
                    The percentage of data that should be split as the testing dataset.

            Returns
            -------
                dict:
                    Dictionary containing the test and train partitions.
        """
        # Get the trajectory ID and number of points in each trajectory.
        unique_traj_dict = dataset['traj_id'].value_counts(ascending=True).to_dict()

        # Calculate the number of trajectories to be selected.
        num_traj_to_select = max(math.ceil(len(unique_traj_dict.keys()) * k), 1)

        # return the above number of trajectories.
        return list(unique_traj_dict.keys())[:num_traj_to_select]

    @staticmethod
    def select_representative_trajectories(dataset: Union[PTRAILDataFrame, pd.DataFrame], target_col: Text,
                                           closeness_cutoff: float, tolerance: float = 0.5):
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
                    The tolerance that is passed into Numpy isClose() method to control
                    what trajectories are considered as close. Note that this is passed
                    as absolute tolerance and relative tolerance is always set to 0.
                    See numpy isClose() documentation for more info.
                closeness_cutoff: float
                    The percentage cutoff to accept a trajectory as a representative trajectory.

             Returns
             -------
                list:
                    The list containing trajectory Ids that are representative of the dataset.
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

            closeness = SelectionHelpers.include_or_not(full_df_stats, single_traj, tolerance)
            if closeness > closeness_cutoff:
                selected_traj.append(traj_ids[i])

        return selected_traj
