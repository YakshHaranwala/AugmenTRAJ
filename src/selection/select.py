"""
    The following module contains algorithms which are used to
    select trajectories that will be augmented.

    | Authors: Nicholas Jesperson, Yaksh J. Haranwala
"""
import pandas as pd

from typing import Union
from random import *
from ptrail.core.TrajectoryDF import PTRAILDataFrame
from ptrail.features.kinematic_features import KinematicFeatures
from ptrail.preprocessing.statistics import Statistics
from src.selection.helpers import SelectionHelpers

import math


class Selection:
    @staticmethod
    def select_randomly(dataset: Union[PTRAILDataFrame, pd.DataFrame], customRandom: Random,
                        test_split_per: float = .2, ):
        """
            Given the trajectories and the test splitting percentage, randomly
            select a percentage of trajectories that will be augmented.

            Parameters
            ----------
                dataset: Union[PTRAILDataFrame, pd.DataFrame]
                    The dataframe containing the trajectory data
                test_split_per: float
                    The percentage of data that should be split as the testing dataset.
                customRandom: random.Random
                    Custom random number generator

            Returns
            -------
                dict:
                    Dictionary containing the test and train partitions.
        """
        # Get all the trajectory IDs from the dataset and make a copy of it.
        unique_values = list(dataset.traj_id.unique())
        unique_values_copy = unique_values.copy()

        # Take out a percentage of trajectories to return as the testing data.
        testValues = []
        for i in range(math.floor(len(unique_values_copy) * test_split_per)):
            print(len(unique_values_copy), customRandom.randrange(len(unique_values_copy)))
            testValues.append(unique_values_copy.pop(customRandom.randrange(len(unique_values_copy))))

        # Return the dictionary containing the train test split.
        return {"test": testValues, "train": unique_values_copy}

    @staticmethod
    def select_traj_with_fewest(dataset: Union[PTRAILDataFrame, pd.DataFrame], customRandom: Random,
                                test_split_per: float = .2, ):
        """
            Given the trajectories and the test splitting percentage, randomly
            select a percentage of trajectories that will be augmented.

            Parameters
            ----------
                dataset: Union[PTRAILDataFrame, pd.DataFrame]
                    The dataframe containing the trajectory data
                customRandom: random.Random
                    Custom random number generator
                test_split_per: float
                    The percentage of data that should be split as the testing dataset.

            Returns
            -------
                dict:
                    Dictionary containing the test and train partitions.
        """
        # Get all the trajectory IDs from the dataset and make a copy of it.
        unique_values = list(dataset.traj_id.unique())

        # Must get a sorted list of trajectories and the number of points 

        trajList = []
        for i in range(len(unique_values)):
            trajSize = (dataset.traj_id == unique_values[i]).sum()
            trajList.append([unique_values[i], trajSize])

        trajList.sort(key=lambda x: x[1])

        # Insert first X values into test value, the remaining are training values. 
        # Take out a percentage of trajectories to return as the testing data.
        testValueTraj = trajList[:math.floor(len(trajList) * test_split_per)]
        trainValueTraj = trajList[math.floor(len(trajList) * test_split_per):]

        testValue = [element[0] for element in testValueTraj]
        trainValue = [element[0] for element in trainValueTraj]
        # Return the dictionary containing the train test split.
        return {"test": testValue, "train": trainValue}

    @staticmethod
    def select_representative_trajectories(dataset: Union[PTRAILDataFrame, pd.DataFrame], target_col: str,
                                           tolerance=0.5, test_split_per=0.2):
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
                test_split_per: float
                    The percentage of data that should be split as the testing dataset.

             Returns
             -------
                dict:
                    Dictionary containing the test and train partitions.
        """
        kinematic_stats = Statistics.generate_kinematic_stats(dataset, target_col)

        df_features = KinematicFeatures.generate_kinematic_features(dataset)
        full_df_stats = df_features[['Distance', 'Distance_from_start', 'Speed', 'Acceleration', 'Jerk',
                            'Bearing', 'Bearing_Rate', 'Rate_of_bearing_rate']].describe(
            percentiles=[.1, .25, .5, .75, .9]).iloc[1:, :]

        traj_ids = df_features.reset_index()['traj_id'].unique().tolist()

        include = []
        selected_trajs = []
        for i in range(len(traj_ids)):
            # Get the required stats for a single trajectory and adjust
            # its dataframe to resemble the stats df of the entire df.
            single_traj = kinematic_stats.reset_index().loc[
                kinematic_stats.reset_index()['traj_id'] == traj_ids[i]
                ].drop(columns=['traj_id', 'Species']).set_index('Columns').transpose()

            include.append(SelectionHelpers.include_or_not(full_df_stats, single_traj, 0.52))
            if SelectionHelpers.include_or_not(full_df_stats, single_traj, 0.52):
                selected_trajs.append(dataset.reset_index().loc[dataset.reset_index()['traj_id'] == traj_ids[i]])

        return PTRAILDataFrame(data_set=pd.concat(selected_trajs), traj_id='traj_id',
                               latitude='lat', longitude='lon', datetime='DateTime')
