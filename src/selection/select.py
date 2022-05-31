"""
    The following module contains algorithms which are used to
    select trajectories that will be augmented.

    | Authors: Nicholas Jesperson, Yaksh J. Haranwala
"""
import pandas as pd

from random import *
from ptrail.core.TrajectoryDF import PTRAILDataFrame
from typing import Union

import math

class Selection:
    @staticmethod
    def select_randomly(dataset: Union[PTRAILDataFrame, pd.DataFrame], customRandom: Random,
                        test_split_per: float = .2,):
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
                        test_split_per: float = .2,):
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
        
        
        trajList.sort(key = lambda x: x[1])
        
        # Insert first X values into test value, the remaining are training values. 
        # Take out a percentage of trajectories to return as the testing data.
        testValueTraj = trajList[:math.floor(len(trajList) *  test_split_per)]
        trainValueTraj = trajList[math.floor(len(trajList) *  test_split_per):]
        
        testValue = [element[0] for element in testValueTraj]
        trainValue = [element[0] for element in trainValueTraj]
        # Return the dictionary containing the train test split.
        return {"test": testValue, "train": trainValue}
