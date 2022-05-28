"""
    The following module contains algorithms which are used to
    select trajectories that will be augmented.

    | Authors: Nicholas Jesperson, Yaksh J. Haranwala
"""
import pandas as pd

from random import *
from ptrail.core.TrajectoryDF import PTRAILDataFrame
from typing import Union


class Selection:
    @staticmethod
    def select_randomly(dataset: Union[PTRAILDataFrame, pd.DataFrame], customRandom: random.Random,
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
        for i in range(math.floor(len(unique_values) * test_split_per)):
            testValues.append(unique_values_copy.pop(customRandom.randrange(len(unique_values_copy))))

        # Return the dictionary containing the train test split.
        return {"test": testValues, "train": unique_values_copy}

    @staticmethod
    def new_selection_algorithm():
        pass
