"""
    This module contains the helper methods for the selection
    algorithms.

    Warning
    -------
        Do not use these methods directly as they are of no use
        without their parent method directly.

    | Authors: Nick Jesperson, Yaksh J. Haranwala
"""
import numpy as np
import pandas as pd


class SelectionHelpers:
    @staticmethod
    def include_or_not(full_df_stats: pd.DataFrame, single_traj_stats: pd.DataFrame, tolerance: float):
        """
            Determine whether a trajectory is a representative trajectory or not.

            Parameters
            ----------
                full_df_stats: pd.DataFrame
                    The dataframe containing the stats for the entire dataset given.
                single_traj_stats: pd.DataFrame
                    The dataframe containing the stats for a single trajectory.
                tolerance: float
                    The tolerance that is passed into Numpy isClose() method to control
                    what trajectories are considered as close. Note that this is passed
                    as absolute tolerance and relative tolerance is always set to 0.
                    See numpy isClose() documentation for more info.

        """
        # Check if the dataframes have the same shape.
        if full_df_stats.shape != single_traj_stats.shape:
            raise ValueError("The dataframes do not have the same shape.")

        # Find the element-wise closeness between the two dataframes.
        closeness = np.isclose(full_df_stats, single_traj_stats, rtol=0, atol=tolerance)

        # Calculate the percentage of elements that are close.
        close_percentage = (closeness.sum() / (closeness.shape[0] * closeness.shape[1]))

        return close_percentage
