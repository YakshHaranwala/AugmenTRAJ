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
    def include_or_not(full_df_stats, single_traj_stats, tolerance):
        """
            Determine whether a trajectory is a representative trajectory or not.

            Note
            ----
                | The tolerance is calculated as follows:
                      stat_range * tolerance multiplier
                | This is done in order to define a strict range for
                  each of the feature of the trajectory.

            Parameters
            ----------
                full_df_stats: pd.DataFrame
                    The dataframe containing the stats for the entire dataset given.
                single_traj_stats: pd.DataFrame
                    The dataframe containing the stats for a single trajectory.
                tolerance: float
                    The tolerance to control the number of trajectories selected for augmentation.
        """
        # TODO: Improve this method.
        # Check if the dataframes have the same shape.
        if full_df_stats.shape != single_traj_stats.shape:
            raise ValueError("The dataframes do not have the same shape.")

        # Find the element-wise closeness between the two dataframes.
        closeness = np.isclose(full_df_stats, single_traj_stats, rtol=0, atol=tolerance/2)

        # Calculate the percentage of elements that are close.
        close_percentage = (closeness.sum() / (closeness.shape[0] * closeness.shape[1]))
        print(f"Closeness: {close_percentage}")

        return close_percentage > tolerance
