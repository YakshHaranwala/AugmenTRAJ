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
    def include_or_not(full_df_stats, single_traj_stats, tolerance_multiplier):
        """
            Determine whether a trajectory is a representative trajectory or not.

            Note
            ----
                | The tolerance is calculated as follows:
                      stat_range * tolerance multiplier
                | This is done in order to define a strict range for
                  each of the feature of the trajectory.

            Note
            ----
                The tolerance_multiplier has to be between 0 and 1 and ideally
                smaller than 0.75.

            Parameters
            ----------
                full_df_stats: pd.DataFrame
                    The dataframe containing the stats for the entire dataset given.
                single_traj_stats: pd.DataFrame
                    The dataframe containing the stats for a single trajectory.
                tolerance_multiplier: float
                    The multiplier to control the number of trajectories selected for augmentation.
        """
        if 0 < tolerance_multiplier <= 1:
            flags = []
            for i in range(len(full_df_stats.columns)):
                # Calculate the tolerance.
                tolerance = (full_df_stats[full_df_stats.columns[i]]['max']
                             - full_df_stats[full_df_stats.columns[i]]['min']) * tolerance_multiplier

                # Find if the stat values of the trajectory are in the desired range.
                closeness = np.allclose(full_df_stats[full_df_stats.columns[i]],
                                        single_traj_stats[single_traj_stats.columns[i]], rtol=0, atol=tolerance)

                # Convert the closeness array to a series and get its value counts for True/False counts.
                val_count_dict = pd.Series(closeness).value_counts().to_dict()
                flags.append(max(val_count_dict, key=val_count_dict.get))

            # If more values are within the range than not, then return True, False otherwise.
            return flags.count(True) > flags.count(False)
        else:
            raise ValueError("The Tolerance multiplier has to be between 0 and 1.")
