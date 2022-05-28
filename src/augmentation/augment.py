"""
    The following module contains algorithms which are used to
    augment the trajectories which are selected for augmentation
    process.

    | Authors: Nicholas Jesperson, Yaksh J. Haranwala
"""
import random

import pandas as pd

from ptrail.core.TrajectoryDF import PTRAILDataFrame
from src.utils.alter import Alter
from typing import Union


class Augmentation:
    @staticmethod
    def augment_trajectories_with_randomly_generated_points(dataset: Union[PTRAILDataFrame, pd.DataFrame],
                                                            pradius: float, k: float, numPoints: int,
                                                            random: random.Random):
        """
            Given the trajectories that are to be augmented, augment the trajectories by
            generating points randomly based on the given pradius. Further explanation can
            be found here: <TODO: Add the paper link here.>

            Parameters
            ----------
                dataset: Union[PTRAILDataFrame, pd.DataFrame]
                    The dataset containing the trajectories to be selected.
                pradius: float
                    The radius within which the points are to be selected.
                k: float
                    The fraction of data which is to be sampled for adding noise.
                numPoints: float
                    #TODO: Explanation
                random: random.Random
                    Custom random number generator

            Returns
            -------
                pd.DataFrame
                    The dataframe containing the augmented dataframe.
        """
        noiseData = dataset.sample(frac=k, replace=False)
        # copy here to create NEW data
        newDataSet = dataset.copy()
        randPoint = random.randint(0, numPoints)
        angle = math.pi * 2 * randPoint / numPoints

        # Using lambda functions here now to alter row by row, need to do this as the lon circle function also
        # uses the latitude
        noiseData['lat'] = noiseData.apply(lambda row: _alterLatCircle(row, angle, pradius), axis=1)
        noiseData['lon'] = noiseData.apply(lambda row: _alterLonCircle(row, angle, pradius), axis=1)
        newDataSet.update(noiseData)

        newDataSet['traj_id'] = newDataSet.apply(lambda row: row.traj_id + str(randPoint), axis=1)
        newDataSet.set_index(["traj_id", "DateTime"])
        return newDataSet
