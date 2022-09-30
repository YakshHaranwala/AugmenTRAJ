"""
    The following module contains algorithms which are used to
    augment the trajectories which are selected for augmentation
    process.

    | Authors: Nicholas Jesperson, Yaksh J. Haranwala
"""
import random
import math

import pandas as pd

from ptrail.core.TrajectoryDF import PTRAILDataFrame
from ptrail.preprocessing.interpolation import Interpolation as ip
import ptrail.utilities.constants as const
from src.utils.alter import Alter
from typing import Union


class Augmentation:
    @staticmethod
    def augment_trajectories_with_randomly_generated_points(dataset: Union[PTRAILDataFrame, pd.DataFrame],
                                                            random: random.Random, circle: str = 'on', k: float=.2):
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
                    # TODO: Explanation
                random: random.Random
                    Custom random number generator

            Returns
            -------
                pd.DataFrame
                    The dataframe containing the augmented dataframe.
        """
        noiseData = dataset.sample(frac=abs(k), replace=False)
        # copy here to create NEW data
        newDataSet = dataset.copy()
        randPoint = random.randint(0, 360)
        angle = math.pi * 2 * randPoint / numPoints

        # Using lambda functions here now to alter row by row, need to do this as the lon circle function also
        # uses the latitude
        if circle == 'on':
            noiseData['lat'] = noiseData.apply(lambda row: Alter.alter_latitude_circle_randomly(row, angle, pradius),
                                               axis=1)
            noiseData['lon'] = noiseData.apply(lambda row: Alter.alter_longitude_circle_randomly(row, angle, pradius),
                                               axis=1)
        elif circle == 'in':
            noiseData['lat'] = noiseData.apply(lambda row: Alter.alter_latitude_randomly(row, pradius),
                                               axis=1)
            noiseData['lon'] = noiseData.apply(lambda row: Alter.alter_longitude_randomly(row, pradius),
                                               axis=1)

        newDataSet.update(noiseData)
        newDataSet['traj_id'] = newDataSet.apply(lambda row: row.traj_id + str(randPoint), axis=1)
        newDataSet.set_index(["traj_id", "DateTime"])
        return newDataSet

    @staticmethod
    def augment_trajectories_with_interpolation(dataset: Union[PTRAILDataFrame, pd.DataFrame],
                                                time_jump: int, ip_type: str = 'linear', numPoints: int = 200):
        """
            Given the trajectories that are to be augmented, augment the trajectories by
            generating additional points randomly based on interpolation. 

            Parameters
            ----------
                dataset: Union[PTRAILDataFrame, pd.DataFrame]
                    The dataset containing the trajectories to be selected.
                time_jump: Input parameter for ptrails interpolation function
                ip_type: determines which kind of interpolation is going to be used. 

            Returns
            -------
                pd.DataFrame
                    The dataframe containing the augmented dataframe.
        """
        dataSetReset = dataset.reset_index()
        randPoint = random.randint(0, numPoints)
        dataSetFilt = dataSetReset.filter(["traj_id", "DateTime", "lat", "lon"])
        if len(dataSetFilt['traj_id'].unique()) > 0:
            augData = ip.interpolate_position(dataSetFilt,
                                              sampling_rate=time_jump,
                                              ip_type=ip_type)

            augData = augData.reset_index()
            augData['traj_id'] = augData.traj_id.apply(lambda traj: traj + str(randPoint))

            return augData
