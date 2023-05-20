"""
    The following module contains algorithms which are used to
    augment the trajectories which are selected for augmentation
    process.

    | Authors: Nicholas Jesperson, Yaksh J. Haranwala
"""
import random
import math

import pandas as pd
from src.utils.alter import Alter
from typing import Union

from ptrail.features.kinematic_features import KinematicFeatures
from ptrail.core.TrajectoryDF import PTRAILDataFrame


class Augmentation:
    @staticmethod
    def augment_trajectories_with_randomly_generated_points(dataset: pd.DataFrame, ids_to_augment: list,
                                                            seed: int, circle: str = 'on'):
        """
            Given the trajectories that are to be augmented, augment the trajectories by
            generating points randomly based on the given pradius. Further explanation can
            be found here:

            Parameters
            ----------
                dataset: Union[PTRAILDataFrame, pd.DataFrame]
                    The dataset containing the trajectories to be selected.
                seed: int
                    The seed that is to be used for random number generation.
                circle: str
                    The method by which shaking of points is to be done.
                ids_to_augment: float
                    The fraction of data which is to be sampled for adding noise.

            Returns
            -------
                pd.DataFrame
                    The dataframe containing the augmented dataframe.
        """
        traj_to_augment = dataset.loc[dataset['traj_id'].isin(ids_to_augment)]
        newDataSet = dataset.copy()
        random.seed(seed)
        angle = random.random() * 360
        randPoint = random.randint(1, 10000001)

        # Using lambda functions here now to alter row by row, need to do this as the lon circle function also
        # uses the latitude
        if circle == 'on':
            traj_to_augment['lat'] = traj_to_augment.apply(lambda row:
                                                           Alter.alter_latitude_on_circle(row, angle), axis=1)
            traj_to_augment['lon'] = traj_to_augment.apply(lambda row:
                                                           Alter.alter_longitude_on_circle(row, angle), axis=1)
        elif circle == 'in':
            traj_to_augment['lat'] = traj_to_augment.apply(lambda row:
                                                           Alter.alter_latitude_in_circle(row), axis=1)
            traj_to_augment['lon'] = traj_to_augment.apply(lambda row:
                                                           Alter.alter_longitude_in_circle(row), axis=1)

        traj_to_augment['traj_id'] = traj_to_augment.apply(lambda row: row.traj_id + 'aug' + str(randPoint), axis=1)
        return pd.concat([newDataSet, traj_to_augment])

    # @staticmethod
    # def augment_trajectories_with_interpolation(dataset: Union[PTRAILDataFrame, pd.DataFrame],
    #                                             time_jump: int, ip_type: str = 'linear', numPoints: int = 200):
    #     """
    #         Given the trajectories that are to be augmented, augment the trajectories by
    #         generating additional points randomly based on interpolation.
    #
    #         Parameters
    #         ----------
    #             dataset: Union[PTRAILDataFrame, pd.DataFrame]
    #                 The dataset containing the trajectories to be selected.
    #             time_jump: Input parameter for ptrails interpolation function
    #             ip_type: determines which kind of interpolation is going to be used.
    #
    #         Returns
    #         -------
    #             pd.DataFrame
    #                 The dataframe containing the augmented dataframe.
    #     """
    #     dataSetReset = dataset.reset_index()
    #     randPoint = random.randint(0, numPoints)
    #     dataSetFilt = dataSetReset.filter(["traj_id", "DateTime", "lat", "lon"])
    #     if len(dataSetFilt['traj_id'].unique()) > 0:
    #         augData = ip.interpolate_position(dataSetFilt,
    #                                           sampling_rate=time_jump,
    #                                           ip_type=ip_type)
    #
    #         augData = augData.reset_index()
    #         augData['traj_id'] = augData.traj_id.apply(lambda traj: traj + str(randPoint))
    #
    #         return augData
