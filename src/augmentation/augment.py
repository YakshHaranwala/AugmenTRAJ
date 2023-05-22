"""
    The following module contains algorithms which are used to
    augment the trajectories which are selected for augmentation
    process.

    | Authors: Nicholas Jesperson, Yaksh J. Haranwala
"""
import math
import random
from typing import Union, List, Text

import pandas as pd
import numpy as np

from src.utils.alter import Alter


class Augmentation:
    @staticmethod
    def augment_trajectories_with_randomly_generated_points(dataset: pd.DataFrame, ids_to_augment: List,
                                                            circle: Text = 'on'):
        """
            Given the trajectories that are to be augmented, augment the trajectories by
            generating points randomly based on the given pradius. Further explanation can
            be found here:

            Parameters
            ----------
                dataset: Union[PTRAILDataFrame, pd.DataFrame]
                    The dataset containing the trajectories to be selected.
                circle: str
                    The method by which shaking of points is to be done.
                ids_to_augment: float
                    The fraction of data which is to be sampled for adding noise.

            Returns
            -------
                pd.DataFrame
                    The dataframe containing the augmented data along with the original ones.
        """
        traj_to_augment = dataset.loc[dataset['traj_id'].isin(ids_to_augment)]
        randPoint = random.randint(1, 10000001)
        angle = random.random() * 360

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
        return pd.concat([dataset, traj_to_augment])

    @staticmethod
    def balance_dataset_with_augmentation(dataset: pd.DataFrame, classification_col: Text,
                                          circle: Text = 'on', target_multiplier: float = 1.1):
        """
            Given the trajectories that are to be augmented, augment the trajectories by
            generating points randomly based on the given pradius. Further explanation can
            be found here:

            Definition
            ----------
                What this method does is that it finds the number of trajectories in each
                class in the dataset, picks up the maximum number for the classes, then
                multiplies it by the `target_multiplier` value and tries to balance
                the dataset by making sure that each class has above calculated number
                of trajectories.

                Example: Consider your class has the following class splits:
                Class A: 50 trajectories
                Class B: 100 trajectories
                Class C: 75 trajectories
                Total: 225 trajectories

                Based on this, the target value for each class would be:
                max_number * target_multiplier: 100 * 1.1 = 110 trajectories.

                Therefore, the final dataset will have the following architecture:
                class A: 110 trajectories
                class B: 110 trajectories
                class C: 110 trajectories
                Total: 330 trajectories

            Parameters
            ----------
                dataset: pd.DataFrame
                    The dataset containing the trajectories to be selected.
                classification_col: str
                    The column that is used as the y value in the classification tasks.
                circle: str
                    The method by which shaking of points is to be done.
                target_multiplier: float
                    The multiplier that is used to calculate the max number of trajectories
                    that each class needs to have.

            Returns
            -------
                pd.DataFrame
                    The dataframe containing the augmented data along with the original ones.
        """
        # Get the unique classes and number of trajectories in each class.
        traj_id_and_class = dataset.groupby(classification_col)['traj_id'].unique().to_dict()

        # Get the class with the maximum number of samples as well as
        # the count itself.
        max_traj_count = max(len(lst) for lst in traj_id_and_class.values())

        # Calculate the target number.
        target_traj_count = math.ceil(max_traj_count * target_multiplier)

        final_dataset = []
        for traj_class, traj_ids in traj_id_and_class.items():
            traj_to_augment = dataset.loc[dataset['traj_id'].isin(traj_ids)]
            final_dataset.append(Augmentation._balance_single_class(dataset=traj_to_augment, circle=circle,
                                                                    target_traj_count=target_traj_count))

        return pd.concat(final_dataset)

    @staticmethod
    def _balance_single_class(dataset: pd.DataFrame, circle: Text, target_traj_count: int):
        """
            Given the dataset and a list of trajectory Ids belonging to a particular class in the
            dataset, augment the trajectories in the given list such that the final dataset has
            the trajectories equal to a given target number.

            Warning
            -------
                This is a helper method for the balance_dataset_with_augmentation(). Do not use it
                directly as it may not yield desired results.

            Parameters
            ----------
                dataset: pd.DataFrame
                    The dataframe containing the trajectory data.
                circle: Text
                    The method by which shaking of points is to be done.
                class_traj_ids: List
                    The list containing all the trajectory Ids of the class that
                    is to be balanced.
                target_traj_count: int
                    The number of trajectories of the target class that the returned
                    dataframe should have.
        """
        curr_traj_count = len(dataset['traj_id'].unique().tolist())

        while curr_traj_count < target_traj_count:
            id_to_augment = np.random.choice(dataset['traj_id'].unique(), 1).tolist()
            dataset = Augmentation.augment_trajectories_with_randomly_generated_points(dataset=dataset, circle=circle,
                                                                                       ids_to_augment=id_to_augment)
            class_traj_ids = dataset['traj_id'].unique().tolist()
            curr_traj_count = len(class_traj_ids)

        return dataset
