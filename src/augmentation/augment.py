"""
    The following module contains algorithms which are used to
    augment the trajectories which are selected for augmentation
    process.

    | Authors: Nicholas Jesperson, Yaksh J. Haranwala
"""
import math
import random
from typing import Union, List, Text

import numpy as np
import pandas as pd

from src.utils.alter import Alter


class Augmentation:
    @staticmethod
    def augment_trajectories_with_randomly_generated_points(dataset: pd.DataFrame, percent_to_shake: float,
                                                            ids_to_augment: List, circle: Text = 'on'):
        """
            Given the trajectories that are to be augmented, augment the trajectories by
            generating points randomly based on the given pradius. Further explanation can
            be found here:

            Parameters
            ----------
                dataset: Union[PTRAILDataFrame, pd.DataFrame]
                    The dataset containing the trajectories to be selected.
                percent_to_shake: float
                    The percentage of points to shake for each trajectory.
                circle: str
                    The method by which shaking of points is to be done.
                ids_to_augment: float
                    The trajectory Ids to be augmented.

            Returns
            -------
                pd.DataFrame
                    The dataframe containing the augmented data along with the original ones.
        """
        # Create a copy of the original dataset and create an angle for augmentation.
        # The randPoint is basically for appending to the new trajectory id to discriminate it from
        # the original and the other augmented trajectories.
        final_dataset = dataset.copy()
        randPoint = random.randint(1, 10000001)
        angle = random.random() * 360

        # Augment each of the trajectory.
        for id_ in ids_to_augment:
            small = final_dataset.loc[final_dataset['traj_id'] == id_]

            # Randomly select the points to be changed based on percent_to_shake given by the user.
            points_to_change = small.groupby('traj_id').apply(lambda x: x.sample(frac=percent_to_shake))\
                                    .index.get_level_values(1)

            # Modify the points one at a time based on the circle method given by the user.
            for i in range(len(points_to_change)):
                row = small.loc[points_to_change[i]]
                if circle == 'on':
                    small.loc[i, 'lat'] = Alter.alter_point_on_circle(row, angle, 'latitude')
                    small.loc[i, 'lon'] = Alter.alter_point_on_circle(row, angle, 'longitude')
                elif circle == 'in':
                    small.loc[i, 'lat'] = Alter.alter_point_in_circle(row, 'latitude')
                    small.loc[i, 'lon'] = Alter.alter_point_in_circle(row, 'longitude')

            # Create and use the new trajectory id for the augmented trajectory.
            small['traj_id'] = id_ + 'aug' + str(randPoint)

            # Add the augmented trajectory to the final dataset.
            final_dataset = pd.concat([final_dataset, small])

        return final_dataset

    @staticmethod
    def balance_dataset_with_augmentation(dataset: pd.DataFrame, classification_col: Text, balance_method: Text,
                                          target_multiplier: float = 1.1, **kwargs):
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

                Example: Consider your data has the following class splits:
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
                balance_method: str
                    The method of balancing the dataset to be used.
                target_multiplier: float
                    The multiplier that is used to calculate the max number of trajectories
                    that each class needs to have.
                **kwargs: Any
                    Ensure that the balance method and its relevant arguments are passed in this.

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
            aug, traj_to_augment = None, dataset.loc[dataset['traj_id'].isin(traj_ids)]
            if balance_method == 'random':
                aug = Augmentation._balance_single_class_with_randomly_generated_points(dataset=traj_to_augment,
                                                                                        circle=kwargs['circle'],
                                                                                        target_traj_count=target_traj_count)
            elif balance_method == 'stretch':
                aug = Augmentation._balance_single_class_with_stretching(dataset=traj_to_augment,
                                                                         target_traj_count=target_traj_count,
                                                                         stretch_method=kwargs['stretch_method'],
                                                                         lat_stretch=kwargs['lat_stretch'],
                                                                         lon_stretch=kwargs['lon_stretch'])
            elif balance_method == 'drop':
                aug = Augmentation._balance_single_class_with_dropping(dataset=traj_to_augment,
                                                                       target_traj_count=target_traj_count,
                                                                       drop_probability=kwargs['drop_probability'])

            final_dataset.append(aug)

        return pd.concat(final_dataset)

    @staticmethod
    def augment_trajectories_by_dropping_points(dataset: pd.DataFrame, ids_to_augment: List,
                                                drop_probability: float = 0.25):
        """
            Given the trajectories that are to be augmented, augment the trajectories by
            randomly dropping points with a given probability.

            Parameters
            ----------
                dataset: Union[PTRAILDataFrame, pd.DataFrame]
                    The dataset containing the trajectories to be selected.
                ids_to_augment: float
                    The trajectory Ids to be augmented.
                drop_probability: float
                    The probability with which points are to be dropped.

            Returns
            -------
                pd.DataFrame
                    The dataframe containing the augmented data along with the original ones.
        """
        traj_to_augment = dataset.loc[dataset['traj_id'].isin(ids_to_augment)].copy()
        randPoint = random.randint(1, 10000001)

        rows_to_drop = traj_to_augment.groupby('traj_id').apply(lambda x: x.sample(frac=drop_probability))\
                                      .index.get_level_values(1)
        traj_to_augment.drop(rows_to_drop, inplace=True)
        traj_to_augment['traj_id'] = traj_to_augment['traj_id'] + 'aug' + str(randPoint)

        return pd.concat([dataset, traj_to_augment], ignore_index=True)

    @staticmethod
    def augment_by_stretching(dataset: pd.DataFrame, ids_to_augment: List, stretch_method: Text,
                              lat_stretch: float, lon_stretch: float):
        """
            Given the trajectories that are to be augmented, augment the trajectories by
            stretching the points.

            Parameters
            ----------
                dataset: Union[PTRAILDataFrame, pd.DataFrame]
                    The dataset containing the trajectories to be selected.
                ids_to_augment: float
                    The trajectory Ids to be augmented.
                stretch_method: Text
                    The stretch method to be used when augmenting the data.
                lat_stretch: float
                    The maximum distance by which the latitude is to be stretched.
                lon_stretch: float
                    The minimum distance by which the longitude is to be stretched.

            Returns
            -------
                pd.DataFrame
                    The dataframe containing the augmented data along with the original ones.
        """
        traj_to_augment = dataset.loc[dataset['traj_id'].isin(ids_to_augment)].reset_index(drop=True)
        randPoint = random.randint(1, 10000001)

        # Using lambda functions here now to alter row by row, need to do this as the lon balance_method function also
        # uses the latitude
        for i in range(len(traj_to_augment)):
            if stretch_method == 'min':
                lat, lon = Alter.calculate_point_based_on_stretch(traj_to_augment.iloc[i], lat_stretch, lon_stretch,
                                                                  'min')
                traj_to_augment.at[i, 'lat'] = lat
                traj_to_augment.at[i, 'lon'] = lon
                traj_to_augment.at[i, 'traj_id'] = traj_to_augment.at[i, 'traj_id'] + 'aug' + stretch_method + str(
                    randPoint)

            elif stretch_method == 'max':
                lat, lon = Alter.calculate_point_based_on_stretch(traj_to_augment.iloc[i], lat_stretch, lon_stretch,
                                                                  'max')
                traj_to_augment.loc[i, 'lat'] = lat
                traj_to_augment.loc[i, 'lon'] = lon
                traj_to_augment.at[i, 'traj_id'] = traj_to_augment.at[i, 'traj_id'] + 'aug' + stretch_method + str(
                    randPoint)

            elif stretch_method == 'min_max_random':
                lat, lon = Alter.calculate_point_based_on_stretch(traj_to_augment.iloc[i], lat_stretch, lon_stretch,
                                                                  'min_max_random')
                traj_to_augment.loc[i, 'lat'] = lat
                traj_to_augment.loc[i, 'lon'] = lon
                traj_to_augment.at[i, 'traj_id'] = traj_to_augment.at[i, 'traj_id'] + 'aug' + stretch_method + str(
                    randPoint)

            else:
                lat, lon = Alter.calculate_point_based_on_stretch(traj_to_augment.iloc[i], lat_stretch, lon_stretch,
                                                                  'random')
                traj_to_augment.loc[i, 'lat'] = lat
                traj_to_augment.loc[i, 'lon'] = lon
                traj_to_augment.at[i, 'traj_id'] = traj_to_augment.at[i, 'traj_id'] + 'aug' + stretch_method + str(
                    randPoint)

        return pd.concat([dataset, traj_to_augment])

    # ----------------------------------- Helper Methods ------------------------------------- #
    @staticmethod
    def _balance_single_class_with_randomly_generated_points(dataset: pd.DataFrame, circle: Text,
                                                             target_traj_count: int):
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
                target_traj_count: int
                    The number of trajectories of the target class that the returned
                    dataframe should have.
        """
        curr_traj_count = len(dataset['traj_id'].unique().tolist())

        while curr_traj_count < target_traj_count:
            id_to_augment = np.random.choice(dataset['traj_id'].unique(), 1).tolist()
            dataset = Augmentation.augment_trajectories_with_randomly_generated_points(dataset=dataset,
                                                                                       circle=circle,
                                                                                       ids_to_augment=id_to_augment)
            class_traj_ids = dataset['traj_id'].unique().tolist()
            curr_traj_count = len(class_traj_ids)

        return dataset

    @staticmethod
    def _balance_single_class_with_stretching(dataset: pd.DataFrame, target_traj_count: int,
                                              stretch_method: Text, lat_stretch: float, lon_stretch: float):
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
                target_traj_count: int
                    The number of trajectories of the target class that the returned
                    dataframe should have.
                stretch_method: Text
                    The method of stretching to be used.
                lat_stretch: float
                    The maximum stretching to be done in the latitude direction.
                lon_stretch: float
                    THe maximum stretching to be done in the longitude direction.
        """
        curr_traj_count = len(dataset['traj_id'].unique().tolist())

        while curr_traj_count < target_traj_count:
            id_to_augment = np.random.choice(dataset['traj_id'].unique(), 1).tolist()
            dataset = Augmentation.augment_by_stretching(dataset=dataset, ids_to_augment=id_to_augment,
                                                         stretch_method=stretch_method, lat_stretch=lat_stretch,
                                                         lon_stretch=lon_stretch)
            class_traj_ids = dataset['traj_id'].unique().tolist()
            curr_traj_count = len(class_traj_ids)

        return dataset

    @staticmethod
    def _balance_single_class_with_dropping(dataset: pd.DataFrame, target_traj_count: int,
                                            drop_probability: float):
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
                target_traj_count: int
                    The number of trajectories of the target class that the returned
                    dataframe should have.
                drop_probability: float
                    The probability with which points are to be dropped.
        """
        curr_traj_count = len(dataset['traj_id'].unique().tolist())

        while curr_traj_count < target_traj_count:
            id_to_augment = np.random.choice(dataset['traj_id'].unique(), 1).tolist()
            dataset = Augmentation.augment_trajectories_by_dropping_points(dataset=dataset,
                                                                           ids_to_augment=id_to_augment,
                                                                           drop_probability=drop_probability)
            class_traj_ids = dataset['traj_id'].unique().tolist()
            curr_traj_count = len(class_traj_ids)

        return dataset
