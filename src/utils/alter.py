"""
    This module contains the functions that are used to alter the
    trajectory points and their selection circles.

    | Authors: Nicholas Jesperson, Yaksh J. Haranwala
"""
import math
import random
import numpy as np


class Alter:
    _EARTH_RADIUS = 6378.137

    @staticmethod
    def alter_point_in_circle(row, point_to_alter):
        """
            Given a latitude point and a radius, alter the point in order
            to create a new latitude within the circle of given radius.

            Parameters
            ----------
                row:
                    The row containing the latitude to be altered.
                point_to_alter:
                    Indicate whether to augment the latitude or longitude.

            Returns
            -------
                float:
                    The altered latitude value.
        """
        # Choose randomly whether to add or subtract from the point.

        sign = random.randint(1, 2)

        if math.isnan(row.Distance):
            dist = 1000
        else:
            dist = row.Distance
        # Generate the angle value.
        # Distance in meters roughly to degrees lat/lon and use 10% of that as the radius we can move
        r = dist * 0.00001 * .1 * math.sqrt(random.uniform(0, 1))
        theta = random.uniform(0, 1) * 2 * math.pi

        # Based on the random number generated above, either subtract
        # or add the theta value and return the point.
        if point_to_alter == 'latitude':
            if sign == 1:
                return row.lat + r * math.cos(theta)
            else:
                return row.lat - r * math.cos(theta)
        else:
            if sign == 1:
                return row.lon + r * math.cos(theta)
            else:
                return row.lon - r * math.cos(theta)

    @staticmethod
    def alter_point_on_circle(row, angle, point_to_alter):
        """
            Alter the latitude value and generate a value that is on the circumference.

            Parameters
            ----------
                row:
                    The row containing the latitude to be altered.
                angle:
                    The angle with which the point is to be changed.
                point_to_alter:
                    Indicate whether to augment the latitude or longitude.

             Returns
            -------
                float:
                    The altered latitude value.
        """
        if math.isnan(row.Distance):
            dist = 1000
        else:
            dist = row.Distance

        try:
            if point_to_alter == 'longitude':
                return row.lon + (180 / math.pi) * \
                       ((dist * 0.00001 * .1 * math.cos(angle)) / math.cos(row.lat * math.pi / 180))
            else:
                return row.lat + (180 / math.pi) * (dist * 0.00001 * .1 * math.sin(angle))
        except ZeroDivisionError:
            print("Latitude division is yielding zero.")

    @staticmethod
    def calculate_point_based_on_stretch(row, lat_stretch: float, lon_stretch: float,
                                         stretch_method: str = 'max'):
        """
            Given a latitude value and a stretch distance, randomly select
            a point between [lat-stretch_distance, lat+stretch_distance].

            Parameters
            ----------
                row:
                    The row containing the latitude, longitude values to be stretched.
                lat_stretch: float
                    The max distance that is to be used while stretching the latitude.
                lon_stretch: float
                    The max distance that is to be used while stretching the longitude.
                stretch_method: str
                    The method that is to be used to stretch the point.

            Returns
            -------
                float:
                    The stretched point value between [lat-stretch_distance, lat+stretch_distance].
        """
        # Calculate the min and the max latitude based on distance.
        lat_degree_to_distance_factor = (1 / ((2 * math.pi / 360) * Alter._EARTH_RADIUS)) / 1000
        min_new_lat = row.lat - (lat_stretch * lat_degree_to_distance_factor)
        max_new_lat = row.lat + (lat_stretch * lat_degree_to_distance_factor)

        # Calculate the min and the max longitude based on distance.
        lon_degree_to_distance_factor = (1 / ((2 * math.pi / 360) * Alter._EARTH_RADIUS)) / 1000
        min_new_lon = row.lon - (lon_stretch * lon_degree_to_distance_factor) / math.cos(row.lat * (math.pi / 180))
        max_new_lon = row.lon + (lon_stretch * lon_degree_to_distance_factor) / math.cos(row.lat * (math.pi / 180))

        # Get a random point by stretching.
        if stretch_method == 'min':
            return min_new_lat, min_new_lon

        if stretch_method == 'max':
            return max_new_lat, max_new_lon

        if stretch_method == 'min_max_random':
            if random.randint(0, 1) == 0:
                return min_new_lat, min_new_lon
            else:
                return max_new_lat, max_new_lon

        if stretch_method == 'random':
            return random.uniform(min_new_lat, max_new_lat), random.uniform(min_new_lon, max_new_lon)

