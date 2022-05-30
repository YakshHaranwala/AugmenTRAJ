"""
    This module contains the functions that are used to alter the
    trajectory points and their selection circles.

    | Authors: Nicholas Jesperson, Yaksh J. Haranwala
"""
from math import sqrt, pi, cos, sin
from random import *

# Create a random generator.
random = Random


class Alter:
    @staticmethod
    def alter_latitude_randomly(lat, pradius):
        """
            Given a latitude point and a radius, alter the point in order
            to create a new latitude within the circle of given radius.

            Parameters
            ----------
                lat: float
                    The latitude to alter.
                pradius: float
                    The radius within which the new latitude is supposed to be.

            Returns
            -------
                float:
                    The altered latitude value.
        """
        # Choose randomly whether to add or subtract from the point.
        sign = random.randint(1, 2)

        # Generate the angle value.
        r = pradius * sqrt(random.random())
        theta = random.random() * 2 * pi

        # Based on the random number generated above, either subtract
        # or add the theta value and return the point.
        if sign == 1:
            return lat + r * cos(theta)
        else:
            return lat - r * cos(theta)

    @staticmethod
    def alter_longitude_randomly(lon, pradius):
        """
            Given a longitude point and a radius, alter the point in order
            to create a new longitude within the circle of given radius.

            Parameters
            ----------
                lon: float
                    The longitude to alter.
                pradius: float
                    The radius within which the new longitude is supposed to be.

            Returns
            -------
                float:
                    The altered latitude value.
        """
        # Choose randomly whether to add or subtract from the point.
        sign = random.randint(1, 2)

        # Generate the angle value.
        r = pradius * sqrt(random.random())
        theta = random.random() * 2 * pi

        # Based on the random number generated above, either subtract
        # or add the theta value and return the point.
        if sign == 1:
            return lon + r * cos(theta)
        else:
            return lon - r * cos(theta)

    @staticmethod
    def alter_latitude_circle_randomly(row, angle, pradius):
        """
            Alter the latitude circle.

            Parameters
            ----------
                # TODO: Nick, can you please finish up the docs here?

            Returns
            -------
                float:

        """
        return row.lat + (180 / math.pi) * (pradius * math.sin(angle))

    @staticmethod
    def alter_longitude_circle_randomly(row, angle, pradius):
        """
            Alter the longitude circle.

            Parameters
            ----------
                # TODO: Nick, can you please finish up the docs here?

             Returns
            -------
                float:
        """
        try:
            return row.lon + (180 / math.pi) * (pradius * math.cos(angle)) / math.cos(row.lat * math.pi / 180)
        except ZeroDivisionError:
            print("Latitude division is yielding zero.")

    @staticmethod
    def alter_traj_randomly(traj, index):
        """
            # TODO: Nick, can you please finish up the docs here?
        """
        return traj + str(index)
