"""
    This module contains the functions that are used to alter the
    trajectory points and their selection circles.

    | Authors: Nicholas Jesperson, Yaksh J. Haranwala
"""
from math import sqrt, pi, cos, sin
from random import *
import math

# Create a random generator.
random = Random


class Alter:
    @staticmethod
    def alter_latitude_randomly(row):
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
        
        sign = randint(1, 2)

        if math.isnan(row.Distance):
            dist = 1000
        else:
            dist = row.Distance
        # Generate the angle value.
        # Distance in meters roughly to degrees lat/lon and use 10% of that as the radius we can move
        r = dist * 0.00001 * .1 * sqrt(uniform(0,1))
        theta = uniform(0,1) * 2 * pi

        # Based on the random number generated above, either subtract
        # or add the theta value and return the point.
        if sign == 1:
            return row.lat + r * cos(theta)
        else:
            return row.lat - r * cos(theta)

    @staticmethod
    def alter_longitude_randomly(row):
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
        sign = randint(1, 2)
        
        if math.isnan(row.Distance):
            dist = 1000
        else:
            dist = row.Distance

        # Generate the angle value.
        r = dist * 0.00001 * .1 * sqrt(uniform(0,1))
        theta = uniform(0,1) * 2 * pi

        # Based on the random number generated above, either subtract
        # or add the theta value and return the point.
        if sign == 1:
            return row.lon + r * cos(theta)
        else:
            return row.lon - r * cos(theta)

    @staticmethod
    def alter_latitude_circle_randomly(row, angle):
        """
            Alter the latitude circle.

            Parameters
            ----------
                # TODO: Nick, can you please finish up the docs here?

            Returns
            -------
                float:

        """
        if math.isnan(row.Distance):
            dist = 1000
        else:
            dist = row.Distance
        return row.lat + (180 / math.pi) * (dist * 0.00001 * .1 * math.sin(angle))

    @staticmethod
    def alter_longitude_circle_randomly(row, angle):
        """
            Alter the longitude circle.

            Parameters
            ----------
                # TODO: Nick, can you please finish up the docs here?

             Returns
            -------
                float:
        """
        if math.isnan(row.Distance):
            dist = 1000
        else:
            dist = row.Distance
        try:
            return row.lon + (180 / math.pi) * (dist * 0.00001 * .1 * math.cos(angle)) / math.cos(row.lat * math.pi / 180)
        except ZeroDivisionError:
            print("Latitude division is yielding zero.")

    @staticmethod
    def alter_traj_randomly(traj, index):
        """
            # TODO: Nick, can you please finish up the docs here?
        """
        return traj + str(index)
