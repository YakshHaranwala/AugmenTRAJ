"""
    This module contains the functions that are used to alter the
    trajectory points and their selection circles.

    | Authors: Nicholas Jesperson, Yaksh J. Haranwala
"""
import math
import random


class Alter:
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
