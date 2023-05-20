"""
    This file contains general utilities that the user can
    use as needed.

    | @author: Yaksh J Haranwala
"""
from mpmath import mp


class Utilities:
    @staticmethod
    def generate_pi_seed(num_seeds):
        """
            Generate seeds for random number generator using the digits
            of Pi after the decimal point.

            Parameters
            ----------
                num_seeds:
                    The number of seeds that are going to be required so that it
                    can be calculated upto how many digits will be needed for seeds.

            Returns
            -------
                generator:
                    Generator which will be used to fetch the seed values.

        """
        mp.dps = (num_seeds + 1) * 4
        pi = mp.pi

        pi_str = str(pi)
        pi_digits = pi_str.replace(".", "")
        for i in range(1, len(pi_digits), 4):
            yield int(pi_digits[i:i + 4])
