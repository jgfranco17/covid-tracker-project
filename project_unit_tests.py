"""
TESTS.PY
File for the unittest test class for project testing
"""
import unittest
import os
from analysis import CovidData
from general import percent_diff, vowel_remove
from general import Colors as col


class Testing(unittest.TestCase):
    """
    Contains various unittests for the different class and general
    methods implemented in the project.
    """
    def test_properties(self):
        """
        Run tests for CovidData class architecture
        """
        test_object = CovidData("Test")
        self.assertEqual(test_object.country, "Test")
        self.assertEqual(test_object.db_location, "sqlite:///test_database.db")
        self.assertEqual(str(type(test_object.db)), "<class 'sqlalchemy.orm.session.Session'>")
        self.assertEqual(len(test_object), test_object.point_count)
        self.assertIsNotNone(test_object.point_count)

        os.remove(test_object.db_location.split("///")[-1])
        del test_object
        col.colprint("Test object deleted.", color='red')

    def test_percent_diff(self):
        """
        Run tests for 'general.py' file, percent_diff function
        """
        self.assertEqual(percent_diff(10, 15), float(-40))
        self.assertEqual(percent_diff(-10, -15), float(-40))
        self.assertEqual(percent_diff(10, -15), float(-1000))
        self.assertEqual(percent_diff(-10, 15), float(-1000))

    def test_vowel_remove(self):
        """
        Run tests for 'general.py' file, vowel_remove function
        """
        self.assertEqual(vowel_remove('Test'), 'Tst')
        self.assertEqual(vowel_remove('Rhythm'), 'Rhythm')
        self.assertEqual(vowel_remove('aeiou'), '')
        self.assertEqual(vowel_remove('Ab1CDe2f'), 'b1CD2f')


if __name__ == "__main__":
    unittest.main()
