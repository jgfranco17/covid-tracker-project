import pytest
import os
from analysis import CovidData
from general import timestamp, Colors, vowel_remove
from mathfunctions import percent_diff


class TestCustom:
    """
    Contains various unittests for the different class and general
    methods implemented in the project. Utilizes the PyTest testing
    tool to build testing structure.
    """

    @staticmethod
    def setup_class():
        print(f"[{timestamp()}]")
        print("Initializing PyTest testing...\n")

    @staticmethod
    def teardown_class():
        print(Colors.green("\n All PyTest test cases completed!"))

    @staticmethod
    def setup_method():
        current_test = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
        print(Colors.yellow("\nNow testing:"), current_test[5:])

    @staticmethod
    def teardown_method():
        print(f"\n{'-' * 30}")

    @staticmethod
    def test_tracker_properties():
        """
        Tests the CovidData class and default properties
        """
        # Object property testing
        tracker_test = CovidData('Test')
        assert tracker_test.country == "Test"
        assert tracker_test.db_location == "sqlite:///test_database.db"
        assert os.path.exists(tracker_test.db_location.split("///")[-1])
        assert str(type(tracker_test.db)) == "<class 'sqlalchemy.orm.session.Session'>"

        # Post-test cleanup
        os.remove(tracker_test.db_location.split("///")[-1])
        del tracker_test

    @staticmethod
    def test_percent_diff():
        assert percent_diff(10, 15) == float(-40)
        assert percent_diff(-10, -15) == float(-40)
        assert percent_diff(10, -15) == float(-1000)
        assert percent_diff(-10, 15) == float(-1000)

    @staticmethod
    def test_vowel_remove():
        assert vowel_remove('Test') == 'Tst'
        assert vowel_remove('Rhythm') == 'Rhythm'
        assert vowel_remove('aeiou') == ''
        assert vowel_remove('Ab1CDe2f') == 'b1CD2f'
