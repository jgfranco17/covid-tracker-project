"""
GENERAL.PY

Hosts miscellaneous functions related to general project management
and system organization.
"""
import datetime
import time
from dataclasses import dataclass

@dataclass(frozen=True)
class Colors:
    """
    OOP-implementation for console formatting options.
    """
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    @staticmethod
    def green(string) -> str:
        return '\033[92m' + str(string) + '\033[0m'

    @staticmethod
    def yellow(string) -> str:
        return '\033[93m' + str(string) + '\033[0m'

    @staticmethod
    def red(string) -> str:
        return '\033[91m' + str(string) + '\033[0m'

    @staticmethod
    def bold(string) -> str:
        return '\033[1m' + str(string) + '\033[0m'

    @staticmethod
    def header(string) -> str:
        return '\033[95m' + str(string) + '\033[0m'

    @staticmethod
    def highlight(string) -> str:
        return '\033[43m' + str(string) + '\033[0m'

    @staticmethod
    def blink(string) -> str:
        return '\033[6m' + str(string) + '\033[0m'

    @staticmethod
    def colprint(string, **kwargs) -> str:
        content = str(string)
        color = kwargs.get('color', 'green').lower()
        bold = kwargs.get('bold', False)
        highlight = kwargs.get('highlight', False)

        try:
            if color == "red":
                content = Colors.red(content)
            elif color == "yellow":
                content = Colors.yellow(content)
            elif color == "green":
                content = Colors.green(content)
            else:
                raise Exception(f"Invalid color requested.")

            if bold:
                content = Colors.bold(content)
            if highlight:
                content = Colors.highlight(content)

            print(content)

        except Exception as e:
            print(f"Printing error: {e}")


def timer(func):
    """
    Takes the run-time of a function.

    Wrapper function, used to get timestamp prior and after
    the execution of function.
    """
    def timer_wrapper(*args, **kwargs):
        start_time = time.time()
        rv = func(*args, **kwargs)
        end_time = time.time()
        print(f'\n[Elapsed time: {round((end_time - start_time), 2)}s]\n')
        return rv

    return timer_wrapper


def percent_diff(expected, actual) -> float:
    """
    Calculates the percent difference between 2 values

    Args:
        expected: Expected value
        actual: Real/acquired value

    Returns:
        float value
    """
    sign = 1 if expected > actual else -1
    value = (abs(actual - expected) / ((actual + expected) / 2)) * 100
    return sign * round(value, 2)


def min_max_change(minimum, maximum, base_value) -> dict:
    return {
        'min': percent_diff(minimum, base_value),
        'max': percent_diff(maximum, base_value)
    }


def timestamp() -> str:
    now = datetime.datetime.now()
    return f'{now.strftime("%d %B %Y, %I:%M:%S %p")}'


def vowel_remove(string: str) -> str:
    vowels = ('a', 'e', 'i', 'o', 'u')
    string_list = list(string)
    no_vowels = list(map(lambda c: c if c.lower() not in vowels else "", string_list))

    return "".join(no_vowels)
