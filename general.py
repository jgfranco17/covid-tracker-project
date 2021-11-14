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
    def colprint(string, **kwargs):
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
        runtime = time_interval(start_time)
        print(f'\n[Elapsed time: {runtime}]\n')
        return rv

    return timer_wrapper


def time_interval(start_period: float) -> str:
    """
    Calculates time interval, given a starting time.

    Parameters:
        start_period (float): Starting time

    Returns:
        time interval (str)
    """
    def convert_time_diff(duration):
        duration = round(duration, 2)
        d_hours = int(duration // 3600)
        d_minutes = int((duration - (d_hours * 3600)) // 60)
        d_seconds = int(round(duration - (d_hours * 3600) - (d_minutes * 60)))

        return d_hours, d_minutes, d_seconds

    try:
        hours, minutes, seconds = convert_time_diff(time.time() - start_period)
        time_elapsed = []

        if hours:
            time_elapsed.append(f'{hours}h')
        if minutes:
            time_elapsed.append(f'{minutes}m')
        if seconds:
            time_elapsed.append(f'{seconds}s')

        return str(' '.join(time_elapsed))

    except Exception as e:
        print(f'Time interval computation error: {e}')
        return "ERROR"


def timestamp() -> str:
    now = datetime.datetime.now()
    return f'{now.strftime("%d %B %Y, %I:%M:%S %p")}'


def vowel_remove(string: str) -> str:
    vowels = ('a', 'e', 'i', 'o', 'u')
    string_list = list(string)
    no_vowels = list(map(lambda c: c if c.lower() not in vowels else "", string_list))

    return "".join(no_vowels)
