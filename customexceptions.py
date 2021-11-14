"""
CUSTOMEXCEPTIONS.PY

Module for building custom exceptions for the project.
"""

class InvalidCategoryError(Exception):
    """
    Exception raised for errors in the input salary.

    Attributes:
        message: Explanation of the error
    """

    def __init__(self, message="Invalid category provided."):
        self.message = message
        super().__init__(self.message)


class DataBoundsError(Exception):
    """
    Exception raised for errors in the input salary.

    Attributes:
        message: Explanation of the error
    """

    def __init__(self, limit, attempt):
        self.message = f'Only {limit} data points available but {attempt} were requested.'
        super().__init__(self.message)
