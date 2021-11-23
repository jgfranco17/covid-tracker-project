"""
ANALYSIS.PY

Hosts the class architecture of the COVID-tracker object.
"""
# General Python modules
import requests
import json
import csv
import numpy as np
import pandas as pd
import logging as lg
import datetime as dt
import traceback as tb
from os import path
from scipy import stats
from matplotlib import pyplot as plt
from sqlalchemy.orm import sessionmaker, class_mapper, ColumnProperty
from tqdm import tqdm as loading
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer

# Project files
from models import Point, create_database
from general import Colors, timer, timestamp
from customexceptions import InvalidCategoryError, DataBoundsError


class CovidData(object):
    """
    Class representation of the COVID-19 Tracker-Modelling system.

    Properties:
       country (str): Country name
       point_count (int): Number of collected data points
       categories (list): List of data categories
       engine (Engine): SQLAlchemy engine for database commits
       db_location (str): File location of SQLite database
       db (Session): Local session for tracker-database connection
    """

    def __init__(self, country: str):
        self.country = country.title()
        self.point_count = 0
        self.categories = [prop.key.title() for prop in class_mapper(Point).iterate_properties if
                           isinstance(prop, ColumnProperty)]
        self.transformer = FunctionTransformer(np.log, validate=True)
        self.regressor = LinearRegression()
        self.data = {"active": None, "confirmed": None, "deaths": None, "date": None}

        # Database storage setup
        self.db_location, self.engine = create_database(country.lower())
        Session = sessionmaker()
        self.db = Session(bind=self.engine)

        # Set up prediction model
        self.day_range = 0
        self.predicted_values = []
        self.max_cases = None
        self.min_cases = None
        self.max_day = 0
        self.min_day = 0
        self.lr_prediction = 0
        self.svm_prediction = 0

        # Configure logging
        log_fmt = "[%(levelname)s] %(asctime)s %(message)s"
        lg.basicConfig(level=lg.DEBUG, format=log_fmt)
        lg.getLogger('matplotlib.font_manager').setLevel(lg.WARNING)

        print(Colors.green(f"\n[{timestamp()}]"))
        print(f"Data Tracker object created for {Colors.bold(self.country)}.")

    def __len__(self):
        return self.point_count

    def __repr__(self):
        return f'<COVID-19 Tracker [{self.country}], {self.point_count} data points>'

    def __eq__(self, other):
        return self.country == other.country and self.point_count == other.point_count

    def processing(func):
        """
        Takes the stored country name and gathers data from the API,
        stores it in the object's database.

        Wrapper function, used to run data retrieval and processing prior
        to the execution of another data-dependent function.
        """

        def load_data(self, *args, **kwargs):
            url = self.get_link()

            try:
                # Get JSON data via API call
                info = json.loads(requests.get(url).text)
                if not self.point_count:
                    print(f'Compiling data points gathered for {Colors.bold(self.country)}.')
                self.point_count = int(len(info))

                # Gathers data for each day, stores in database
                new_added = 0
                for point in loading(range(self.point_count), desc=Colors.blink('Loading points')):
                    datapoint = info[point]
                    point_date = dt.datetime.strptime(datapoint["Date"], '%Y-%m-%dT%H:%M:%SZ')

                    check = self.db.query(Point).filter(
                        Point.country == self.country, Point.date == point_date
                    )
                    if check.first() is None:
                        self.add_point(
                            country=self.country,
                            date=point_date,
                            active=datapoint["Active"],
                            confirmed=datapoint["Confirmed"],
                            deaths=datapoint["Deaths"]
                        )
                        new_added += 1

                print(f'Compiled {Colors.yellow(self.point_count)} data points.')

                if new_added != 0:
                    print(f"Added in {new_added} additional points.")

            except Exception as e:
                print(f'{Colors.red("ERROR - Data failed to compile.")}\nReason: {e}')

            finally:
                # Ensure that the decorator returns the function regardless
                return func(self, *args, **kwargs)

        return load_data

    def loadup(func):
        """
        Takes the stored country name and gathers data from the API,
        stores it in the object's database.

        Wrapper function, used to run data retrieval and processing prior
        to the execution of another data-dependent function.
        """

        def load_wrapper(self, *args, **kwargs):
            points = self.db.query(Point).filter(Point.country == self.country)
            try:
                self.data = {
                    "active": [point.active for point in points],
                    "confirmed": [point.confirmed for point in points],
                    "deaths": [point.deaths for point in points],
                    "date": [point.date for point in points]
                }

                print(f'Loaded data points for {self.country}.')

            except Exception as e:
                print(f'{Colors.red("ERROR - Data failed to load.")}\nReason: {e}')

            finally:
                # Ensure that the decorator returns the function regardless
                return func(self, *args, **kwargs)

        return load_wrapper

    def add_point(self, **kwargs):
        """
        Stores the point data in the SQLAlchemy database.

        Keyword Args:
            country (str): Country of data point
            date (datetime.datetime): Date of recorded point
            active (int): Number of active cases in country
            confirmed (int): Number of confirmed cases (total)
            deaths (int): Number of deaths from COVID-19
        """
        country = kwargs.get('country', None)
        date = kwargs.get('date', None)
        active = kwargs.get('active', None)
        confirmed = kwargs.get('confirmed', None)
        deaths = kwargs.get('deaths', None)

        try:
            if None in kwargs.values():
                raise Exception("Invalid NoneType parameter given.")

            new_point = Point(
                country=country,
                date=date,
                active=active,
                confirmed=confirmed,
                deaths=deaths
            )
            self.db.add(new_point)
            self.db.commit()

        except Exception as e:
            print(f'Error during Point commit: {e}')

    def get_link(self) -> str:
        return f'https://api.covid19api.com/total/country/{self.country.lower()}'

    @staticmethod
    def linear_data(x, y) -> dict:
        slope, intercept, r, p, std_err = stats.linregress(x, y)
        lin_rel = round((r ** 2), 4)
        line = lambda t: (slope * t) + intercept
        model = list(map(line, x))
        sign = "+" if intercept > 0 else "-"

        return {
            'slope': slope,
            'intercept': intercept,
            'relation': lin_rel,
            'model': model,
            'equation': f'{slope:,.2f}t {sign} {abs(intercept):,.2f}',
            'function': line
        }

    @staticmethod
    def polynomial_data(x, y, deg: int = 2) -> dict:
        fit = np.polyfit(x, y, deg)
        polynomial = np.poly1d(fit)
        line = np.linspace(x[0], x[-1], max(y))
        poly_rel = round(r2_score(y, polynomial(x)), 4)
        coefficients = list(map(lambda c: float(c), fit))
        eq_comp = [
            f'{"+" if coefficients[i+1] > 0 else "-"} {abs(coefficients[i]):,.2f}t^{deg-i+1}' for i in
            range(deg) if round(coefficients[i], 2) != 0
        ]
        poly_eq_form = ' '.join(eq_comp)

        return {
            'relation': poly_rel,
            'line': line,
            'polynomial': polynomial(line),
            'equation': poly_eq_form
        }

    @staticmethod
    def logarithmic_data(x, y) -> dict:
        logfit = np.polyfit(np.log(x), y, 1)
        A = f'{"+" if logfit[0] > 0 else "-"} {abs(logfit[0]):,.2f}'
        B = f'{"+" if logfit[0] > 0 else "-"} {abs(logfit[1]):,.2f}'

        return {
            'log': logfit,
            'equation': f'{A} + {B}log(x)',
            'A': logfit[0],
            'B': logfit[1]
        }

    @staticmethod
    def percent_diff(expected, actual) -> float:
        """
        Calculates the percent difference between 2 values

        Args:
            expected: Expected value
            actual: Real/acquired value

        Returns:
            Float value
        """
        sign = 1 if expected > actual else -1
        value = (abs(actual - expected) / ((actual + expected) / 2)) * 100
        return sign * round(value, 2)

    @staticmethod
    def logfunc(var, coeff_outer, coeff_inner, constant) -> float:
        """
        Return values from a general log function.
        """
        return coeff_outer * np.log(coeff_inner * var) + constant

    def min_max_change(self, minimum, maximum, base_value) -> dict:
        """

        Args:
            minimum: Smaller value
            maximum: Larger value
            base_value: base number to compare to

        Returns:
            Dictionary of results
        """
        return {
            'min': self.percent_diff(minimum, base_value),
            'max': self.percent_diff(maximum, base_value)
        }

    @timer
    @processing
    def case_updates(self, **kwargs):
        """
        Gets the data of cases over a given span of days. Compares the data analytically.

        Keyword Args:
            start (int): Date to start the analysis
            end (int): Date to end the analysis
            plot (bool): Check if the model should plot a chart
        """
        try:
            # Check keyword arguments for specifiers
            category = kwargs.get('category', 'confirmed').lower()
            span = kwargs.get('span', 5)
            plot = kwargs.get('plot', False)
            avg = lambda d: sum(d) / len(d)
            if span > self.point_count:
                raise DataBoundsError(limit=self.point_count, attempt=span)

            # Setting up raw data
            points = self.db.query(Point).filter(Point.country == self.country)

            if category == "confirmed":
                data_set = [point.confirmed for point in points][-span:]
            elif category == "active":
                data_set = [point.active for point in points][-span:]
            elif category == "deaths":
                data_set = [point.deaths for point in points][-span:]
            else:
                raise InvalidCategoryError()

            if plot:
                self.data_plot(
                    data_set=data_set,
                    polynomial=True,
                    ylabel=category.title()
                )

            print("-" * 25)
            print(f"{self.country.upper()} STATISTICAL DATA")
            print(f"Presenting {category} ({span} of {self.point_count} data points).")
            print(f"Highest: {max(data_set):,}\nLowest: {min(data_set):,}")
            print(f"{span}-day average: {avg(data_set):,.2f}")
            print(f"Standard deviation over {span} days: {np.std(data_set):,.2f}")

        except Exception as e:
            print(f'ERROR - Data failed to compare.\nReason: {e}')

    @processing
    def update(self):
        print(f"Updated data for {self.country}.")

    @timer
    @processing
    @loadup
    def predict(self, forecast: int = 15, category: str = "active"):
        """
        Uses SVM learning algorithm to predict data for COVID-19
        category of choice.

        Args:
            forecast (int): Duration of prediction forecast
            category (str): Data set to predict
        """
        show_data = False
        self.day_range = forecast  # as forecast duration increases, accuracy decreases

        # Set up dataframe
        data_set = self.data[category.lower()]
        dates = self.data["date"]
        print(type(data_set), type(dates))
        idx = [i + 1 for i in range(len(data_set))]
        df = pd.DataFrame(
            data_set,
            index=idx,
            columns=[category.title()]
        )
        print(f'Getting data for {self.country}: {category}')

        df['Prediction'] = df[[category.title()]].shift(-forecast)
        print(f'{category.title()} data: {len(df)} data points')
        dropped = df.drop(['Prediction'])
        x_data = np.array(dropped, 1)
        y_data = np.array(df['Prediction'])
        print("Successfully converted the dataframes into arrays.")
        X = x_data[:-forecast]
        Y = y_data[:-forecast]
        print(dropped)

        # Creating the predictive model
        try:
            # Setting up the training via SVR
            x_tr, x_te, y_tr, y_te = train_test_split(X, Y, test_size=0.2)
            svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
            svr_rbf.fit(x_tr, y_tr)

            # Model confidence scoring
            svm_confidence = svr_rbf.score(x_te, y_te)
            self.regressor.fit(x_tr, y_tr)
            lr_confidence = self.regressor.score(x_te, y_te)
            print(f'SV Model Confidence: {round(svm_confidence * 100, 3)}%')
            print(f'Regression Confidence: {round(lr_confidence * 100, 3)}%')

            # Generate SVR model predictions
            x_forecast = x_data[-forecast:]
            self.lr_prediction = self.regressor.predict(x_forecast)
            self.svm_prediction = svr_rbf.predict(x_forecast)
            if show_data:
                print(f'SVR Data Prediction:\n{self.svm_prediction}')
                print(f'Linear Regression Data Prediction:\n{self.lr_prediction}')

            # Compiling the data sets
            predicted_values = (self.lr_prediction + self.svm_prediction) / 2
            self.max_cases = float(round(np.max(predicted_values), 2))
            self.min_cases = float(round(np.min(predicted_values), 2))
            self.predicted_values = predicted_values.tolist()
            self.get_model_data()

        except Exception as err:
            print(f'Error during training model: {err}')
            tb.print_exc()

    def get_model_data(self):
        print('--- KEY INFORMATION ---')
        print(f'COUNTRY: {self.country}\n')
        print(f'--- CURRENT DATA FOR THE NEXT {self.day_range} DAYS ---')
        print(f'MAX: {self.max_cases}, in {self.max_day} days')
        print(f'MIN: {self.min_cases}, in {self.min_day} days')

    def data_plot(self, data_set: list, **kwargs):
        """
        Generates plot of country data from Day 1 to present.

        Args:
            data_set (list): Data set to be worked with

        Keyword Args:
            lin (bool): Check if a linear plot should be made
            poly (bool): Check if a polynomial plot should be made
            log (bool): Check if a logarithmic plot should be made
            x_label (str): X-axis chart label
            y_label (str): Y-axis chart label
        """
        try:
            # Generate plot data and basic scatter plot
            span = len(data_set)
            lin = kwargs.get('linear', False)
            poly = kwargs.get('polynomial', False)
            log = kwargs.get('logarithmic', False)
            x_label = kwargs.get('xlabel', f'Last {span} Days')
            y_label = kwargs.get('ylabel', "Data")
            lin_rel, poly_rel = 0, 0
            x, y = list(range(1, len(data_set) + 1)), data_set
            plt.scatter(x, y)

            # Additional analyses and formatting
            if lin:
                lin_plot_data = self.linear_data(x, y)
                plt.plot(x, lin_plot_data['model'], 'r-')
                print("-" * 25)
                print(f"C(t) = {lin_plot_data['equation']}")
                print(f"Linear R-squared: {lin_plot_data['relation']}")
                lin_rel = lin_plot_data['relation']

            if poly:
                poly_plot_data = self.polynomial_data(x, y, 3)
                plt.plot(poly_plot_data['line'], poly_plot_data['polynomial'], 'g-')
                print("-" * 25)
                print(f"C(t) = {poly_plot_data['equation']}")
                print(f"Polynomial R-squared: {poly_plot_data['relation']}")
                poly_rel = poly_plot_data['relation']

            if log:
                x_trans = self.transformer.fit_transform(x)
                regressor = LinearRegression()
                results = regressor.fit(x_trans, y)
                y_fit = results.predict(x_trans)
                plt.plot(x, y_fit, c="F92672")
                log_plot_data = self.logarithmic_data(x, y)
                print("-" * 25)
                print(f"C(t) = {log_plot_data['equation']}")

            # If both are collected, draw comparison of regressions
            if lin and poly:
                values = {'Linear': lin_rel, 'Polynomial': poly_rel}
                higher, lower = max(values, key=values.get), min(values, key=values.get)
                higher_reg, lower_reg = values[lower], round(values[higher] - values[lower], 5)
                print(f"{higher} regression yielded a higher R^2 at {values[higher]}")
                print(f"{lower} regression was only {higher_reg}, around {lower_reg} lower")

            # Chart formatting and labels
            plt.xlim(0, span)
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.title(f'COVID-19 Data: {self.country}, {span} days')
            plt.show()

        except Exception as e:
            print(f'ERROR - Data failed to plot.\nReason: {e}')
            tb.print_exc()

    @processing
    @loadup
    def export_data(self):
        """
        Export gathered data as CSV file.
        """
        csv_file = f'data_{self.country.lower()}.csv'
        try:
            if not path.exists(csv_file):
                with open(csv_file, 'w', encoding='UTF8') as f:
                    header = ["Date", "Active", "Confirmed", "Deaths"]
                    writer = csv.writer(f)
                    writer.writerow(header)

                    # Go through list of stored points, write to file
                    for i in range(self.point_count):
                        data_input = [
                            str(self.data["date"][i]),
                            str(self.data["active"][i]),
                            str(self.data["confirmed"][i]),
                            str(self.data["deaths"][i])
                        ]
                        writer.writerow(data_input)

                print("Data compiled and exported to CSV file.")

            else:
                print(f'Data for {self.country} has already been exported to CSV file.')

        except Exception as e:
            lg.warn(f'Export error: {e}')
