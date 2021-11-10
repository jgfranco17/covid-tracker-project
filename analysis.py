"""
ANALYSIS.PY

Hosts the class architecture of the COVID-tracker object.
"""
# General Python modules
import matplotlib.pyplot as plt
import requests
import json
import numpy as np
import datetime as dt
import traceback as tb
from scipy import stats
from sqlalchemy.orm import sessionmaker, class_mapper, ColumnProperty
from tqdm import tqdm as loading
from sklearn.metrics import r2_score

# Project files
from models import Point, create_database
from general import Colors, timer, timestamp
from scraping import Scraper


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
        self.scraper = Scraper()

        # Database storage setup
        self.db_location, self.engine = create_database(country.lower())
        Session = sessionmaker()
        self.db = Session(bind=self.engine)

        print(Colors.green(f"\n[{timestamp()}]"))
        print(f"Data Tracker object created for {Colors.bold(self.country)}.")

    def __len__(self):
        return self.point_count

    def __repr__(self):
        return f'<COVID-19 Tracker [{self.country}], {self.point_count} data points>'

    def __eq__(self, other):
        return self.country == other.country

    def processing(func):
        """
        Takes the stored country name and gathers data from the API,
        stores it in the object's database.

        Wrapper function, used to run data retrieval and processing prior
        to the execution of another data-dependent function.
        """

        def load_data(self, *args, **kwargs):
            url = f'https://api.covid19api.com/total/country/{self.country.lower()}'

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

                    check = self.db.query(Point).filter(Point.country == self.country, Point.date == point_date).first()
                    if check is None:
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
                tb.print_exc()

            finally:
                return func(self, *args, **kwargs)

        return load_data

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
    def polynomial_data(x, y, deg: int = 2):
        fit = np.polyfit(x, y, deg)
        polynomial = np.poly1d(fit)
        line = np.linspace(x[0], x[-1], max(y))
        poly_rel = round(r2_score(y, polynomial(x)), 4)
        coefficients = list(map(lambda c: float(c), fit))
        eq_comp = [f'{"+" if coefficients[i] > 0 else "-"} {abs(coefficients[i]):,.2f}t^{deg - i}' for i in
                   range(deg + 1)]
        poly_eq_form = ' '.join(eq_comp)

        return {
            'relation': poly_rel,
            'line': line,
            'polynomial': polynomial(line),
            'equation': poly_eq_form
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
                raise Exception(f"Invalid time span; only {self.point_count} data points available but {span} were requested.")

            # Setting up raw data
            points = self.db.query(Point).filter(Point.country == self.country)

            if category == "confirmed":
                data_set = [point.confirmed for point in points][-span:]
            elif category == "active":
                data_set = [point.active for point in points][-span:]
            elif category == "deaths":
                data_set = [point.deaths for point in points][-span:]
            else:
                raise ValueError(f"Invalid category provided")

            if plot:
                self.data_plot(
                    data_set=data_set,
                    polynomial=True
                )

            print("-" * 25)
            print(f"{self.country.upper()} STATISTICAL DATA")
            print(f"Presenting {category} ({span} of {self.point_count} data points)")
            print(f"{span}-day average: {avg(data_set):,.2f}")
            print(f"Highest: {max(data_set):,}\nLowest: {min(data_set):,}")
            print(f"Standard deviation over {span} days: {np.std(data_set):,.2f}")

        except Exception as e:
            print(f'ERROR - Data failed to compare.\nReason: {e}')
            tb.print_exc()

    @processing
    def update(self):
        print(f"Updated data for {self.country}.")

    def data_plot(self, data_set: list, **kwargs):
        """[summary]
        Generates plot of country data from Day 1 to present.

        Args:
            data_set (list): Data set to be worked with

        Keyword Args:
            lin (bool): Check if a linear plot should be made
            poly (bool): Check if a polynomial plot should be made
            x_label (str): X-axis chart label
            y_label (str): Y-axis chart label
        """
        try:
            # Generate plot data and basic scatter plot
            span = len(data_set)
            lin = kwargs.get('linear', False)
            poly = kwargs.get('polynomial', False)
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

            # If both are collected, draw comparison of regressions
            if lin and poly:
                print("-" * 25)
                values = {'Linear': lin_rel, 'Polynomial': poly_rel}
                higher, lower = max(values, key=values.get), min(values, key=values.get)
                print(f"{higher} regression yielded a higher R^2 at {values[higher]}")
                print(f"{lower} regression was only {values[lower]}, around {values[higher] - values[lower]:.5f} lower")

            # Chart formatting and labels
            print("-" * 25)
            plt.xlim(0, span)
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.title(f'COVID-19 Data: {self.country}, {span} days')
            plt.show()

        except Exception as e:
            print(f'ERROR - Data failed to plot.\nReason: {e}')
            tb.print_exc()

    def compare_data(self, other, **kwargs):
        """
        Compares the data of this tracker's country and that of another tracker object.

        Args:
            other (CovidData): CovidData tracker object instance to compare with
        """
        category = kwargs.get('category', 'active').lower()
        span = kwargs.get('span', 30)
        main_objects = [self, other]

        try:
            # Create collection of data
            collection = {}
            for cls in loading(main_objects, desc='Loading data for comparison'):
                points = cls.db.query(Point).filter(Point.country == cls.country)
                average = lambda d: sum(d) / len(d)

                if category == "confirmed":
                    data = [point.confirmed for point in points]
                elif category == "active":
                    data = [point.active for point in points]
                elif category == "deaths":
                    data = [point.confirmed for point in points]
                else:
                    raise Exception(f"Invalid category provided.")

                entry = {
                    'country': cls.country,
                    'data': data[-span:],
                    'max': max(data[-span:]),
                    'min': min(data[-span:]),
                    'average': round(average(data[-span:]), 2),
                    'std dev': round(float(np.std(data[-span:])), 2)
                }

                collection.update({cls.country: entry})
                print(f"Gathered data for {cls.country}.")

            # Generate plot data and basic scatter plot
            data_own, data_other = collection[self.country]['data'], collection[other.country]['data']
            x, y_own, y_other = list(range(1, len(data_own) + 1)), data_own, data_other
            plt.scatter(x, y_own)
            plt.scatter(x, y_other)
            plt.xlim(0, span + 1)
            plt.xlabel(f'Last {span} Days')
            plt.ylabel(f'{category.title()} Cases')
            plt.title(f'COVID-19 Data: {", ".join([cls.country for cls in main_objects])} ({span} days)')
            plt.show()

            # Display each country's data
            for country in collection:
                data_set = collection[country]
                print("-" * 25)
                print(f"{country.upper()} DATA")
                print(f"{span}-day average: {data_set['average']:,}")
                print(f"Highest: {data_set['max']:,} | Lowest: {data_set['min']:,}")
                print(f"Data range: {data_set['max'] - data_set['min']:,}")
                print(f"Standard deviation over {span} days: {data_set['std dev']:,.2f}")

        except Exception as e:
            print(f'ERROR - Failed to compare trackers.\nReason: {e}')
            tb.print_exc()
