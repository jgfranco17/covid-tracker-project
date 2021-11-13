"""
Created on 12 Sept 2021
@author: Chino Franco

COVID-19 data tracking for different countries using API requests
from https://api.covid19api.com/ routes.
"""
from analysis import CovidData
from controller import Controller

def main():
    location = 'Japan'
    category = 'deaths'

    japan_tracker = CovidData(location)
    japan_tracker.case_updates(category=category, span=60, plot=True)


if __name__ == "__main__":
    main()

