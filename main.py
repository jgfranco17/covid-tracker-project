"""
Created on 12 Sept 2021
@author: Chino Franco

COVID-19 data tracking for different countries using API requests
from https://api.covid19api.com/ routes.
"""
from analysis import CovidData
from controller import Controller


if __name__ == "__main__":
    location = 'Japan'
    category = 'Confirmed'

    japan_tracker = CovidData(location)
    controls = Controller(japan_tracker)
    controls.run()

