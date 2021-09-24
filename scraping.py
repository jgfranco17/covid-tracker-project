"""
SCRAPING.PY

OOP-implementation of a BeautifulSoup web scraper.
"""
from bs4 import BeautifulSoup
import requests
import json
import re
from general import vowel_remove


class Scraper:
    @staticmethod
    def scraping(url):
        site = requests.get(url).text
        doc = BeautifulSoup(site, "html.parser")
        return doc

    def lookup(self, country_name):
        initials = vowel_remove(country_name)[:2]
        lookup_url = f'https://covid19.who.int/region/wpro/country/{initials}'
        page = self.scraping(lookup_url)
        return page

    def news(self, country):
        reddit_url = f'https://www.reddit.com/r/news/search/?q=covid%20{country}&restrict_sr=1'
        news_page = self.scraping(reddit_url)
        info = news_page.find_all(class_="sc-AxhCb jXrfEt")
        return info
