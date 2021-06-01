import os
import numpy as np
from sklearn.neighbors import NearestNeighbors
import psycopg2
from config import config


class Searcher:
    def __init__(self, index_path):
        self.index_path = index_path

    def search(self, query_features, n_neigbhours):
        results = {}
        feature_list = []

        try:
            params = config()
            connection = psycopg2

