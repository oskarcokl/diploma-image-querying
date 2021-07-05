import psycopg2
import sys

sys.path.insert(0, "./cbir/")
from cbir.config import config


class DbConnector:
    def __init__(self):
        try:
            params = config()
            self.connection = psycopg2.connect(**params)
            self.cursor = self.connection.cursor()
        except (Exception, psycopg2.DatabaseError) as e:
            print(e)

    def close(self):
        try:
            if self.cursor is not None:
                self.cursor.close()
            if self.connection is not None:
                self.connection.close()
        except (Exception) as e:
            print(e)
