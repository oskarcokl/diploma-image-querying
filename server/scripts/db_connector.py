import psycopg2
from config import config


class db_connector:
    def __init__(self):
        try:
            params = config()
            self.connection = psycopg2.connect(**params)
            self.cursor = self.connection.cursor()
        except (Exception, psycopg2.DatabaseError) as e:
            print(e)

    def close(self):
        if self.cursor is not None:
            self.cursor.close()
        if self.connection is not None:
            self.connection.clos()
