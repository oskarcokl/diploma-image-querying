import sys
import logging

import psycopg2

sys.path.insert(0, ".")
from .config import config


class DbConnector:
    """
    DbConncector is used to connect to PostgreSQL database.

    The operations on tables contained in the database are 
    in the table_operations.py file. The paramaters of the
    connection are specified in the config.py file in the
    db_utils module.

    Attributes
    ----------
    connection : connection
        a connection to the PostgreSQL database.
    cursor : cursor
        cursor of the connection. Used to exectue operations 
        on the database.
    """

    def __init__(self):
        try:
            params = config()
            self.connection = psycopg2.connect(**params)
            self.cursor = self.connection.cursor()
        except (Exception, psycopg2.DatabaseError) as e:
            logging.error(e)

    def close(self):
        try:
            if self.cursor is not None:
                self.cursor.close()
            if self.connection is not None:
                self.connection.close()
        except (Exception) as e:
            logging.error(e)
