import sys

import psycopg2
from config import config

sys.path.insert(0, "../")
from db_connector import DbConnector


def create_table(commands):
    dbConnector = DbConnector()
    dbConnector.cursor.execute(commands)


def table_exists(table):
    dbConnector = DbConnector()
    sql = """SELECT EXISTS(SELECT * FROM information_schema.tables WHERE table_name=%s)"""
    dbConnector.cursor.execute(sql, (table,))
    return dbConnector.cursor.fetchone()[0]


if __name__ == "__main__":
    print(table_exists("cbir_index"))
