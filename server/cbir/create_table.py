import sys

import psycopg2
from config import config

sys.path.insert(0, "../")
from db_connector import DbConnector


def create_table(commands):
    dbConnector = DbConnector()
    dbConnector.cursor.execute(commands)
    dbConnector.connection.commit()


def drop_table(table):
    dbConnector = DbConnector()
    sql = f"DROP TABLE {table}"
    print(sql)
    try:
        dbConnector.cursor.execute(sql)
        dbConnector.connection.commit()
    except (Exception, psycopg2.DatabaseError) as e:
        print(e)


def table_exists(table):
    dbConnector = DbConnector()
    sql = """SELECT EXISTS(SELECT * FROM information_schema.tables WHERE table_name=%s)"""
    dbConnector.cursor.execute(sql, (table,))
    return dbConnector.cursor.fetchone()[0]


if __name__ == "__main__":
    drop_table("cbir_index")
