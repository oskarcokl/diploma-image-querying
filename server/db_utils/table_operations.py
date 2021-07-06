import sys

import psycopg2

sys.path.insert(0, "../")
from db_connector import DbConnector


def create_table(command):
    db_connector = DbConnector()
    try:
        db_connector.cursor.execute(command)
        db_connector.cursor.close()
        db_connector.connection.commit()
    except (Exception, psycopg2.DatabaseError) as e:
        print(e)


def drop_table(table):
    db_connector = DbConnector()
    sql = f"DROP TABLE {table}"
    try:
        db_connector.cursor.execute(sql)
        db_connector.cursor.close()
        db_connector.connection.commit()
    except (Exception, psycopg2.DatabaseError) as e:
        print(e)


def table_exists(table):
    db_connector = DbConnector()
    sql = """SELECT EXISTS(SELECT * FROM information_schema.tables WHERE table_name=%s)"""
    db_connector.cursor.execute(sql, (table,))
    return db_connector.cursor.fetchone()[0]


def insert_tuple_list(tuple_list):
    db_connector = DbConnector()
    sql = """INSERT INTO cbir_index (image_name, image_vector)
             VALUES(%s, %s) RETURNING id;"""

    try:
        print("Writing image vectors to database.")
        id = db_connector.cursor.executemany(sql, tuple_list)
        db_connector.connection.commit()
        return id
    except (Exception, psycopg2.DatabaseError) as e:
        print(e)


def inser_tuple(tuple):
    db_connector = DbConnector()
    sql = """INSERT INTO cbir_index (image_name, image_vector)
             VALUES(%s, %s) RETURNING id;"""

    try:
        print("Writing image vector to database.")
        id = db_connector.cursor.execute(sql, tuple)
        db_connector.connection.commit()
        return id
    except (Exception, psycopg2.DatabaseError) as e:
        print(e)


if __name__ == "__main__":
    drop_table("cbir_index")
