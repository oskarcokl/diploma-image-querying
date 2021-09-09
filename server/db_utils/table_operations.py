import logging
import sys

import psycopg2

sys.path.insert(0, ".")
from .db_connector import DbConnector


def create_table(command, db_connector=None):
    if not db_connector:
        db_connector = DbConnector()
    try:
        db_connector.cursor.execute(command)
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


def insert_tuple(tuple, db_connector=None):
    if not db_connector:
        db_connector = DbConnector()
    sql = """INSERT INTO cbir_index (image_name, image_vector)
             VALUES(%s, %s) RETURNING id;"""

    try:
        print("Writing image vector to database.")
        db_connector.cursor.execute(sql, tuple)
        id = db_connector.cursor.fetchone()[0]
        db_connector.connection.commit()
        return id
    except (Exception, psycopg2.DatabaseError) as e:
        print(e)


def insert_many_tuples(tuple_list):
    db_connector = DbConnector()
    ids = []

    for tuple in tuple_list:
        ids.append(insert_tuple(tuple, db_connector=db_connector))
    return ids


def insert_tuple_reduced(tuple, db_connector=None):
    if not db_connector:
        db_connector = DbConnector()
    sql = """INSERT INTO reduced_features (image_name, image_vector)
             VALUES(%s, %s) RETURNING id;"""

    try:
        print("Writing image vector to database.")
        db_connector.cursor.execute(sql, tuple)
        id = db_connector.cursor.fetchone()[0]
        db_connector.connection.commit()
        return id
    except (Exception, psycopg2.DatabaseError) as e:
        print(e)


def insert_many_tuples_reduced(tuple_list):
    db_connector = DbConnector()
    ids = []

    for tuple in tuple_list:
        ids.append(insert_tuple_reduced(tuple, db_connector=db_connector))
    return ids


def get_reduced_feature_vector(img_name, db_connector=None):
    if not db_connector:
        db_connector = DbConnector()

    sql = """SELECT image_vector FROM reduced_features WHERE image_name=%s"""

    try:
        #logging.info(f"Getting feature vector for {img_name}")
        db_connector.cursor.execute(sql, (img_name,))
        feature_vector = db_connector.cursor.fetchone()[0]
        return feature_vector
    except (Exception, psycopg2.DatabaseError) as e:
        print(e)


def get_reduced_feature_vectors(table_name, img_names):
    try:
        db_connector = DbConnector()
        feature_vectors = []
        for img_name in img_names:
            feature_vector = get_reduced_feature_vector(
                table_name, img_name, db_connector=db_connector)
            feature_vectors.append(feature_vector)

        return feature_vectors
    except (Exception, psycopg2.DatabaseError) as e:
        print(e)


if __name__ == "__main__":
    drop_table("cbir_index")
