import logging
import sys

import psycopg2
import numpy as np

sys.path.insert(0, ".")
from .db_connector import DbConnector

# Type definitions
Vector = "list[float]"
Data = "list[list[int, string, Vector]]"
Tuple = "tuple[str, Vector]"
TupleList = "list[Tuple]"


def create_table(command: str, db_connector: DbConnector = None):
    """
    Create table based on provided command.

    Paramaters
    ----------
    command : string
        SQL type command to create a tabl
    db_connector : DbConnector
        Optional DbConnector object
    """

    if not db_connector:
        db_connector = DbConnector()

    try:
        db_connector.cursor.execute(command)
        db_connector.connection.commit()
    except (Exception, psycopg2.DatabaseError) as e:
        logging.error(e)


def drop_table(table: str, db_connector: DbConnector = None):
    """
    Drop specified table

    Parameters
    ----------
    table : str
        Name of table to drop
    db_connector : DbConnector
        Optional DbConnector object
    """
    if not db_connector:
        db_connector = DbConnector()

    sql = f"DROP TABLE {table}"

    try:
        db_connector.cursor.execute(sql)
        db_connector.cursor.close()
        db_connector.connection.commit()
    except (Exception, psycopg2.DatabaseError) as e:
        logging.error(e)


def table_exists(table: str, db_connector: DbConnector = None) -> bool:
    """
    Check if table exists, returns bool value.

    Parameters
    ----------
    table : str
        Name of table to check
    db_connector : DbConnector
        Optional DbConnector object
    """
    if not db_connector:
        db_connector = DbConnector()

    sql = """SELECT EXISTS(SELECT * FROM information_schema.tables WHERE table_name=%s)"""
    db_connector.cursor.execute(sql, (table,))

    return db_connector.cursor.fetchone()[0]


def insert_tuple_list(tuple_list: TupleList, db_connector: DbConnector = None) -> int:
    """
    Insert list of tuples into cbir_index table.

    Parameters
    ----------
    tuple_list : TupleList
        List of tuples (str, Vector) to be inserted
    db_connector : DbConnector
        Optional DbConnector object
    """

    if not db_connector:
        db_connector = DbConnector()

    sql = """INSERT INTO cbir_index (image_name, image_vector)
             VALUES(%s, %s) RETURNING id;"""

    try:
        logging.info("Writing image vectors to cbir_index table.")
        id = db_connector.cursor.executemany(sql, tuple_list)
        db_connector.connection.commit()
        return id
    except (Exception, psycopg2.DatabaseError) as e:
        logging.error(e)


def insert_tuple(tuple: Tuple, db_connector: DbConnector = None) -> int:
    """
    Insert single tuple into cbir_index table.

    Returns id of inserted tuple.

    Parameters
    ----------
    tuple : Tuple
        Tuple (str, Vector) to be inserted
    db_connector : DbConnector
        Optional DbConnector object
    """

    if not db_connector:
        db_connector = DbConnector()

    sql = """INSERT INTO cbir_index (image_name, image_vector)
             VALUES(%s, %s) RETURNING id;"""

    try:
        logging.info("Writing image vector to database.")
        db_connector.cursor.execute(sql, tuple)
        id = db_connector.cursor.fetchone()[0]
        db_connector.connection.commit()
        return id
    except (Exception, psycopg2.DatabaseError) as e:
        logging.error(e)


def insert_many_tuples(tuple_list: TupleList, db_connector: DbConnector = None) -> "list[int]":
    """
    Insert many tuples into cbir_index table.

    Insertion is done with for loop in which we call insert_tuple().
    Returns ids of inserted tuples.

    Parameters
    ----------
    tuple_list : TupleList
        List of tuples (str, Vector) to be inserted
    db_connector : DbConnector
        Optional DbConnector object
    """

    if not db_connector:
        db_connector = DbConnector()

    ids = []

    for tuple in tuple_list:
        ids.append(insert_tuple(tuple, db_connector=db_connector))
    return ids


def insert_tuple_list_reduced(tuple_list: TupleList, db_connector: DbConnector = None) -> int:
    """
    Insert list of tuples into reduced_features table.

    Parameters
    ----------
    tuple_list : TupleList
        List of tuples (str, Vector) to be inserted
    db_connector : DbConnector
        Optional DbConnector object
    """

    if not db_connector:
        db_connector = DbConnector()

    sql = """INSERT INTO reduced_features (image_name, image_vector)
             VALUES(%s, %s) RETURNING id;"""

    try:
        logging.info("Writing image vectors to database.")
        id = db_connector.cursor.executemany(sql, tuple_list)
        db_connector.connection.commit()
        return id
    except (Exception, psycopg2.DatabaseError) as e:
        logging.error(e)


def insert_tuple_reduced(tuple: Tuple, db_connector: DbConnector = None) -> int:
    """
    Insert single tuple into reduced_features table.

    Returns id of inserted tuple.

    Parameters
    ----------
    tuple : Tuple
        Tuple (str, Vector) to be inserted
    db_connector : DbConnector
        Optional DbConnector object
    """

    if not db_connector:
        db_connector = DbConnector()

    sql = """INSERT INTO reduced_features (image_name, image_vector)
             VALUES(%s, %s) RETURNING id;"""

    try:
        logging.info("Writing image vector to database.")
        db_connector.cursor.execute(sql, tuple)
        id = db_connector.cursor.fetchone()[0]
        db_connector.connection.commit()
        return id
    except (Exception, psycopg2.DatabaseError) as e:
        logging.error(e)


def insert_many_tuples_reduced(tuple_list: TupleList, db_connector: DbConnector = None) -> "list[int]":
    """
    Insert many tuples into reduced_features table.

    Insertion is done with for loop in which we call insert_tuple_reduced().
    Returns ids of inserted tuples.

    Parameters
    ----------
    tuple_list : TupleList
        List of tuples (str, Vector) to be inserted
    db_connector : DbConnector
        Optional DbConnector object
    """

    if not db_connector:
        db_connector = DbConnector()

    ids = []

    for tuple in tuple_list:
        ids.append(insert_tuple_reduced(tuple, db_connector=db_connector))
    return ids


def get_reduced_feature_vector(img_name: str, db_connector: DbConnector = None) -> Vector:
    """
    Get a feature vector from reduced_features table.

    Parameters
    ----------
    img_name : str
        Name of image for which to retrieve feature vector
    db_connector: DbConnector
        Optional DbConnector object
    """

    if not db_connector:
        db_connector = DbConnector()

    sql = """SELECT image_vector FROM reduced_features WHERE image_name=%s"""

    try:
        logging.info(
            f"Getting feature vector for {img_name} from reduced_features table.")
        db_connector.cursor.execute(sql, (img_name,))
        feature_vector = db_connector.cursor.fetchone()[0]
        return feature_vector
    except (Exception, psycopg2.DatabaseError) as e:
        logging.error(e)


def get_reduced_feature_vectors(img_names: "list[str]", db_connector: DbConnector = None) -> "list[Vector]":
    """
    Get feature vectors from reduced_features table for names in img_names.

    Parameters
    ----------
    img_names : list[str]
        List of img names for which to retrieve feature vectors
    db_connector: DbConnector
        Optional DbConnector object
    """

    if not db_connector:
        db_connector = DbConnector()

    try:
        feature_vectors = []

        for img_name in img_names:
            feature_vector = get_reduced_feature_vector(
                img_name, db_connector=db_connector)
            feature_vectors.append(feature_vector)

        return feature_vectors
    except (Exception, psycopg2.DatabaseError) as e:
        logging.error(e)


def get_feature_vector(img_name: str, db_connector: DbConnector = None) -> Vector:
    """
    Get a feature vector from cbir_index table.

    Parameters
    ----------
    img_name : str
        Name of image for which to retrieve feature vector
    db_connector: DbConnector
        Optional DbConnector object
    """

    if not db_connector:
        db_connector = DbConnector()

    sql = """SELECT image_vector FROM cbir_index WHERE image_name=%s"""

    try:
        logging.info(
            f"Getting feature vector for {img_name} from cbir_index table.")
        db_connector.cursor.execute(sql, (img_name,))
        feature_vector = db_connector.cursor.fetchone()[0]
        return feature_vector
    except (Exception, psycopg2.DatabaseError) as e:
        logging.error(e)


def get_feature_vectors(img_names: str, db_connector: DbConnector = None) -> "list[Vector]":
    """
    Get feature vectors from cbir_index table for names in img_names.

    Parameters
    ----------
    img_names : list[str]
        List of img names for which to retrieve feature vectors
    db_connector: DbConnector
        Optional DbConnector object
    """

    if not db_connector:
        db_connector = DbConnector()

    try:
        feature_vectors = []

        for img_name in img_names:
            feature_vector = get_feature_vector(
                img_name, db_connector=db_connector)
            feature_vectors.append(feature_vector)

        return feature_vectors
    except (Exception, psycopg2.DatabaseError) as e:
        logging.error(e)


def get_feature_vectors_all(db_connector: DbConnector = None) -> "list[Vector]":
    """
    Get all feature vector from cbir_index table. 

    Parameters
    ----------
    db_connector: DbConnector
        Optional DbConnector object
    """

    if not db_connector:
        db_connector = DbConnector()

    try:
        logging.info("Getting feature all vectors")
        db_connector.cursor.execute("SELECT image_vector FROM cbir_index")
        data = db_connector.cursor.fetchall()
        data_array = np.array(data, dtype=object)
        feature_vectors = data_array[:, 0]
        return feature_vectors
    except (Exception, psycopg2.DatabaseError) as e:
        logging.error(e)


def get_data_all(db_connector: DbConnector = None) -> Data:
    """
    Get all table entries from cbir_index table

    Parameters
    ----------
    db_connector: DbConnector
        Optional DbConnector object 
    """
    if not db_connector:
        db_connector = DbConnector()
    try:
        db_connector.cursor.execute("SELECT * FROM cbir_index")
        data = db_connector.cursor.fetchall()
        logging.info("Number of indexed images: ", len(data))
        data_array = np.array(data, dtype=object)
        return data_array
    except (Exception, psycopg2.DatabaseError) as e:
        logging.error(e)
