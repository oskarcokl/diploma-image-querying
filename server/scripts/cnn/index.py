import psycopg2
from config import config
from create_table import create_table
import argparse


def insert_image_vector(image_name, image_vector):
    sql = """INSERT INTO cbir_index(image_name, image_vector) 
             VALUES(%s, %s) RETURNING id;"""

    connection = None
    image_id = None
    try:
        params = config()
        connection = psycopg2.connect(**params)
        cursor = connection.cursor()
        cursor.execute(sql, (image_name, image_vector))
        image_id = cursor.fetchone()[0]
        connection.commit()
        cursor.close()
    except (Exception, psycopg2.DatabaseError) as e:
        print(e)
    finally:
        if connection is not None:
            connection.close()

    return image_id


def insert_image_vector_connection(image_name, image_vector, connection):
    sql = """INSERT INTO cbir_index(image_name, image_vector) 
             VALUES(%s, %s) RETURNING id;"""

    image_id = None
    try:
        cursor = connection.cursor()
        cursor.execute(sql, (image_name, image_vector))
        image_id = cursor.fetchone()[0]
        connection.commit()
        cursor.close()
    except (Exception, psycopg2.DatabaseError) as e:
        print(e)

    return image_id


# This function is intented to be run only when setting up the initial db.
def init_index():
    commands = (
        """
        CREATE TABLE cbir_index (
            id SERIAL PRIMARY KEY,
            image_name VARCHAR(255) NOT NULL,
            image_vector INTEGER[4096]
        )
        """,
    )
    create_table(commands)


if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument(
        "-d",
        "--dataset",
        help="Path to directory that contains the images to be indexed",
    )
    argParser.add_argument(
        "-I",
        "--init",
        help="Set if initializing the db for the first time. Will add all pictures in dataset path",
        action="store_true",
    )
    args = vars(argParser.parse_args())
    if args.get("init"):
        init_index(args.get("dataset"))
    else:
        index()
