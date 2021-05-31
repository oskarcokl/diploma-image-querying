import psycopg2
from config import config


def connect():
    # Connect to PostgreSQL database
    connection = None

    try:
        params = config()
        print("Connectino to PostgresQL database...")
        connection = psycopg2.connect(**params)

        cursor = connection.cursor()

        print("PostgreSQL databse version: ")
        cursor.execute("SELECT version()")

        db_version = cursor.fetchone()
        print(db_version)

        cursor.close()
    except (Exception, psycopg2.DatabaseError) as e:
        print(e)
    finally:
        if connection is not None:
            connection.close()
            print("Database connection close.")


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


if __name__ == "__main__":
    image_name = "example1.jpg"
    image_vector = [69] * 4096
    print(insert_image_vector(image_name, image_vector))
