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


if __name__ == "__main__":
    connect()
