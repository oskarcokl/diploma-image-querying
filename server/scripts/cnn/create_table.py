import psycopg2
from config import config


def create_table():
    commands = (
        """
        CREATE TABLE cbir_index (
            id SERIAL PRIMARY KEY,
            image_name VARCHAR(255) NOT NULL,
            image_vector INTEGER[4096]
        )
        """,
    )

    connection = None
    try:
        params = config()
        connection = psycopg2.connect(**params)
        cursor = connection.cursor()

        for command in commands:
            cursor.execute(command)

        cursor.close()
        connection.commit()
    except (Exception, psycopg2.DatabaseError) as e:
        print(e)
    finally:
        if connection is not None:
            connection.close()


if __name__ == "__main__":
    create_table()
