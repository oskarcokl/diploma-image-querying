cli = "../db_utils/database.ini"
server = "./db_utils/database.ini"


"""
Config file for PostgreSQL database.
"""


def config():
    db = {
        "host": "localhost",
        "database": "image_querying",
        "user": "postgres",
        "password": "harambe2016!"
    }
    return db
