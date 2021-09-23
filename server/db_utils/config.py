cli = "../db_utils/database.ini"
server = "./db_utils/database.ini"


def config():
    db = {
        "host": "localhost",
        "database": "image_querying",
        "user": "postgres",
        "password": "harambe2016!"
    }
    return db
