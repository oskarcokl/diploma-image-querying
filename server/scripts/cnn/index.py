import psycopg2

connection = psycopg2.connect(filename="database.ini", section="postgresql")
