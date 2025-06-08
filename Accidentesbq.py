# Conexion a una base de datos MySQL
import pymysql

conn = pymysql.connect(
        host="localhost",
        user="root",
        password="Mysql123#",
        database="Accidentalidadbq"
    )

cursor = conn.cursor()
cursor.execute("SELECT * FROM accidentebq")
resultados = cursor.fetchall()
