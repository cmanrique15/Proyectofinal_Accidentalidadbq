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

# Se importan las librerias necesarias
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

# 1) se realiza la exploración y limpieza de datos
#creación Dataframe con pandas

columnas = ['FECHA_ACCIDENTE', 'DIRECCION_ACCIDENTE', 'CONDICION_VICTIMA', 
            'GRAVEDAD_ACCIDENTE', 'CLASE_ACCIDENTE', 'SEXO_VICTIMA', 'EDAD_VICTIMA','CANTIDAD_VICTIMAS']

df = pd.DataFrame(resultados, columns=columnas)

print(df)
print(df.shape)
print(df.dtypes)

df[['FECHA', 'HORA']] = df['FECHA_ACCIDENTE'].str.split(' ', n=1, expand=True)
df['FECHA'] = pd.to_datetime(df['FECHA'], format= '%m/%d/%Y', errors='coerce')

# Eliminar la columna 'HORA' si ya no es necesaria
df.drop('HORA', axis=1, inplace=True)    
df.drop('FECHA_ACCIDENTE', axis=1, inplace=True)

df =df.astype (
    {
    'DIRECCION_ACCIDENTE': 'string',
    'CONDICION_VICTIMA':'string',
    'GRAVEDAD_ACCIDENTE':'string',
    'CLASE_ACCIDENTE':'string',
    'SEXO_VICTIMA':'string',
    
})

df['EDAD_VICTIMA'] = pd.to_numeric(df['EDAD_VICTIMA'], errors='coerce')
df['CANTIDAD_VICTIMAS'] = pd.to_numeric(df['CANTIDAD_VICTIMAS'], errors='coerce')

print(df.info())

