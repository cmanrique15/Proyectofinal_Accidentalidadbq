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

#Antes de eliminar
print(f"\n Duplicados antes de limpiar: {df.duplicated().sum()}")
print(df.duplicated())

# Eliminar duplicados
df = df.drop_duplicates()
print(f"\n Duplicados después de limpiar: {df.duplicated().sum()}")

print("Valores nulos antes de limpieza:\n",df.isnull().sum())

df['EDAD_VICTIMA'] = pd.to_numeric(df['EDAD_VICTIMA'], errors='coerce')
df['CANTIDAD_VICTIMA'] = pd.to_numeric(df['CANTIDAD_VICTIMA'], errors='coerce')

# Eliminar registros con valores cero o nulos importantes
df = df.dropna()
df = df[df['CANTIDAD_VICTIMA'] > 0]
df = df[(df['EDAD_VICTIMA'] > 0) & (df['EDAD_VICTIMA'] <= 100)]
df['SEXO_VICTIMA'] = df['SEXO_VICTIMA'].str.strip()
df = df[df['SEXO_VICTIMA'] != '']
df = df.dropna(subset=['SEXO_VICTIMA'])

print("Valores nulos después de limpieza:\n",df.isnull().sum())
print(f"Filas después de limpiar: {len(df)}")

# Nueva columna: Día de la semana
df['DIA_SEMANA'] = df['FECHA'].dt.day_name()

# Nueva columna: Rango de edad
bins = [0, 17, 30, 45, 60, 200]
labels = ['<18', '18-30', '31-45', '46-60', '60+']
df['RANGO_EDAD'] = pd.cut(df['EDAD_VICTIMA'], bins=bins, labels=labels)

#Guardar base de datos limpia con los cambios aplicados
df.to_csv('accidentesbq_limpio.csv', index=False)
