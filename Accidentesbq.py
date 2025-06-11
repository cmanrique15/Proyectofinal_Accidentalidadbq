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

# Filtrar solo los años entre 2021 y 2025
df = df[(df['FECHA'].dt.year >= 2021) & (df['FECHA'].dt.year <= 2025)]
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

# Hipótesis 1: Zonas con mayor cantidad de víctimas fatales Análisis de datos
print("\nConteo de muertes por tipo de accidente")
muertes_por_clase = df[df['GRAVEDAD_ACCIDENTE'] == 'muerto']['CLASE_ACCIDENTE'].value_counts()
print(muertes_por_clase)

print("\nTasa de mortalidad por tipo de accidente - compara cuántas muertes hay en proporción al total de accidentes de cada tipo")
# Total de accidentes por tipo
total_por_clase = df['CLASE_ACCIDENTE'].value_counts()
# Muertes por tipo
muertes_por_clase = df[df['GRAVEDAD_ACCIDENTE'] == 'muerto']['CLASE_ACCIDENTE'].value_counts()
# Tasa de mortalidad
tasa_mortalidad = (muertes_por_clase / total_por_clase).sort_values(ascending=False)
print(tasa_mortalidad)

# Asegurar formato uniforme
df['GRAVEDAD_ACCIDENTE'] = df['GRAVEDAD_ACCIDENTE'].str.strip().str.lower()  # herido / muerto
df['DIRECCION ACCIDENTE'] = df['DIRECCION ACCIDENTE'].str.strip().str.upper()
# Agrupar por dirección y tipo de gravedad (muerto/herido)
zonas = df.groupby(['DIRECCION ACCIDENTE', 'GRAVEDAD_ACCIDENTE']).size().unstack(fill_value=0)

# Asegurar que las columnas 'muerto' y 'herido' existen (por si no hay datos)
for col in ['muerto', 'herido']:
    if col not in zonas.columns:
        zonas[col] = 0

print("\n Total y tasa de mortalidad:")
zonas['TOTAL_ACCIDENTES'] = zonas['muerto'] + zonas['herido'] #TOTAL_ACCIDENTES: suma de muertos y heridos por dirección
zonas['TASA_MORTALIDAD_%'] = (zonas['muerto'] / zonas['TOTAL_ACCIDENTES']) * 100
zonas['TASA_MORTALIDAD_%'] = zonas['TASA_MORTALIDAD_%'].round(2)

# Top 10 direcciones con más muertes
top_muertos = zonas.sort_values('muerto', ascending=False).head(10)
print("\n Zonas con más víctimas fatales:")
print(top_muertos[['muerto', 'TOTAL_ACCIDENTES']])

# Top 10 direcciones con más accidentes (totales)
top_accidentes = zonas.sort_values('TOTAL_ACCIDENTES', ascending=False).head(10)
print("\n Zonas con más accidentes totales:")
print(top_accidentes[['TOTAL_ACCIDENTES', 'muerto']])

# Top zonas con mayor tasa de mortalidad (%)
top_tasa = zonas[zonas['TOTAL_ACCIDENTES'] >= 5].sort_values('TASA_MORTALIDAD_%', ascending=False).head(10)
print("\n Zonas con mayor tasa de mortalidad (%):")
print(top_tasa[['TASA_MORTALIDAD_%', 'TOTAL_ACCIDENTES']])

# Visualización Hipotesis 1 
# Visualización para Top 10 zonas con más muertes:
plt.figure(figsize=(12, 6))
bars = plt.barh(top_muertos.index, top_muertos['muerto'], color='pink')
plt.xlabel('Cantidad de fallecidos')
plt.title('Zonas con mayor cantidad de víctimas fatales')
plt.gca().invert_yaxis()
plt.tight_layout()

# Etiquetas numéricas al final de cada barra
for bar in bars:
    width = bar.get_width()
    plt.text(width + 1, bar.get_y() + bar.get_height()/2,
             str(int(width)), va='center')

plt.show()

# Visualización para la cantidad total de accidentes:
plt.figure(figsize=(12, 6))
bars_total = plt.barh(top_accidentes.index, top_accidentes['TOTAL_ACCIDENTES'], color='skyblue')
plt.xlabel('Cantidad total de accidentes')
plt.title('Zonas con mayor cantidad total de accidentes')
plt.gca().invert_yaxis()
plt.tight_layout()

# Etiquetas numéricas
for bar in bars_total:
    width = bar.get_width()
    plt.text(width + 1, bar.get_y() + bar.get_height()/2,
             str(int(width)), va='center')

plt.show()

# Visualización de la tasa de mortalidad
plt.figure(figsize=(12, 6))
bars_tasa = plt.barh(top_tasa.index, top_tasa['TASA_MORTALIDAD_%'], color='orange')
plt.xlabel('Tasa de mortalidad (%)')
plt.title('Zonas con mayor tasa de mortalidad (con al menos 5 accidentes)')
plt.gca().invert_yaxis()
plt.tight_layout()

# Etiquetas con porcentaje
for bar in bars_tasa:
    width = bar.get_width()
    plt.text(width + 0.5, bar.get_y() + bar.get_height()/2, f'{width:.2f}%', va='center')

plt.show()

# Calculos para Hipotesis 2
# Asegurar formato
df['GRAVEDAD_ACCIDENTE'] = df['GRAVEDAD_ACCIDENTE'].str.strip().str.lower()
df['CONDICION_VICTIMA'] = df['CONDICION_VICTIMA'].str.strip().str.lower()

# Crear subconjuntos para los grupos clave
moto_joven = df[(df['CONDICION_VICTIMA'] == 'motociclista') & (df['EDAD_VICTIMA'] >= 18) & (df['EDAD_VICTIMA'] <= 30)]
peaton_mayor = df[(df['CONDICION_VICTIMA'] == 'peaton') & (df['EDAD_VICTIMA'] > 60)]

# Calcular tasas de mortalidad
def calcular_tasa(df_grupo, nombre):
    total = len(df_grupo)
    muertos = len(df_grupo[df_grupo['GRAVEDAD_ACCIDENTE'] == 'muerto'])
    tasa = (muertos / total) * 100 if total > 0 else 0
    print(f"{nombre} - Total: {total}, Muertos: {muertos}, Tasa de mortalidad: {tasa:.2f}%")
        
# Código para comparación con la tasa general de mortalidad:
total = len(df)
muertos = len(df[df['GRAVEDAD_ACCIDENTE'] == 'muerto'])
tasa_general = (muertos / total) * 100
print(f"\nTasa general de mortalidad en el dataset: {tasa_general:.2f}%")

# Tasa general de mortalidad por condicion_victima:
# Calcular tasa de mortalidad por rol
roles = df.groupby('CONDICION_VICTIMA')['GRAVEDAD_ACCIDENTE'].value_counts().unstack(fill_value=0)
roles['TASA_MORTALIDAD_%'] = (roles['muerto'] / (roles['muerto'] + roles['herido'])) * 100
roles = roles.sort_values('TASA_MORTALIDAD_%', ascending=False)
print(" Tasa de mortalidad por rol de víctima:\n")
print(roles[['muerto', 'herido', 'TASA_MORTALIDAD_%']])

# Tasa de mortalidad por Rango_Edad:
# Calcular tasa de mortalidad por rango de edad
edades = df.groupby('RANGO_EDAD')['GRAVEDAD_ACCIDENTE'].value_counts().unstack(fill_value=0)
edades['TASA_MORTALIDAD_%'] = (edades['muerto'] / (edades['muerto'] + edades['herido'])) * 100
edades = edades.sort_values('TASA_MORTALIDAD_%', ascending=False)
print("\n Tasa de mortalidad por rango de edad:\n")
print(edades[['muerto', 'herido', 'TASA_MORTALIDAD_%']])

# Mostrar resultados
print("Tasa de mortalidad por grupo:\n")
calcular_tasa(moto_joven, "Motociclistas jóvenes (18-30 años)")
calcular_tasa(peaton_mayor, "Peatones mayores de 60 años")
