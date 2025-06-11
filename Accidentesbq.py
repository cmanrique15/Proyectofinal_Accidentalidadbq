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

# Estadísticas Hipótesis 2

# 1. Medidas generales
# Estadísticas generales de EDAD_VICTIMA
print("Media:", df['EDAD_VICTIMA'].mean())
print("Mediana:", df['EDAD_VICTIMA'].median())
print("Moda:", df['EDAD_VICTIMA'].mode()[0])
print("Desviación estándar:", df['EDAD_VICTIMA'].std())
print("Cuartiles:\n", df['EDAD_VICTIMA'].quantile([0.25, 0.5, 0.75]))
# 2. Comparar estadísticas entre muertos y heridos:
df.groupby('GRAVEDAD_ACCIDENTE')['EDAD_VICTIMA'].describe()

# Análisis de distribuciones
# Histograma de edades de las víctimas según gravedad del accidente
import seaborn as sns
import matplotlib.pyplot as plt
sns.histplot(data=df, x='EDAD_VICTIMA', hue='GRAVEDAD_ACCIDENTE', kde=True)
plt.title("Distribución de edades según gravedad")
plt.show()

# ¿A que tipo de distribución se ajustan los datos?
from scipy.stats import shapiro, normaltest

# Solo edades
stat, p = shapiro(df['EDAD_VICTIMA'])
print("Shapiro-Wilk p-value:", p)

#Visualizaciones Hipótesis 2
# 1 Distribución de gravedad
plt.figure()
sns.countplot(data=df, x='GRAVEDAD_ACCIDENTE')
plt.title('Distribución de Gravedad del Accidente')
plt.show()

# 2 Gravedad vs Rango de Edad
plt.figure(figsize=(8,5))
sns.countplot(data=df, x='RANGO_EDAD', hue='GRAVEDAD_ACCIDENTE', palette='Set2')
plt.title("Gravedad del accidente por rango de edad")
plt.xlabel("Rango de edad")
plt.ylabel("Cantidad")
plt.legend(title='Gravedad')
plt.tight_layout()
plt.show()

# 3 Gravedad vs Condición de la víctima
plt.figure(figsize=(10,5))
sns.countplot(data=df, x='CONDICION_VICTIMA', hue='GRAVEDAD_ACCIDENTE', palette='Set1')
plt.title("Gravedad del accidente por tipo de víctima")
plt.xlabel("Condición de la víctima")
plt.ylabel("Cantidad")
plt.xticks(rotation=45)
plt.legend(title='Gravedad')
plt.tight_layout()
plt.show()

# 4 Gravedad según edad y condición
plt.figure()
sns.boxplot(data=df[df['GRAVEDAD_ACCIDENTE'] != 'ileso'], x='CONDICION_VICTIMA', y='EDAD_VICTIMA', hue='GRAVEDAD_ACCIDENTE')
plt.title("Edad y Gravedad según condición")
plt.xticks(rotation=45)
plt.show()

# 5 Mapa de calor
heatmap_data = df.pivot_table(
    values='GRAVEDAD_ACCIDENTE',
    index='CONDICION_VICTIMA',
    columns='RANGO_EDAD',
    aggfunc=lambda x: (x == 'muerto').mean()
)

plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_data, annot=True, cmap='Reds', fmt=".2f")
plt.title('Proporción de fallecidos por condición y rango de edad')
plt.ylabel('Condición de la víctima')
plt.xlabel('Rango de edad')
plt.tight_layout()
plt.show()

# 6 Distribución por sexo y condición
plt.figure()
sns.countplot(data=df, x='SEXO_VICTIMA', hue='CONDICION_VICTIMA')
plt.title('Sexo vs Rol en el Accidente')
plt.show()

# 7 Distribución por tipo de accidente
plt.figure()
sns.countplot(data=df, x='CLASE_ACCIDENTE', order=df['CLASE_ACCIDENTE'].value_counts().index)
plt.title('Tipos de Accidente')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Modelados
# Modelo 1 Random Forest para predecir la gravedad del accidente (modelo de clasificación supervisada)

# Importar librerias necesarias
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#Cargar y limpiar datos,
df['GRAVEDAD_BINARIA'] = df['GRAVEDAD_ACCIDENTE'].apply(lambda x: 1 if x == 'muerto' else 0)

#Variables predictoras (puedes ajustar según tus datos),
variables = ['SEXO_VICTIMA', 'EDAD_VICTIMA', 'CONDICION_VICTIMA', 'CLASE_ACCIDENTE', 'DIA_SEMANA']

#Quitar filas con valores faltantes,
df_modelo = df[variables + ['GRAVEDAD_BINARIA']].dropna()

#convertimos las variables de texto a numero (columna binaria 0 o 1),
df_modelo = pd.get_dummies(df_modelo, columns=['SEXO_VICTIMA', 'CONDICION_VICTIMA', 'CLASE_ACCIDENTE', 'DIA_SEMANA'], drop_first=True)

#Separar X e y,
X = df_modelo.drop('GRAVEDAD_BINARIA', axis=1) #todo menos la columna de gravedad
y = df_modelo['GRAVEDAD_BINARIA']

#Dividir en conjunto de entrenamiento y prueba,
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#entrenamiento del modelo
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

#Hacer predicciones y evaluar el modelo
y_pred = modelo.predict(X_test) 

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nReporte de clasificación:\n", classification_report(y_test, y_pred))

from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_estimator(modelo, X_test, y_test, display_labels=["No Muerto", "Muerto"], cmap="Reds")
plt.title("Matriz de Confusión")
plt.show()
