import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Cargar datos
df = pd.read_csv('datos_individuales_31_ejecuciones.csv')

# Corregir columna Generaciones_Iteraciones para PACO
if 'iterations_completed' in df.columns:
    df.loc[df['Algoritmo'] == 'PACO', 'Generaciones_Iteraciones'] = df.loc[df['Algoritmo'] == 'PACO', 'iterations_completed']
elif 'Iteraciones_Completadas' in df.columns:
    df.loc[df['Algoritmo'] == 'PACO', 'Generaciones_Iteraciones'] = df.loc[df['Algoritmo'] == 'PACO', 'Iteraciones_Completadas']
elif 'Iteraciones' in df.columns:
    df.loc[df['Algoritmo'] == 'PACO', 'Generaciones_Iteraciones'] = df.loc[df['Algoritmo'] == 'PACO', 'Iteraciones']

# Calcular promedios
hypervolume_means = df.groupby(['Dataset', 'Algoritmo'])['Hypervolume'].mean().unstack()

hypervolume_means.plot(kind='bar', color=['#1f77b4', '#ff7f0e', '#2ca02c'], figsize=(10, 6))
plt.title('Hipervolumen Medio por Dataset y Algoritmo')
plt.xlabel('Dataset')
plt.ylabel('Hipervolumen Medio')
plt.legend(title='Algoritmo')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# Calcular tiempos medios
time_means = df.groupby(['Dataset', 'Algoritmo'])['Tiempo'].mean().unstack()

time_means.plot(kind='bar', color=['#1f77b4', '#ff7f0e', '#2ca02c'], figsize=(10, 6))
plt.title('Tiempo de Ejecución Medio por Dataset y Algoritmo')
plt.xlabel('Dataset')
plt.ylabel('Tiempo Medio (segundos)')
plt.legend(title='Algoritmo')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# Calcular generaciones/iteraciones medias
generations_means = df.groupby(['Dataset', 'Algoritmo'])['Generaciones_Iteraciones'].mean().unstack()

# Filtrar solo columnas que tienen datos (no son todo NaN)
generations_means_filtered = generations_means.dropna(axis=1, how='all')
generations_means_filtered.plot(kind='bar', color=['#1f77b4', '#ff7f0e', '#2ca02c'][:len(generations_means_filtered.columns)], figsize=(10, 6))
plt.title('Generaciones/Iteraciones Medias por Dataset y Algoritmo')
plt.xlabel('Dataset')
plt.ylabel('Generaciones/Iteraciones Medias')
plt.legend(title='Algoritmo')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# Calcular tamaño medio del frente de Pareto
pareto_means = df.groupby(['Dataset', 'Algoritmo'])['Tamaño_Pareto'].mean().unstack()

pareto_means.plot(kind='bar', color=['#1f77b4', '#ff7f0e', '#2ca02c'], figsize=(10, 6))
plt.title('Tamaño Medio del Frente de Pareto por Dataset y Algoritmo')
plt.xlabel('Dataset')
plt.ylabel('Tamaño Medio del Frente de Pareto')
plt.legend(title='Algoritmo')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# Convertir a minutos y calcular promedios
minutes_means = df.groupby(['Dataset', 'Algoritmo'])['Minutos_Medios'].mean().unstack()

minutes_means.plot(kind='bar', color=['#1f77b4', '#ff7f0e', '#2ca02c'], figsize=(10, 6))
plt.title('Minutos medios por Dataset y Algoritmo')
plt.xlabel('Dataset')
plt.ylabel('Minutos medios')
plt.legend(title='Algoritmo')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()


cost_means = df.groupby(['Dataset', 'Algoritmo'])['Costo_Medio'].mean().unstack()

cost_means.plot(kind='bar', color=['#1f77b4', '#ff7f0e', '#2ca02c'], figsize=(10, 6))
plt.title('Costo Medio por Dataset y Algoritmo')
plt.xlabel('Dataset')
plt.ylabel('Costo Medio')
plt.legend(title='Algoritmo')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()



# Gráfica estilo la imagen de referencia
fig, ax = plt.subplots(figsize=(12, 6))

datasets = ['users1.json', 'users2.json', 'users3.json', 'users4.json', 'users5.json']
algoritmos = ['NSGA-II', 'SPEA2', 'PACO']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

x = np.arange(len(datasets))
width = 0.25

for i, algoritmo in enumerate(algoritmos):
    means = [df[(df['Dataset'] == dataset) & (df['Algoritmo'] == algoritmo)]['Hypervolume'].mean()
             for dataset in datasets]
    ax.bar(x + i*width, means, width, label=algoritmo, color=colors[i])

ax.set_xlabel('Dataset')
ax.set_ylabel('Hipervolumen Medio')
ax.set_title('Comparación de Hipervolumen por Dataset y Algoritmo')
ax.set_xticks(x + width)
ax.set_xticklabels(datasets, rotation=45)
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()