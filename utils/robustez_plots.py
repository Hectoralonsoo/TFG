import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar datos
df = pd.read_csv('datos_individuales_31_ejecuciones.csv')

# Configurar estilo
sns.set_style("whitegrid")
colors = {'NSGA-II': '#2ecc71', 'SPEA2': '#3498db', 'PACO': '#e74c3c'}

# Crear figura con 5 subplots (uno por dataset)
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('Distribución del Hypervolume por Dataset (31 Ejecuciones)',
             fontsize=16, fontweight='bold')

datasets = sorted(df['Dataset'].unique())

for idx, dataset in enumerate(datasets):
    row = idx // 3
    col = idx % 3
    ax = axes[row, col]

    # Filtrar datos del dataset
    df_dataset = df[df['Dataset'] == dataset]

    # Crear boxplot
    sns.boxplot(data=df_dataset, x='Algoritmo', y='Hypervolume',
                palette=colors, ax=ax)
    sns.stripplot(data=df_dataset, x='Algoritmo', y='Hypervolume',
                  color='black', alpha=0.3, size=3, ax=ax)

    # Configurar subplot
    ax.set_title(f'{dataset.replace(".json", "")}', fontweight='bold', fontsize=12)
    ax.set_ylabel('Hypervolume' if col == 0 else '', fontweight='bold')
    ax.set_xlabel('Algoritmo', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Añadir estadísticas
    for i, algo in enumerate(['NSGA-II', 'SPEA2', 'PACO']):
        data = df_dataset[df_dataset['Algoritmo'] == algo]['Hypervolume']
        if len(data) > 0:
            median = data.median()
            mean = data.mean()
            std = data.std()
            ax.text(i, ax.get_ylim()[1] * 0.95,
                    f'μ={mean:.3f}\nσ={std:.3f}',
                    ha='center', va='top', fontsize=7,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Ocultar subplot vacío
axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig('boxplots_por_dataset.png', dpi=300, bbox_inches='tight')
print("✅ Gráfico guardado: boxplots_por_dataset.png")

# ============================================================================
# TABLA COMPARATIVA POR DATASET
# ============================================================================
print("\n" + "=" * 100)
print("COMPARACIÓN DE HYPERVOLUME POR DATASET")
print("=" * 100)

comparison_data = []
for dataset in datasets:
    print(f"\n{dataset.replace('.json', '')}:")
    print("-" * 100)
    for algo in ['NSGA-II', 'SPEA2', 'PACO']:
        data = df[(df['Dataset'] == dataset) & (df['Algoritmo'] == algo)]['Hypervolume']
        if len(data) > 0:
            comparison_data.append({
                'Dataset': dataset.replace('.json', ''),
                'Algoritmo': algo,
                'Media': data.mean(),
                'Mediana': data.median(),
                'Desv.Est': data.std(),
                'Min': data.min(),
                'Max': data.max(),
                'CV(%)': (data.std() / data.mean() * 100)
            })
            print(f"  {algo:10s} - Media: {data.mean():.4f}, Mediana: {data.median():.4f}, "
                  f"Std: {data.std():.4f}, CV: {(data.std() / data.mean() * 100):.2f}%")

comparison_df = pd.DataFrame(comparison_data)

# Guardar tabla
comparison_df.to_csv('comparacion_por_dataset.csv', index=False)
print(f"\n✅ Tabla guardada: comparacion_por_dataset.csv")

# Mejores algoritmos por dataset
print("\n" + "=" * 100)
print("MEJOR ALGORITMO POR DATASET (Mayor HV Medio)")
print("=" * 100)

for dataset in datasets:
    dataset_name = dataset.replace('.json', '')
    dataset_comp = comparison_df[comparison_df['Dataset'] == dataset_name]
    best_algo = dataset_comp.loc[dataset_comp['Media'].idxmax()]
    print(
        f"{dataset_name:10s}: {best_algo['Algoritmo']:10s} (HV={best_algo['Media']:.4f}, CV={best_algo['CV(%)']:.2f}%)")

plt.show()