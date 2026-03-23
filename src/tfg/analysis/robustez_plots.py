import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('datos_individuales_31_ejecuciones.csv')

sns.set_style("whitegrid")
colors = {'NSGA-II': '#2ecc71', 'SPEA2': '#3498db', 'PACO': '#e74c3c'}

datasets = sorted(df['Dataset'].unique())

for idx, dataset in enumerate(datasets):
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.suptitle(f'Distribución del Hypervolume - {dataset.replace(".json", "")} (31 Ejecuciones)',
                 fontsize=14, fontweight='bold')

    df_dataset = df[df['Dataset'] == dataset]

    sns.boxplot(data=df_dataset, x='Algoritmo', y='Hypervolume',
                palette=colors, ax=ax)
    sns.stripplot(data=df_dataset, x='Algoritmo', y='Hypervolume',
                  color='black', alpha=0.3, size=4, ax=ax)

    ax.set_ylabel('Hypervolume', fontweight='bold', fontsize=12)
    ax.set_xlabel('Algoritmo', fontweight='bold', fontsize=12)
    ax.grid(True, alpha=0.3)

    for i, algo in enumerate(['NSGA-II', 'SPEA2', 'PACO']):
        data = df_dataset[df_dataset['Algoritmo'] == algo]['Hypervolume']
        if len(data) > 0:
            median = data.median()
            mean = data.mean()
            std = data.std()
            ax.text(i, ax.get_ylim()[1] * 0.95,
                    f'μ={mean:.3f}\nσ={std:.3f}',
                    ha='center', va='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    filename = f'boxplot_{dataset.replace(".json", "")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✅ Gráfico guardado: {filename}")

    plt.show()

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

comparison_df.to_csv('comparacion_por_dataset.csv', index=False)
print(f"\n✅ Tabla guardada: comparacion_por_dataset.csv")

print("\n" + "=" * 100)
print("MEJOR ALGORITMO POR DATASET (Mayor HV Medio)")
print("=" * 100)

for dataset in datasets:
    dataset_name = dataset.replace('.json', '')
    dataset_comp = comparison_df[comparison_df['Dataset'] == dataset_name]
    best_algo = dataset_comp.loc[dataset_comp['Media'].idxmax()]
    print(
        f"{dataset_name:10s}: {best_algo['Algoritmo']:10s} (HV={best_algo['Media']:.4f}, CV={best_algo['CV(%)']:.2f}%)")