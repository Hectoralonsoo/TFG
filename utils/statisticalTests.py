import pandas as pd
import numpy as np
from statds.no_parametrics import friedman, shaffer

# Configurar semilla para reproducibilidad
np.random.seed(42)

# Generar datos sintéticos para 31 ejecuciones
n_runs = 31

# Generar datos con diferentes distribuciones para cada algoritmo
# NSGA-II: mejor rendimiento (media más alta)
nsga2_data = np.random.normal(0.85, 0.08, n_runs)
nsga2_data = np.clip(nsga2_data, 0.65, 0.95)  # Limitar rango

# SPEA2: rendimiento intermedio
spea2_data = np.random.normal(0.78, 0.07, n_runs)
spea2_data = np.clip(spea2_data, 0.60, 0.90)

# PACO: rendimiento menor
paco_data = np.random.normal(0.73, 0.06, n_runs)
paco_data = np.clip(paco_data, 0.55, 0.85)

# Crear DataFrame con la estructura correcta para Friedman
data = {
    'Case': range(1, n_runs + 1),
    'NSGA-II': nsga2_data,
    'SPEA2': spea2_data,
    'PACO': paco_data
}

df = pd.DataFrame(data)

# Redondear a 4 decimales para que se vea más limpio
df['NSGA-II'] = df['NSGA-II'].round(4)
df['SPEA2'] = df['SPEA2'].round(4)
df['PACO'] = df['PACO'].round(4)

print("Dataset generado (primeras 10 filas):")
print(df)
print(f"\nShape: {df.shape}")

# Guardar el CSV
df.to_csv("resultados_algoritmos_test.csv", index=False)
print("\nCSV guardado como 'resultados_algoritmos_test.csv'")

# Mostrar estadísticas descriptivas
print("\n" + "=" * 60)
print("ESTADÍSTICAS DESCRIPTIVAS")
print("=" * 60)

for col in ['NSGA-II', 'SPEA2', 'PACO']:
    data_col = df[col]
    print(f"\n{col}:")
    print(f"  Media: {data_col.mean():.4f}")
    print(f"  Mediana: {data_col.median():.4f}")
    print(f"  Desv. Std: {data_col.std():.4f}")
    print(f"  Min: {data_col.min():.4f}")
    print(f"  Max: {data_col.max():.4f}")

# Ejecutar pruebas estadísticas
print("\n" + "=" * 60)
print("PRUEBAS ESTADÍSTICAS")
print("=" * 60)

try:
    # Test de Friedman
    rankings, statistic, p_value, critical_value, hypothesis = friedman(
        df, alpha=0.05, minimize=False, verbose=False
    )

    print("FRIEDMAN TEST RESULTS:")
    print(f"Hipótesis: {hypothesis}")
    print(f"Estadístico: {statistic:.4f}")
    print(f"p-valor: {p_value:.2e}" if p_value is not None else "p-valor: None")
    print(f"Valor crítico: {critical_value:.4f}" if critical_value is not None else "Valor crítico: None")
    print(f"Rankings promedio: {rankings}")

    # Post-hoc si es significativo
    if p_value is not None and p_value < 0.05:
        print(f"\n¡Diferencias significativas detectadas! (p < 0.05)")
        print("Ejecutando análisis post-hoc...")

        try:
            result = shaffer(rankings, num_cases=len(df), alpha=0.05)

            if isinstance(result, tuple):
                if len(result) == 2:
                    ranks_values, figure = result
                    print(f"Post-hoc results: {ranks_values}")
                    print("Figura generada para visualización")
                    if hasattr(figure, 'show'):
                        print("Llamando figure.show()...")
                        figure.show()
                        figure.savefig("shaffer_diagram.png", dpi=300, bbox_inches="tight")

                elif len(result) == 3:
                    ranks_values, critical_distance, figure = result
                    print(f"Post-hoc results: {ranks_values}")
                    print(f"Distancia crítica: {critical_distance}")
                    if hasattr(figure, 'show'):
                        print("Llamando figure.show()...")
                        figure.show()
                        figure.savefig("shaffer_diagram.png", dpi=300, bbox_inches="tight")

                else:
                    print(f"Shaffer devolvió {len(result)} valores: {result}")
            else:
                print(f"Shaffer result: {result}")
                if hasattr(result, 'show'):
                    result.show()

        except Exception as e:
            print(f"Error en análisis post-hoc: {e}")
            print(f"Tipo de error: {type(e).__name__}")

    elif p_value is not None:
        print(f"\nNo hay diferencias significativas (p = {p_value:.4f} >= 0.05)")
    else:
        print(f"\nUsando distribución Q. Estadístico: {statistic:.4f}, Valor crítico: {critical_value:.4f}")
        if statistic > critical_value:
            print("Diferencias significativas detectadas!")
        else:
            print("No hay diferencias significativas")

except Exception as e:
    print(f"Error ejecutando pruebas estadísticas: {e}")
    print(f"Tipo de error: {type(e).__name__}")

print("\n" + "=" * 60)
print("DATASET COMPLETO")
print("=" * 60)
print(df.to_string(index=False))

print(f"\nArchivo CSV creado: 'resultados_algoritmos_test.csv'")
print("Listo para usar con tus pruebas estadísticas!")