import json
import pandas as pd
import os
import numpy as np
from inspyred.ec.analysis import hypervolume


def load_global_reference_points(
        stats_file=r"C:\Users\hctr0\PycharmProjects\TFG_Hector\pareto_outputs\pareto_stats_summary.csv"):
    """
    Carga los puntos de referencia globales desde el archivo de estadísticas
    """
    try:
        stats_df = pd.read_csv(stats_file)
        print(f"Cargando puntos de referencia globales desde: {stats_file}")

        global_stats = {}
        for _, row in stats_df.iterrows():
            dataset = row['dataset']
            global_stats[dataset] = {
                'min_cost': row['min_cost'],
                'max_cost': row['max_cost'],
                'min_minutes': row['min_minutes'],
                'max_minutes': row['max_minutes']
            }
            print(f"  {dataset}: Cost[{row['min_cost']:.2f}, {row['max_cost']:.2f}], "
                  f"Minutes[{row['min_minutes']:.0f}, {row['max_minutes']:.0f}]")

        return global_stats

    except Exception as e:
        print(f"ERROR: No se pudo cargar {stats_file}: {e}")
        return None


def calculate_hypervolume_for_run(pareto_points, global_stats, dataset_name, algorithm_name, debug=False):
    """
    Calcula el hipervolumen para una ejecución individual
    """
    if not pareto_points or dataset_name not in global_stats:
        return 0.0

    stats = global_stats[dataset_name]
    global_min_cost = stats['min_cost']
    global_max_cost = stats['max_cost']
    global_min_minutes = stats['min_minutes']
    global_max_minutes = stats['max_minutes']

    normalized_points = []

    print(f"\n*** DEBUG NORMALIZACIÓN - {dataset_name} - {algorithm_name} ***")
    print(f"Rangos globales: Coste[{global_min_cost:.2f}, {global_max_cost:.2f}], "
          f"Minutos[{global_min_minutes:.0f}, {global_max_minutes:.0f}]")

    for i, (neg_minutes, cost) in enumerate(pareto_points):
        if algorithm_name != "PACO":
            minutes = -neg_minutes
        else:
            minutes = neg_minutes

        # NORMALIZACIÓN PARA MAXIMIZACIÓN
        cost_range = global_max_cost - global_min_cost
        minutes_range = global_max_minutes - global_min_minutes

        if cost_range == 0:
            cost_maximized = 1.0
        else:
            cost_maximized = (global_max_cost - cost) / cost_range

        if minutes_range == 0:
            minutes_maximized = 1.0
        else:
            minutes_maximized = (minutes - global_min_minutes) / minutes_range

        # Validar que estén en [0,1]
        cost_maximized = max(0.0, min(1.0, cost_maximized))
        minutes_maximized = max(0.0, min(1.0, minutes_maximized))

        normalized_points.append([cost_maximized, minutes_maximized])

        # DEBUG: Mostrar cada punto normalizado
        print(f"Punto {i + 1}: Original (coste={cost:.2f}, minutos={minutes:.2f}) -> "
              f"Normalizado (coste={cost_maximized:.4f}, minutos={minutes_maximized:.4f})")

    try:
        if len(normalized_points) > 0:
            ref_point = [0.0, 0.0]
            hv = hypervolume(normalized_points, reference_point=ref_point)
            print(f"Hipervolumen calculado: {hv:.4f}")
            return hv
        else:
            return 0.0
    except Exception as e:
        print(f"Error calculando HV: {e}")
        return 0.0


def calculate_pareto_statistics(pareto_points, algorithm_name):
    """
    Calcula estadísticas del frente de Pareto (costo medio y minutos medios)
    """
    if not pareto_points:
        return None, None

    costs = []
    minutes_list = []

    for neg_minutes, cost in pareto_points:
        # Convertir minutos según el algoritmo
        if algorithm_name != "PACO":
            minutes = -neg_minutes  # NSGA-II y SPEA2 tienen minutos negativos
        else:
            minutes = neg_minutes  # PACO tiene minutos positivos

        costs.append(cost)
        minutes_list.append(minutes)

    avg_cost = np.mean(costs) if costs else 0.0
    avg_minutes = np.mean(minutes_list) if minutes_list else 0.0

    return avg_cost, avg_minutes


def extract_individual_runs(base_path, algorithm_name, datasets, global_stats):
    """
    Extrae los datos individuales de cada ejecución con hipervolumen calculado
    """
    all_data = []

    print(f"\n{'=' * 80}")
    print(f"EXTRAYENDO DATOS INDIVIDUALES - {algorithm_name}")
    print(f"{'=' * 80}")

    for folder, dataset_file in datasets:
        file_path = os.path.join(base_path, folder, f"summary_complete_{folder}.json")

        print(f"\nProcesando: {dataset_file}")
        print(f"Archivo: {file_path}")

        if not os.path.exists(file_path):
            print(f"ADVERTENCIA: El archivo no existe. Saltando...")
            continue

        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"ERROR al cargar {file_path}: {e}")
            continue

        # Contador de ejecuciones por configuración
        execution_counter = {}

        for entry in data:
            if entry["dataset"] == dataset_file:
                config = entry["config"]

                # Incrementar contador de ejecución para esta configuración
                if config not in execution_counter:
                    execution_counter[config] = 0
                execution_counter[config] += 1

                # Calcular hipervolumen para esta ejecución individual
                pareto_points = entry.get('pareto_points', [])
                hv = calculate_hypervolume_for_run(pareto_points, global_stats, dataset_file, algorithm_name)

                # Calcular estadísticas del frente de Pareto
                avg_cost, avg_minutes = calculate_pareto_statistics(pareto_points, algorithm_name)

                # Extraer datos de esta ejecución individual
                run_data = {
                    'Dataset': dataset_file,
                    'Algoritmo': algorithm_name,
                    'Configuracion': config,
                    'Ejecucion': execution_counter[config],
                    'Hypervolume': hv,
                    'Tiempo': entry.get('execution_time', 0),
                    'Generaciones_Iteraciones': entry.get('generations', entry.get('iterations_completed',
                                                                                   entry.get('iterations', 0))),
                    'Tamaño_Pareto': entry.get('pareto_size', 0),
                    'Num_Puntos_Pareto': len(pareto_points),
                    'Costo_Medio': avg_cost if avg_cost is not None else 0.0,
                    'Minutos_Medios': avg_minutes if avg_minutes is not None else 0.0
                }

                all_data.append(run_data)

        # Mostrar resumen de lo extraído
        print(f"  Configuraciones encontradas: {len(execution_counter)}")
        for config, count in execution_counter.items():
            print(f"     - {config}: {count} ejecuciones")

    return all_data


def main():
    """
    Script principal para extraer datos individuales de múltiples algoritmos
    """

    # Cargar puntos de referencia globales PRIMERO
    print("=" * 80)
    print("CARGANDO PUNTOS DE REFERENCIA GLOBALES")
    print("=" * 80)
    global_stats = load_global_reference_points()

    if global_stats is None:
        print("ERROR: No se pudieron cargar los puntos de referencia globales. Terminando...")
        return

    # Datasets a procesar
    datasets = [
        ("users1", "users1.json"),
        ("users2", "users2.json"),
        ("users3", "users3.json"),
        ("users4", "users4.json"),
        ("users5", "users5.json")
    ]

    # Configuración de algoritmos (ajusta las rutas según tu estructura)
    algorithms = [
        {
            'name': 'NSGA-II',
            'base_path': r"C:\Users\hctr0\PycharmProjects\TFG_Hector\31Executions\NSGA2\summaries"
        },
        {
            'name': 'SPEA2',
            'base_path': r"C:\Users\hctr0\PycharmProjects\TFG_Hector\31Executions\SPEA2\summaries"
        },
        {
            'name': 'PACO',
            'base_path': r"C:\Users\hctr0\PycharmProjects\TFG_Hector\31Executions\PACO\summaries"
        }
    ]

    all_runs = []

    # Extraer datos de cada algoritmo
    for algo in algorithms:
        print(f"\n{'#' * 80}")
        print(f"PROCESANDO ALGORITMO: {algo['name']}")
        print(f"{'#' * 80}")

        runs = extract_individual_runs(algo['base_path'], algo['name'], datasets, global_stats)
        all_runs.extend(runs)

    # Crear DataFrame con todos los datos
    if not all_runs:
        print("\nNo se extrajeron datos de ningún algoritmo")
        return

    df = pd.DataFrame(all_runs)

    # Mostrar resumen
    print(f"\n{'=' * 80}")
    print("RESUMEN DE DATOS EXTRAÍDOS")
    print(f"{'=' * 80}")
    print(f"Total de ejecuciones individuales: {len(df)}")
    print(f"\nPor algoritmo:")
    print(df.groupby('Algoritmo').size())
    print(f"\nPor dataset:")
    print(df.groupby('Dataset').size())
    print(f"\nPor algoritmo y dataset:")
    print(df.groupby(['Algoritmo', 'Dataset']).size())

    # Guardar CSV
    output_file = "../statisticalAnalysis/datos_individuales_31_ejecuciones.csv"
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\nDatos guardados en: {output_file}")

    # Mostrar primeras filas
    print(f"\n{'=' * 80}")
    print("PRIMERAS 10 FILAS DEL CSV:")
    print(f"{'=' * 80}")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(df.head(600).to_string())

    # Estadísticas básicas por algoritmo
    print(f"\n{'=' * 80}")
    print("ESTADÍSTICAS POR ALGORITMO:")
    print(f"{'=' * 80}")

    for algo in df['Algoritmo'].unique():
        algo_data = df[df['Algoritmo'] == algo]
        print(f"\n{algo}:")
        print(f"  Ejecuciones: {len(algo_data)}")
        print(f"  Hypervolume - Media: {algo_data['Hypervolume'].mean():.4f}, "
              f"Std: {algo_data['Hypervolume'].std():.4f}, "
              f"Min: {algo_data['Hypervolume'].min():.4f}, "
              f"Max: {algo_data['Hypervolume'].max():.4f}")
        print(f"  Tiempo - Media: {algo_data['Tiempo'].mean():.2f}s, "
              f"Std: {algo_data['Tiempo'].std():.2f}s, "
              f"Min: {algo_data['Tiempo'].min():.2f}s, "
              f"Max: {algo_data['Tiempo'].max():.2f}s")
        print(f"  Generaciones/Iteraciones - Media: {algo_data['Generaciones_Iteraciones'].mean():.2f}, "
              f"Std: {algo_data['Generaciones_Iteraciones'].std():.2f}")
        print(f"  Tamaño Pareto - Media: {algo_data['Tamaño_Pareto'].mean():.2f}, "
              f"Std: {algo_data['Tamaño_Pareto'].std():.2f}")
        print(f"  Costo Medio - Media: {algo_data['Costo_Medio'].mean():.2f}, "
              f"Std: {algo_data['Costo_Medio'].std():.2f}")
        print(f"  Minutos Medios - Media: {algo_data['Minutos_Medios'].mean():.2f}, "
              f"Std: {algo_data['Minutos_Medios'].std():.2f}")

    print(f"\n{'=' * 80}")
    print("LISTO - Ahora puedes usar este CSV para crear boxplots")
    print(f"El hipervolumen ha sido calculado para CADA ejecución individual")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()