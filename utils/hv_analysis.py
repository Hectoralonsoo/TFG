import json
import numpy as np
from inspyred.ec.analysis import hypervolume
import pandas as pd
import os
from pathlib import Path


def load_global_reference_points(
        stats_file="C:\\Users\\hctr0\\PycharmProjects\\TFG_Hector\\pareto_outputs\\pareto_stats_summary.csv"):
    """
    Carga los puntos de referencia globales desde el archivo de estadísticas
    """
    try:
        stats_df = pd.read_csv(stats_file)
        print(f"Cargando puntos de referencia globales desde: {stats_file}")
        print(f"Datasets encontrados: {list(stats_df['dataset'].values)}")

        # Crear diccionario con estadísticas por dataset
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


def calculate_hypervolume_for_dataset_with_global_ref(file_path, dataset_name, global_stats):
    """
    Calcula el hipervolumen para un dataset específico usando puntos de referencia globales
    CORREGIDO: Maneja correctamente la maximización de ambos objetivos para inspyred
    """
    print(f"\n{'=' * 60}")
    print(f"PROCESANDO: {dataset_name}")
    print(f"Archivo: {file_path}")
    print(f"{'=' * 60}")

    # Verificar que el archivo existe
    if not os.path.exists(file_path):
        print(f"ADVERTENCIA: El archivo {file_path} no existe. Saltando...")
        return None

    # Verificar que tenemos estadísticas globales para este dataset
    if dataset_name not in global_stats:
        print(f"ADVERTENCIA: No se encontraron estadísticas globales para {dataset_name}")
        return None

    # Obtener estadísticas globales para este dataset
    stats = global_stats[dataset_name]
    global_min_cost = stats['min_cost']
    global_max_cost = stats['max_cost']
    global_min_minutes = stats['min_minutes']
    global_max_minutes = stats['max_minutes']

    print(f"Usando puntos de referencia globales:")
    print(f"  Cost: [{global_min_cost:.2f}, {global_max_cost:.2f}]")
    print(f"  Minutes: [{global_min_minutes:.0f}, {global_max_minutes:.0f}]")

    # Cargar el archivo
    try:
        with open(file_path) as f:
            data = json.load(f)
    except Exception as e:
        print(f"ERROR: No se pudo cargar {file_path}: {e}")
        return None

    config_points = {}
    config_metrics = {}
    total_points = 0

    # Agrupar puntos por configuración
    for entry in data:
        if entry["dataset"] == dataset_name:
            config = entry["config"]

            # Inicializar estructuras de datos para esta configuración
            if config not in config_points:
                config_points[config] = []
                config_metrics[config] = {
                    'execution_times': [],
                    'generations': [],
                    'pareto_sizes': []
                }

            # Recopilar métricas de ejecución
            config_metrics[config]['execution_times'].append(entry.get('execution_time', 0))
            config_metrics[config]['generations'].append(entry.get('generations', 0))
            config_metrics[config]['pareto_sizes'].append(entry.get('pareto_size', 0))

            # Procesar puntos de Pareto con normalización CORRECTA para maximización
            for neg_minutes, cost in entry["pareto_points"]:
                minutes = -neg_minutes

                # TRANSFORMACIÓN PARA MAXIMIZACIÓN:
                # Objetivo 1: Maximizar eficiencia (minimizar costo) -> (max_cost - cost) / (max_cost - min_cost)
                # Objetivo 2: Maximizar velocidad (maximizar minutos) -> (minutes - min_minutes) / (max_minutes - min_minutes)

                # Normalización para maximización (valores altos = mejor)
                cost_maximized = (global_max_cost - cost) / (global_max_cost - global_min_cost)
                minutes_maximized = (minutes - global_min_minutes) / (global_max_minutes - global_min_minutes)

                # VALIDACIÓN: Asegurar que los valores estén en [0,1]
                if cost_maximized < 0 or cost_maximized > 1:
                    print(f"  ADVERTENCIA: cost_maximized fuera de rango: {cost_maximized:.6f} (cost={cost:.2f})")
                if minutes_maximized < 0 or minutes_maximized > 1:
                    print(
                        f"  ADVERTENCIA: minutes_maximized fuera de rango: {minutes_maximized:.6f} (minutes={minutes:.0f})")

                cost_maximized = max(0, min(1, cost_maximized))
                minutes_maximized = max(0, min(1, minutes_maximized))

                # Ahora ambos objetivos están en [0,1] donde 1 es mejor (maximización)
                config_points[config].append([cost_maximized, minutes_maximized])
                total_points += 1

    print(f"Total puntos procesados: {total_points}")
    print(f"Configuraciones encontradas: {len(config_points)}")

    # Calcular hipervolumen por configuración
    hv_by_config = []
    # Punto de referencia para maximización: (0,0) - cualquier punto real será mejor
    ref_point = [0.0, 0.0]

    print(f"Usando punto de referencia para MAXIMIZACIÓN: {ref_point}")
    print(f"Hipervolumen esperado en rango: [0, 1.0]")
    print("Interpretación: cost_maximized=1 significa costo mínimo, minutes_maximized=1 significa tiempo máximo")

    for config, points in config_points.items():
        if not points:
            continue

        # Convertir puntos a array numpy para facilitar el manejo
        points_array = np.array(points)

        # Obtener frente de Pareto (para MAXIMIZACIÓN)
        pareto_indices = []
        for i, point in enumerate(points_array):
            is_dominated = False
            for j, other_point in enumerate(points_array):
                if i != j:
                    # Para maximización: un punto domina a otro si es mejor o igual en todos los objetivos
                    # y estrictamente mejor en al menos uno
                    if (other_point[0] >= point[0] and other_point[1] >= point[1] and
                            (other_point[0] > point[0] or other_point[1] > point[1])):
                        is_dominated = True
                        break
            if not is_dominated:
                pareto_indices.append(i)

        pareto_front = points_array[pareto_indices].tolist()

        # Calcular medias de las métricas
        metrics = config_metrics[config]
        mean_exec_time = np.mean(metrics['execution_times']) if metrics['execution_times'] else 0
        mean_generations = np.mean(metrics['generations']) if metrics['generations'] else 0
        mean_pareto_size = np.mean(metrics['pareto_sizes']) if metrics['pareto_sizes'] else 0

        # Calcular desviaciones estándar
        std_exec_time = np.std(metrics['execution_times']) if len(metrics['execution_times']) > 1 else 0
        std_generations = np.std(metrics['generations']) if len(metrics['generations']) > 1 else 0
        std_pareto_size = np.std(metrics['pareto_sizes']) if len(metrics['pareto_sizes']) > 1 else 0

        try:
            # VALIDACIÓN: Verificar que todos los puntos estén en [0,1]
            points_array = np.array(pareto_front)
            if points_array.size > 0:
                min_vals = points_array.min(axis=0)
                max_vals = points_array.max(axis=0)

                if min_vals[0] < 0 or min_vals[1] < 0 or max_vals[0] > 1 or max_vals[1] > 1:
                    print(f"  ADVERTENCIA: Puntos fuera de [0,1] en config {config}")
                    print(f"    Rango cost_maximized: [{min_vals[0]:.6f}, {max_vals[0]:.6f}]")
                    print(f"    Rango minutes_maximized: [{min_vals[1]:.6f}, {max_vals[1]:.6f}]")

                # Calcular hipervolumen
                # Con ref_point=[0,0] y puntos en [0,1], el HV máximo es 1.0
                hv = hypervolume(pareto_front, reference_point=ref_point)

                # VALIDACIÓN: Verificar que el HV esté en rango esperado
                max_possible_hv = 1.0
                if hv > max_possible_hv:
                    print(f"  ⚠️  ADVERTENCIA: HV ({hv:.6f}) excede máximo teórico ({max_possible_hv:.6f})")
                elif hv < 0:
                    print(f"  ⚠️  ADVERTENCIA: HV negativo ({hv:.6f})")

            else:
                hv = 0.0

            hv_by_config.append({
                "dataset": dataset_name,
                "config": config,
                "num_points": len(points),
                "num_pareto_points": len(pareto_front),
                "num_runs": len(metrics['execution_times']),
                "hypervolume": hv,
                "mean_execution_time": mean_exec_time,
                "std_execution_time": std_exec_time,
                "mean_generations": mean_generations,
                "std_generations": std_generations,
                "mean_pareto_size": mean_pareto_size,
                "std_pareto_size": std_pareto_size
            })

            print(f"Config {config}: {len(points)} puntos totales, {len(pareto_front)} en frente de Pareto")
            print(f"  HV = {hv:.6f} (0=peor, 1=mejor)")
            print(f"  Exec time: {mean_exec_time:.2f}±{std_exec_time:.2f}s")
            print(f"  Generations: {mean_generations:.1f}±{std_generations:.1f}")
            print(f"  Pareto size: {mean_pareto_size:.1f}±{std_pareto_size:.1f}")

        except Exception as e:
            print(f"ERROR calculando hipervolumen para config {config}: {e}")
            hv_by_config.append({
                "dataset": dataset_name,
                "config": config,
                "num_points": len(points),
                "num_pareto_points": len(pareto_front) if 'pareto_front' in locals() else 0,
                "num_runs": len(metrics['execution_times']),
                "hypervolume": 0.0,
                "mean_execution_time": mean_exec_time,
                "std_execution_time": std_exec_time,
                "mean_generations": mean_generations,
                "std_generations": std_generations,
                "mean_pareto_size": mean_pareto_size,
                "std_pareto_size": std_pareto_size
            })

    return hv_by_config


def main():
    # Cargar puntos de referencia globales
    global_stats = load_global_reference_points(
        "C:\\Users\\hctr0\\PycharmProjects\\TFG_Hector\\pareto_outputs\\pareto_stats_summary.csv")
    if global_stats is None:
        print("ERROR: No se pudieron cargar los puntos de referencia globales. Terminando...")
        return

    # Configuración de rutas
    base_path = "C:\\Users\\hctr0\\PycharmProjects\\TFG_Hector\\results\\NSGA2\\summaries"

    # Datasets a procesar
    datasets = [
        ("users1", "users1.json"),
        ("users2", "users2.json"),
        ("users3", "users3.json"),
        ("users4", "users4.json"),
        ("users5", "users5.json")
    ]

    all_results = []

    # Procesar cada dataset
    for folder, dataset_file in datasets:
        file_path = os.path.join(base_path, folder, f"summary_complete_{folder}.json")
        results = calculate_hypervolume_for_dataset_with_global_ref(file_path, dataset_file, global_stats)

        if results:
            all_results.extend(results)

    if not all_results:
        print("No se obtuvieron resultados para ningún dataset.")
        return

    # Crear DataFrame con todos los resultados
    results_df = pd.DataFrame(all_results)

    print(f"\n{'=' * 120}")
    print("RESULTADOS COMPLETOS - HIPERVOLUMEN CORREGIDO (MAXIMIZACIÓN)")
    print(f"{'=' * 120}")
    print("INTERPRETACIÓN:")
    print("- cost_maximized: 1 = costo mínimo (mejor), 0 = costo máximo (peor)")
    print("- minutes_maximized: 1 = tiempo máximo (mejor), 0 = tiempo mínimo (peor)")
    print("- Hipervolumen: 0 = peor rendimiento, 1 = mejor rendimiento posible")

    # Mostrar resultados ordenados por dataset y luego por hipervolumen
    results_sorted = results_df.sort_values(["dataset", "hypervolume"], ascending=[True, False])

    # Formatear el DataFrame para mejor visualización
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.float_format', '{:.4f}'.format)

    print(results_sorted.to_string(index=False))

    # Guardar resultados completos
    results_sorted.to_csv("hypervolume_results_corrected.csv", index=False)
    print(f"\nResultados completos guardados en 'hypervolume_results_corrected.csv'")

    # Análisis por dataset
    print(f"\n{'=' * 120}")
    print("ANÁLISIS POR DATASET (HIPERVOLUMEN CORREGIDO)")
    print(f"{'=' * 120}")

    dataset_summary = []
    for dataset in results_df['dataset'].unique():
        dataset_data = results_df[results_df['dataset'] == dataset]
        summary = {
            'dataset': dataset,
            'num_configs': len(dataset_data),
            'best_hv': dataset_data['hypervolume'].max(),
            'worst_hv': dataset_data['hypervolume'].min(),
            'mean_hv': dataset_data['hypervolume'].mean(),
            'std_hv': dataset_data['hypervolume'].std(),
            'best_config': dataset_data.loc[dataset_data['hypervolume'].idxmax(), 'config'],
            'mean_exec_time_all': dataset_data['mean_execution_time'].mean(),
            'mean_generations_all': dataset_data['mean_generations'].mean(),
            'mean_pareto_size_all': dataset_data['mean_pareto_size'].mean()
        }
        dataset_summary.append(summary)

    summary_df = pd.DataFrame(dataset_summary)
    print(summary_df.to_string(index=False))

    summary_df.to_csv("hypervolume_summary_corrected.csv", index=False)
    print(f"\nResumen por dataset guardado en 'hypervolume_summary_corrected.csv'")

    print(f"\n{'=' * 120}")
    print("MEJORES CONFIGURACIONES POR DATASET (HIPERVOLUMEN CORREGIDO)")
    print(f"{'=' * 120}")

    best_configs = results_df.loc[results_df.groupby('dataset')['hypervolume'].idxmax()]
    best_configs_display = best_configs[[
        'dataset', 'config', 'hypervolume', 'num_points', 'num_pareto_points', 'num_runs',
        'mean_execution_time', 'mean_generations', 'mean_pareto_size'
    ]]
    print(best_configs_display.to_string(index=False))

    best_configs_display.to_csv("best_configs_corrected.csv", index=False)
    print(f"\nMejores configuraciones guardadas en 'best_configs_corrected.csv'")

    print(f"\n{'=' * 120}")
    print("ANÁLISIS DE MEJOR CONFIGURACIÓN GLOBAL - HIPERVOLUMEN CORREGIDO")
    print(f"{'=' * 120}")

    config_means = results_df.groupby('config').agg({
        'hypervolume': ['mean', 'std', 'count'],
        'mean_execution_time': 'mean',
        'mean_generations': 'mean',
        'mean_pareto_size': 'mean',
        'num_points': 'mean',
        'num_pareto_points': 'mean',
        'num_runs': 'mean'
    }).round(6)

    config_means.columns = [
        'hv_mean', 'hv_std', 'hv_count',
        'exec_time_mean', 'generations_mean', 'pareto_size_mean',
        'points_mean', 'pareto_points_mean', 'runs_mean'
    ]

    config_means = config_means.reset_index()

    config_means_sorted = config_means.sort_values('hv_mean', ascending=False)

    print("RANKING DE CONFIGURACIONES POR HIPERVOLUMEN PROMEDIO (CORREGIDO):")
    print("-" * 140)

    for idx, row in config_means_sorted.iterrows():
        rank = config_means_sorted.index.get_loc(idx) + 1
        print(f"Rank {rank:2d}: {row['config']:<25}")
        print(f"         HV promedio: {row['hv_mean']:.6f} ± {row['hv_std']:.6f} (n={int(row['hv_count'])} datasets)")
        print(f"         Tiempo medio: {row['exec_time_mean']:.2f}s")
        print(f"         Generaciones: {row['generations_mean']:.1f}")
        print(f"         Tamaño Pareto: {row['pareto_size_mean']:.1f}")
        print(f"         Puntos Pareto promedio: {row['pareto_points_mean']:.1f}")
        print()

    config_means_sorted.to_csv("config_ranking_corrected.csv", index=False)
    print(f"Ranking completo guardado en 'config_ranking_corrected.csv'")

    best_global_config = config_means_sorted.iloc[0]

    print(f"\n{'🏆' * 60}")
    print("MEJOR CONFIGURACIÓN GLOBAL - HIPERVOLUMEN CORREGIDO:")
    print(f"{'🏆' * 60}")
    print(f"Configuración: {best_global_config['config']}")
    print(f"Hipervolumen promedio: {best_global_config['hv_mean']:.6f} ± {best_global_config['hv_std']:.6f}")
    print(f"Evaluada en {int(best_global_config['hv_count'])} datasets")
    print(f"Tiempo de ejecución promedio: {best_global_config['exec_time_mean']:.2f} segundos")
    print(f"Generaciones promedio: {best_global_config['generations_mean']:.1f}")
    print(f"Tamaño Pareto promedio: {best_global_config['pareto_size_mean']:.1f}")
    print(f"Puntos Pareto promedio: {best_global_config['pareto_points_mean']:.1f}")
    print("💡 Interpretación: Mayor HV = mejor balance entre bajo costo y alto tiempo de ejecución")

    print(f"\n{'=' * 100}")
    print(f"DETALLES DE '{best_global_config['config']}' POR DATASET:")
    print(f"{'=' * 100}")

    best_config_details = results_df[results_df['config'] == best_global_config['config']].sort_values('dataset')
    best_config_summary = best_config_details[[
        'dataset', 'hypervolume', 'num_pareto_points', 'mean_execution_time',
        'mean_generations', 'mean_pareto_size', 'num_runs'
    ]]

    print(best_config_summary.to_string(index=False))

    best_config_summary.to_csv("best_global_config_details_corrected.csv", index=False)
    print(f"\nDetalles de la mejor configuración guardados en 'best_global_config_details_corrected.csv'")

    print(f"\n{'🎯' * 50}")
    print("CORRECCIÓN APLICADA:")
    print(f"{'🎯' * 50}")
    print("✅ Transformación correcta para maximización:")
    print("   - Costo: (max_cost - cost) / (max_cost - min_cost)")
    print("   - Tiempo: (minutes - min_minutes) / (max_minutes - min_minutes)")
    print("✅ Punto de referencia: [0, 0] para maximización")
    print("✅ Dominancia corregida para maximización")
    print("✅ Hipervolumen ahora refleja correctamente el rendimiento")


if __name__ == "__main__":
    main()