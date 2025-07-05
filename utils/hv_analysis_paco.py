import json
import numpy as np
from inspyred.ec.analysis import hypervolume
import pandas as pd
import os
from pathlib import Path


def calculate_hypervolume_for_dataset(file_path, dataset_name):
    """
    Calcula el hipervolumen para un dataset específico (adaptado para PACO)
    """
    print(f"\n{'=' * 60}")
    print(f"PROCESANDO: {dataset_name}")
    print(f"Archivo: {file_path}")
    print(f"{'=' * 60}")

    # Verificar que el archivo existe
    if not os.path.exists(file_path):
        print(f"ADVERTENCIA: El archivo {file_path} no existe. Saltando...")
        return None

    # Cargar el archivo
    try:
        with open(file_path) as f:
            data = json.load(f)
    except Exception as e:
        print(f"ERROR: No se pudo cargar {file_path}: {e}")
        return None

    all_costs = []
    all_minutes = []

    # Recopilar todos los puntos para normalización global
    for entry in data:
        if entry["dataset"] == dataset_name:
            for cost, minutes in entry["pareto_points"]:
                all_costs.append(cost)
                all_minutes.append(minutes)

    if not all_costs:
        print(f"ADVERTENCIA: No se encontraron datos para {dataset_name}")
        return None

    # Máximos y mínimos globales para normalización
    max_cost = max(all_costs)
    min_cost = min(all_costs)
    max_minutes = max(all_minutes)
    min_minutes = min(all_minutes)

    print(f"Max cost: {max_cost:.4f}")
    print(f"Min cost: {min_cost:.4f}")
    print(f"Max minutes: {max_minutes:.4f}")
    print(f"Min minutes: {min_minutes:.4f}")
    print(f"Total puntos encontrados: {len(all_costs)}")

    config_points = {}
    config_metrics = {}

    # Agrupar puntos por configuración y recopilar métricas
    for entry in data:
        if entry["dataset"] == dataset_name:
            config = entry["config"]

            # Inicializar estructuras de datos para esta configuración
            if config not in config_points:
                config_points[config] = []
                config_metrics[config] = {
                    'execution_times': [],
                    'iterations': [],
                    'pareto_sizes': [],
                    'solutions_evaluated': [],
                    'terminated_early': []
                }

            # Recopilar métricas de ejecución (adaptado para PACO)
            config_metrics[config]['execution_times'].append(entry.get('execution_time', 0))
            config_metrics[config]['iterations'].append(entry.get('iterations_completed', 0))
            config_metrics[config]['pareto_sizes'].append(entry.get('pareto_size', 0))
            config_metrics[config]['solutions_evaluated'].append(entry.get('solutions_evaluated', 0))
            config_metrics[config]['terminated_early'].append(entry.get('terminated_early', False))

            for cost, minutes in entry["pareto_points"]:
                cost_norm = (cost - min_cost) / (max_cost - min_cost) if max_cost != min_cost else 0
                minutes_norm = (minutes - min_minutes) / (
                            max_minutes - min_minutes) if max_minutes != min_minutes else 0
                config_points[config].append([cost_norm, minutes_norm])

    # Calcular hipervolumen por configuración
    hv_by_config = []
    ref_point = [1.0, 1.0]

    print(f"\nConfigurations encontradas: {len(config_points)}")

    for config, points in config_points.items():
        if not points:
            continue

        # Convertir a DataFrame para análisis
        df = pd.DataFrame(points, columns=["cost_norm", "minutes_norm"])

        # Segunda normalización (como en el código original)
        df["cost_final"] = df["cost_norm"] / df["cost_norm"].max() if df["cost_norm"].max() > 0 else df["cost_norm"]
        df["minutes_final"] = df["minutes_norm"] / df["minutes_norm"].max() if df["minutes_norm"].max() > 0 else df[
            "minutes_norm"]

        pareto_front = df[["cost_final", "minutes_final"]].values.tolist()

        # Calcular medias de las métricas
        metrics = config_metrics[config]
        mean_exec_time = np.mean(metrics['execution_times']) if metrics['execution_times'] else 0
        mean_iterations = np.mean(metrics['iterations']) if metrics['iterations'] else 0
        mean_pareto_size = np.mean(metrics['pareto_sizes']) if metrics['pareto_sizes'] else 0
        mean_solutions_evaluated = np.mean(metrics['solutions_evaluated']) if metrics['solutions_evaluated'] else 0
        early_termination_rate = np.mean(metrics['terminated_early']) if metrics['terminated_early'] else 0

        # Calcular desviaciones estándar
        std_exec_time = np.std(metrics['execution_times']) if len(metrics['execution_times']) > 1 else 0
        std_iterations = np.std(metrics['iterations']) if len(metrics['iterations']) > 1 else 0
        std_pareto_size = np.std(metrics['pareto_sizes']) if len(metrics['pareto_sizes']) > 1 else 0
        std_solutions_evaluated = np.std(metrics['solutions_evaluated']) if len(
            metrics['solutions_evaluated']) > 1 else 0

        try:
            hv = hypervolume(pareto_front, reference_point=ref_point)
            hv_by_config.append({
                "dataset": dataset_name,
                "config": config,
                "num_points": len(points),
                "num_runs": len(metrics['execution_times']),
                "hypervolume": hv,
                "mean_execution_time": mean_exec_time,
                "std_execution_time": std_exec_time,
                "mean_iterations": mean_iterations,
                "std_iterations": std_iterations,
                "mean_pareto_size": mean_pareto_size,
                "std_pareto_size": std_pareto_size,
                "mean_solutions_evaluated": mean_solutions_evaluated,
                "std_solutions_evaluated": std_solutions_evaluated,
                "early_termination_rate": early_termination_rate
            })
            print(f"Config {config}: {len(points)} puntos, {len(metrics['execution_times'])} runs")
            print(f"  HV = {hv:.6f}")
            print(f"  Exec time: {mean_exec_time:.2f}±{std_exec_time:.2f}s")
            print(f"  Iterations: {mean_iterations:.1f}±{std_iterations:.1f}")
            print(f"  Pareto size: {mean_pareto_size:.1f}±{std_pareto_size:.1f}")
            print(f"  Solutions evaluated: {mean_solutions_evaluated:.1f}±{std_solutions_evaluated:.1f}")
            print(f"  Early termination rate: {early_termination_rate:.2%}")
        except Exception as e:
            print(f"ERROR calculando hipervolumen para config {config}: {e}")
            hv_by_config.append({
                "dataset": dataset_name,
                "config": config,
                "num_points": len(points),
                "num_runs": len(metrics['execution_times']),
                "hypervolume": 0.0,
                "mean_execution_time": mean_exec_time,
                "std_execution_time": std_exec_time,
                "mean_iterations": mean_iterations,
                "std_iterations": std_iterations,
                "mean_pareto_size": mean_pareto_size,
                "std_pareto_size": std_pareto_size,
                "mean_solutions_evaluated": mean_solutions_evaluated,
                "std_solutions_evaluated": std_solutions_evaluated,
                "early_termination_rate": early_termination_rate
            })

    return hv_by_config


def main():
    # Configuración de rutas - ADAPTAR SEGÚN TU ESTRUCTURA DE CARPETAS
    base_path = "C:\\Users\\hctr0\\PycharmProjects\\TFG_Hector\\results\\PACO\\summaries"

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
        results = calculate_hypervolume_for_dataset(file_path, dataset_file)

        if results:
            all_results.extend(results)

    if not all_results:
        print("No se obtuvieron resultados para ningún dataset.")
        return

    # Crear DataFrame con todos los resultados
    results_df = pd.DataFrame(all_results)

    print(f"\n{'=' * 120}")
    print("RESULTADOS COMPLETOS - HIPERVOLUMEN Y MÉTRICAS PACO POR DATASET Y CONFIGURACIÓN")
    print(f"{'=' * 120}")

    # Mostrar resultados ordenados por dataset y luego por hipervolumen
    results_sorted = results_df.sort_values(["dataset", "hypervolume"], ascending=[True, False])

    # Formatear el DataFrame para mejor visualización
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.float_format', '{:.4f}'.format)

    print(results_sorted.to_string(index=False))

    # Guardar resultados completos
    results_sorted.to_csv("paco_hypervolume_results_all_datasets_complete.csv", index=False)
    print(f"\nResultados completos guardados en 'paco_hypervolume_results_all_datasets_complete.csv'")

    # Análisis por dataset
    print(f"\n{'=' * 120}")
    print("ANÁLISIS POR DATASET")
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
            'mean_iterations_all': dataset_data['mean_iterations'].mean(),
            'mean_pareto_size_all': dataset_data['mean_pareto_size'].mean(),
            'mean_early_termination_rate': dataset_data['early_termination_rate'].mean()
        }
        dataset_summary.append(summary)

    summary_df = pd.DataFrame(dataset_summary)
    print(summary_df.to_string(index=False))

    # Guardar resumen
    summary_df.to_csv("paco_hypervolume_summary_by_dataset_complete.csv", index=False)
    print(f"\nResumen por dataset guardado en 'paco_hypervolume_summary_by_dataset_complete.csv'")

    # Mejores configuraciones por dataset
    print(f"\n{'=' * 120}")
    print("MEJORES CONFIGURACIONES PACO POR DATASET (CON MÉTRICAS COMPLETAS)")
    print(f"{'=' * 120}")

    best_configs = results_df.loc[results_df.groupby('dataset')['hypervolume'].idxmax()]
    best_configs_display = best_configs[[
        'dataset', 'config', 'hypervolume', 'num_points', 'num_runs',
        'mean_execution_time', 'mean_iterations', 'mean_pareto_size', 'early_termination_rate'
    ]]
    print(best_configs_display.to_string(index=False))

    # Guardar mejores configuraciones
    best_configs_display.to_csv("paco_best_configs_by_dataset_complete.csv", index=False)
    print(f"\nMejores configuraciones guardadas en 'paco_best_configs_by_dataset_complete.csv'")

    # Análisis de correlaciones
    print(f"\n{'=' * 120}")
    print("ANÁLISIS DE CORRELACIONES PACO")
    print(f"{'=' * 120}")

    correlation_cols = ['hypervolume', 'mean_execution_time', 'mean_iterations', 'mean_pareto_size',
                        'early_termination_rate']
    correlations = results_df[correlation_cols].corr()
    print("Matriz de correlaciones:")
    print(correlations.to_string())

    # Guardar correlaciones
    correlations.to_csv("paco_correlation_analysis.csv")
    print(f"\nAnálisis de correlaciones guardado en 'paco_correlation_analysis.csv'")

    # ANÁLISIS DE MEJOR CONFIGURACIÓN GLOBAL (PROMEDIO ENTRE DATASETS)
    print(f"\n{'=' * 120}")
    print("ANÁLISIS DE MEJOR CONFIGURACIÓN GLOBAL PACO - PROMEDIO ENTRE DATASETS")
    print(f"{'=' * 120}")

    # Calcular media de hipervolumen por configuración across datasets
    config_means = results_df.groupby('config').agg({
        'hypervolume': ['mean', 'std', 'count'],
        'mean_execution_time': 'mean',
        'mean_iterations': 'mean',
        'mean_pareto_size': 'mean',
        'early_termination_rate': 'mean',
        'num_points': 'mean',
        'num_runs': 'mean'
    }).round(6)

    # Aplanar el índice multi-nivel
    config_means.columns = [
        'hv_mean', 'hv_std', 'hv_count',
        'exec_time_mean', 'iterations_mean', 'pareto_size_mean', 'early_term_rate_mean',
        'points_mean', 'runs_mean'
    ]

    # Resetear el índice para tener 'config' como columna
    config_means = config_means.reset_index()

    # Ordenar por hipervolumen medio (descendente)
    config_means_sorted = config_means.sort_values('hv_mean', ascending=False)

    print("RANKING DE CONFIGURACIONES PACO POR HIPERVOLUMEN PROMEDIO:")
    print("-" * 140)

    # Mostrar resultados formateados
    for idx, row in config_means_sorted.iterrows():
        print(f"Rank {config_means_sorted.index.get_loc(idx) + 1:2d}: {row['config']:<25}")
        print(f"         HV promedio: {row['hv_mean']:.6f} ± {row['hv_std']:.6f} (n={int(row['hv_count'])} datasets)")
        print(f"         Tiempo medio: {row['exec_time_mean']:.2f}s")
        print(f"         Iteraciones: {row['iterations_mean']:.1f}")
        print(f"         Tamaño Pareto: {row['pareto_size_mean']:.1f}")
        print(f"         Terminación temprana: {row['early_term_rate_mean']:.2%}")
        print()

    # Guardar ranking completo
    config_means_sorted.to_csv("paco_config_ranking_global.csv", index=False)
    print(f"Ranking completo guardado en 'paco_config_ranking_global.csv'")

    # Identificar la mejor configuración
    best_global_config = config_means_sorted.iloc[0]

    print(f"\n{'🏆' * 50}")
    print("MEJOR CONFIGURACIÓN GLOBAL PACO:")
    print(f"{'🏆' * 50}")
    print(f"Configuración: {best_global_config['config']}")
    print(f"Hipervolumen promedio: {best_global_config['hv_mean']:.6f} ± {best_global_config['hv_std']:.6f}")
    print(f"Evaluada en {int(best_global_config['hv_count'])} datasets")
    print(f"Tiempo de ejecución promedio: {best_global_config['exec_time_mean']:.2f} segundos")
    print(f"Iteraciones promedio: {best_global_config['iterations_mean']:.1f}")
    print(f"Tamaño Pareto promedio: {best_global_config['pareto_size_mean']:.1f}")
    print(f"Tasa de terminación temprana: {best_global_config['early_term_rate_mean']:.2%}")

    # Detalles por dataset de la mejor configuración
    print(f"\n{'=' * 100}")
    print(f"DETALLES DE '{best_global_config['config']}' POR DATASET:")
    print(f"{'=' * 100}")

    best_config_details = results_df[results_df['config'] == best_global_config['config']].sort_values('dataset')
    best_config_summary = best_config_details[[
        'dataset', 'hypervolume', 'mean_execution_time', 'mean_iterations', 'mean_pareto_size',
        'early_termination_rate', 'num_runs'
    ]]

    print(best_config_summary.to_string(index=False))

    # Guardar detalles de la mejor configuración
    best_config_summary.to_csv("paco_best_global_config_details.csv", index=False)
    print(f"\nDetalles de la mejor configuración guardados en 'paco_best_global_config_details.csv'")

    # Análisis de consistencia
    print(f"\n{'=' * 100}")
    print("ANÁLISIS DE CONSISTENCIA DE CONFIGURACIONES PACO:")
    print(f"{'=' * 100}")

    config_means_sorted['hv_cv'] = config_means_sorted['hv_std'] / config_means_sorted['hv_mean']

    most_consistent = config_means_sorted.sort_values('hv_cv').head(5)

    print("TOP 5 CONFIGURACIONES MÁS CONSISTENTES (menor variabilidad):")
    for idx, row in most_consistent.iterrows():
        rank_in_hv = config_means_sorted[config_means_sorted['config'] == row['config']].index[0] + 1
        print(f"{most_consistent.index.get_loc(idx) + 1}. {row['config']:<25} (Rank HV: {rank_in_hv})")
        print(f"   CV: {row['hv_cv']:.4f}, HV: {row['hv_mean']:.6f} ± {row['hv_std']:.6f}")

    consistency_analysis = config_means_sorted[['config', 'hv_mean', 'hv_std', 'hv_cv', 'hv_count']].copy()
    consistency_analysis.to_csv("paco_consistency_analysis.csv", index=False)
    print(f"\nAnálisis de consistencia guardado en 'paco_consistency_analysis.csv'")

    print(f"\n{'=' * 100}")
    print("RECOMENDACIÓN FINAL PARA PACO:")
    print(f"{'=' * 100}")
    print(f"Para el algoritmo PACO, usa la configuración: {best_global_config['config']}")
    print(f"Esta configuración tiene el mejor hipervolumen promedio ({best_global_config['hv_mean']:.6f})")
    print(f"y ha sido evaluada consistentemente en {int(best_global_config['hv_count'])} datasets diferentes.")


if __name__ == "__main__":
    main()