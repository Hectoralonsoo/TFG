import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from inspyred.ec.analysis import hypervolume


def validate_hypervolume_results(file_path, dataset_name, top_configs):
    """
    Valida los resultados de hipervolumen analizando la distribución de puntos
    """
    print(f"\n{'=' * 60}")
    print(f"VALIDACIÓN PARA: {dataset_name}")
    print(f"{'=' * 60}")

    with open(file_path) as f:
        data = json.load(f)

    all_costs = []
    all_inv_minutes = []

    # Recopilar todos los puntos
    for entry in data:
        if entry["dataset"] == dataset_name:
            for neg_minutes, cost in entry["pareto_points"]:
                minutes = -neg_minutes
                inv_minutes = 1 / minutes
                all_costs.append(cost)
                all_inv_minutes.append(inv_minutes)

    max_cost = max(all_costs)
    min_cost = min(all_costs)
    max_inv_minutes = max(all_inv_minutes)
    min_inv_minutes = min(all_inv_minutes)

    print(f"Rango original - Cost: [{min_cost:.2f}, {max_cost:.2f}], Inv_minutes: [{min_inv_minutes:.10f}, {max_inv_minutes:.10f}]")

    config_analysis = {}

    for entry in data:
        if entry["dataset"] == dataset_name:
            config = entry["config"]
            if config in top_configs:
                if config not in config_analysis:
                    config_analysis[config] = {
                        'points': [],
                        'original_points': []
                    }

                for neg_minutes, cost in entry["pareto_points"]:
                    minutes = -neg_minutes
                    inv_minutes = 1 / minutes
                    cost_norm = (cost - min_cost) / (max_cost - min_cost)
                    inv_minutes_norm = (inv_minutes - min_inv_minutes) / (max_inv_minutes - min_inv_minutes)

                    config_analysis[config]['original_points'].append([cost, inv_minutes])
                    config_analysis[config]['points'].append([cost_norm, inv_minutes_norm])

    # Análisis detallado por configuración
    for config, data_dict in config_analysis.items():
        points = np.array(data_dict['points'])
        original_points = np.array(data_dict['original_points'])

        print(f"\n--- Configuración: {config} ---")
        print(f"Número de puntos: {len(points)}")

        # Estadísticas de puntos normalizados
        print("Puntos normalizados:")
        print(f"  Cost - Min: {points[:, 0].min():.4f}, Max: {points[:, 0].max():.4f}, Mean: {points[:, 0].mean():.4f}")
        print(
            f"  Inv_minutes - Min: {points[:, 1].min():.4f}, Max: {points[:, 1].max():.4f}, Mean: {points[:, 1].mean():.4f}")

        # Aplicar doble normalización (como en tu código)
        df = pd.DataFrame(points, columns=["cost", "inv_minutes"])
        df["cost_norm"] = (df["cost"] - df["cost"].min()) / (df["cost"].max() - df["cost"].min())
        df["inv_minutes_norm"] = (df["inv_minutes"] - df["inv_minutes"].min()) / (
        df["inv_minutes"].max() - df["inv_minutes"].min())

        double_norm_points = df[["cost_norm", "inv_minutes_norm"]].values

        print("Después de doble normalización:")
        print(f"  Cost - Min: {double_norm_points[:, 0].min():.4f}, Max: {double_norm_points[:, 0].max():.4f}")
        print(f"  Inv_minutes - Min: {double_norm_points[:, 1].min():.4f}, Max: {double_norm_points[:, 1].max():.4f}")

        # Calcular hipervolumen con diferentes puntos de referencia
        ref_points = [
            [0.0, 0.0],
            [0.1, 0.1],
            [0.5, 0.5]
        ]

        print("Hipervolumen con diferentes puntos de referencia:")
        for ref_point in ref_points:
            try:
                hv = hypervolume(double_norm_points.tolist(), reference_point=ref_point)
                print(f"  Ref {ref_point}: {hv:.6f}")
            except:
                print(f"  Ref {ref_point}: ERROR")

        # Verificar si hay puntos en las esquinas
        corner_threshold = 0.95
        corner_points = double_norm_points[
            (double_norm_points[:, 0] > corner_threshold) |
            (double_norm_points[:, 1] > corner_threshold)
            ]
        print(f"Puntos cerca de esquinas (>{corner_threshold}): {len(corner_points)}")


def compare_normalization_methods(file_path, dataset_name, config_name):
    """
    Compara diferentes métodos de normalización
    """
    print(f"\n{'=' * 60}")
    print(f"COMPARACIÓN DE NORMALIZACIÓN: {config_name}")
    print(f"{'=' * 60}")

    with open(file_path) as f:
        data = json.load(f)

    # Obtener puntos para la configuración específica
    config_points = []
    all_costs = []
    all_inv_minutes = []

    for entry in data:
        if entry["dataset"] == dataset_name:
            for neg_minutes, cost in entry["pareto_points"]:
                minutes = -neg_minutes
                inv_minutes = 1 / minutes
                all_costs.append(cost)
                all_inv_minutes.append(inv_minutes)

            if entry["config"] == config_name:
                for neg_minutes, cost in entry["pareto_points"]:
                    minutes = -neg_minutes
                    inv_minutes = 1 / minutes
                    config_points.append([cost, inv_minutes])

    config_points = np.array(config_points)

    # Método 1: Normalización global simple
    max_cost_global = max(all_costs)
    max_inv_minutes_global = max(all_inv_minutes)

    norm1 = config_points.copy()
    norm1[:, 0] = norm1[:, 0] / max_cost_global
    norm1[:, 1] = norm1[:, 1] / max_inv_minutes_global

    # Método 2: Tu método (doble normalización)
    df = pd.DataFrame(norm1, columns=["cost", "inv_minutes"])
    df["cost_norm"] = df["cost"] / df["cost"].max()
    df["inv_minutes_norm"] = df["inv_minutes"] / df["inv_minutes"].max()
    norm2 = df[["cost_norm", "inv_minutes_norm"]].values

    # Método 3: Normalización Min-Max
    norm3 = config_points.copy()
    norm3[:, 0] = (norm3[:, 0] - norm3[:, 0].min()) / (norm3[:, 0].max() - norm3[:, 0].min())
    norm3[:, 1] = (norm3[:, 1] - norm3[:, 1].min()) / (norm3[:, 1].max() - norm3[:, 1].min())

    # Comparar hipervolúmenes
    ref_point = [0.0, 0.0]
    methods = [
        ("Global", norm1),
        ("Doble (tu método)", norm2),
        ("Min-Max", norm3)
    ]

    for name, points in methods:
        try:
            hv = hypervolume(points.tolist(), reference_point=ref_point)
            print(f"{name:15}: HV = {hv:.6f}, Max values = [{points[:, 0].max():.3f}, {points[:, 1].max():.3f}]")
        except Exception as e:
            print(f"{name:15}: ERROR - {e}")


# Ejemplo de uso
if __name__ == "__main__":
    # Configuraciones a analizar (las mejores de cada dataset)
    datasets_configs = {
        "users1.json": "npoint_reset_low_crossover",
        "users2.json": "uniform_reset_high_crossover",
        "users3.json": "uniform_inversion_high_mutation",
        "users4.json": "uniform_reset_low_mutation",
        "users5.json": "uniform_reset_high_crossover"
    }

    base_path = "C:\\Users\\hctr0\\PycharmProjects\\TFG_Hector\\results\\NSGA2\\summaries"

    for dataset, config in datasets_configs.items():
        folder = dataset.replace('.json', '')
        file_path = f"{base_path}\\{folder}\\summary_complete_{folder}.json"

        # Validar resultados
        validate_hypervolume_results(file_path, dataset, [config])

        # Comparar métodos de normalización
        compare_normalization_methods(file_path, dataset, config)

        print("\n" + "=" * 80)