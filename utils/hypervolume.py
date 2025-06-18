import json
import numpy as np
from inspyred.ec.analysis import hypervolume
import csv


def normalize_front(front):
    front = np.array(front)
    mins = np.min(front, axis=0)
    maxs = np.max(front, axis=0)
    ranges = np.maximum(maxs - mins, 1e-10)
    normalized = (front - mins) / ranges
    print(f'Front' + str(front))
    print(f'Valores normalizados' + str(normalized))
    print(f'Rangos' + str(ranges))
    print(f'Valores normalizados mins' + str(mins))
    print(f'Valores normalizados maxs' + str(maxs))

    return normalized.tolist(), mins.tolist(), maxs.tolist()


def compute_hypervolumes_from_runs(file_path, reference_point=[1.1, 1.1]):
    with open(file_path, "r", encoding="utf-8") as f:
        runs = json.load(f)

    hv_results = []

    for run in runs:
        run_id = run.get("run", "N/A")
        pareto_points = run.get("pareto_points", [])

        if not pareto_points:
            continue

        # Convertir de [-coste, minutos] a [minutos, coste] (ambos a maximizar)
        converted = [[p[1], -p[0]] for p in pareto_points]
        normalized, _, _ = normalize_front(converted)
        hv = hypervolume(normalized, reference_point)

        hv_results.append({
            "run": run_id,
            "hipervolumen": hv,
            "num_puntos": len(pareto_points)
        })

    return hv_results

def compute_hypervolumes_from_runs_global(file_path, reference_point=[1.1, 1.1]):
    import json
    import numpy as np
    from inspyred.ec.analysis import hypervolume

    with open(file_path, "r", encoding="utf-8") as f:
        runs = json.load(f)

    all_points = []

    # Recoger todos los puntos
    for run in runs:
        pareto = run.get("pareto_points", [])
        converted = [[p[1], -p[0]] for p in pareto]  # [minutos, coste]
        all_points.extend(converted)

    # Obtener mins y maxs globales
    all_points = np.array(all_points)
    mins = np.min(all_points, axis=0)
    maxs = np.max(all_points, axis=0)
    ranges = np.maximum(maxs - mins, 1e-10)

    results = []
    for run in runs:
        run_id = run.get("run", "N/A")
        pareto = run.get("pareto_points", [])
        if not pareto:
            continue

        converted = np.array([[p[1], -p[0]] for p in pareto])
        normalized = (converted - mins) / ranges
        hv = hypervolume(normalized.tolist(), reference_point)

        results.append({
            "run": run_id,
            "hipervolumen": hv,
            "num_puntos": len(pareto)
        })

    return results


file_path = "../results/NSGA2/summary_runs.json"
hv_results = compute_hypervolumes_from_runs_global(file_path)

for r in hv_results:
    print(f"Run {r['run']}: HV = {r['hipervolumen']:.4f} con {r['num_puntos']} soluciones")


def save_hv_results_to_csv(hv_results, output_path):
    with open(output_path, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["run", "hipervolumen", "num_puntos"])
        writer.writeheader()
        writer.writerows(hv_results)

# Guardar a CSV
save_hv_results_to_csv(hv_results, "resultadosNSGA2.csv")
