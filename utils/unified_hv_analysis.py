import json
import os
import pandas as pd
from pathlib import Path

uploaded_files = {
    "SPEA2": "C:\\Users\\hctr0\\PycharmProjects\\TFG_Hector\\results\\SPEA2\\summary_all_experiments.json",
    "NSGA2": "C:\\Users\\hctr0\\PycharmProjects\\TFG_Hector\\results\\NSGA2\\summary_all_experiments.json",
    "PACO":  "C:\\Users\\hctr0\\PycharmProjects\\TFG_Hector\\results\\PACO\\summary_all_experiments.json"
}

output_folder = "C:\\Users\\hctr0\\PycharmProjects\\TFG_Hector\\pareto_outputs"
os.makedirs(output_folder, exist_ok=True)

pareto_points_by_dataset = {}

for algorithm, file_path in uploaded_files.items():
    print(f"Leyendo archivo: {file_path}")
    with open(file_path, "r") as f:
        data = json.load(f)

    for entry in data:
        dataset = entry["dataset"]
        points = entry["pareto_points"]

        normalized_points = []

        for p in points:
            # Detectar si el formato es [-minutes, cost]
            if algorithm != "PACO" and p[0] < 0 and p[1] > 0:
                minutes = -p[0]
                cost = p[1]
            else:
                minutes = p[0]
                cost = p[1]

            normalized_points.append([cost, minutes])

        pareto_points_by_dataset.setdefault(dataset, []).extend(normalized_points)

# Guardar CSV y calcular estadísticas
output_paths = []
stats_by_dataset = []

for dataset, points in pareto_points_by_dataset.items():
    df = pd.DataFrame(points, columns=["cost", "minutes"])
    csv_path = os.path.join(output_folder, f"pareto_points_{dataset.replace('.json', '')}.csv")

    try:
        df.to_csv(csv_path, index=False)
        print(f"✅ Guardado: {csv_path}")
        output_paths.append(csv_path)
    except Exception as e:
        print(f"❌ Error al guardar {csv_path}: {e}")

    # Validación adicional
    if df["cost"].max() < df["minutes"].max():
        print(f"⚠️ Revisión recomendada para {dataset}: minutos más altos que coste.")

    stats = {
        "dataset": dataset,
        "min_cost": df["cost"].min(),
        "max_cost": df["cost"].max(),
        "min_minutes": df["minutes"].min(),
        "max_minutes": df["minutes"].max(),
    }
    stats_by_dataset.append(stats)

# Guardar resumen
stats_df = pd.DataFrame(stats_by_dataset)
stats_path = os.path.join(output_folder, "pareto_stats_summary.csv")
stats_df.to_csv(stats_path, index=False)
print(f"\n📊 Estadísticas guardadas en: {stats_path}")
print(stats_df)
