import json
import copy
import numpy as np

from Loaders.LoadUsers import load_users_from_json


def save_pareto_archive_paco(archive, n_users, path="soluciones_PACO.json"):
    solutions_data = []

    for idx, (solution, objectives) in enumerate(archive):
        minutes = objectives[0]
        cost = objectives[1]

        # Reconstruir estructura: 1 array por usuario, cada uno con sus 12 meses
        configuration = []
        n_months = 12

        for user_idx in range(n_users):
            user_platforms = []
            for month in range(n_months):
                index = month * n_users + user_idx
                user_platforms.append(int(solution[index]))  # Aseguramos que sea int para JSON
            configuration.append(user_platforms)

        solutions_data.append({
            "solution_id": idx + 1,
            "minutes": minutes,
            "cost": cost,
            "configuration": configuration
        })

    with open(path, "w", encoding="utf-8") as f:
        json.dump(solutions_data, f, indent=4, ensure_ascii=False)

    print(f"✅ Archivo de soluciones guardado en {path}")





