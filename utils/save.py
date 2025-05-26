import json
import copy
import numpy as np

from Loaders.LoadUsers import load_users_from_json
from utils.evaluation import calcular_minutos_ponderados

'''
def save_pareto_archive_paco(archive, n_users, args):
    solutions_data = []

    for idx, (solution, objectives) in enumerate(archive):
        minutes = objectives[0]
        cost = objectives[1]

        n_months = 12


        candidate = [[solution[month * n_users + user] for month in range(n_months)]
                     for user in range(n_users)]

        # Calcular contenidos vistos (esto actualiza monthly_data_by_user en args)
        calcular_minutos_ponderados(candidate, args)
        monthly_data = args.get("monthly_data_by_user")

        output = {
            "candidate": candidate,
            "objectives": objectives,
            "monthly_data": monthly_data
        }

        with open(f"../results/PACO/pareto_solution_{idx}_PACO.json", 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"✅ {len(archive)} soluciones exportadas a ../results/PACO")

'''


def save_pareto_archive_paco(archive, n_users, args):
    solutions_data = []

    for idx, (solution, objectives) in enumerate(archive):
        minutes = float(objectives[0])  # asegurar float nativo
        cost = float(objectives[1])     # asegurar float nativo

        n_months = 12

        # Convertir solution de np.ndarray a lista de int nativos
        solution_list = solution.tolist()

        candidate = [[int(solution_list[month * n_users + user]) for month in range(n_months)]
                     for user in range(n_users)]

        calcular_minutos_ponderados(candidate, args)
        monthly_data = args.get("monthly_data_by_user")

        # Convertir monthly_data a tipos nativos si es necesario
        def convert(obj):
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(i) for i in obj]
            elif isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            else:
                return obj

        output = {
            "candidate": candidate,
            "objectives": [minutes, cost],
            "monthly_data": convert(monthly_data)
        }

        with open(f"../results/PACO/pareto_solution_{idx}_PACO.json", 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"✅ {len(archive)} soluciones exportadas a ../results/PACO")


