import random
from time import time
import json

import inspyred
from inspyred import ec
from inspyred.ec import emo
from Loaders.LoadStreamingPlans import load_streaming_plan_json
from Loaders.LoadPlatforms import load_platforms_json
from Loaders.LoadUsers import load_users_from_json

# ==============================
# Cargar datos
# ==============================
streamingPlans = load_streaming_plan_json("Data/streamingPlans.json")
users = load_users_from_json("Data/users.json")

# Cargar películas por plataforma
with open("Data/MoviesPlatform.json", "r") as f:
    movies_by_platform = json.load(f)
#print(movies_by_platform)

with open("Data/SeriesPlatform.json", "r") as f:
    series_by_platform = json.load(f)
#print(series_by_platform)


with open("Data/indice_plataformas.json", "r", encoding="utf-8") as f:
    platforms_indexed = json.load(f)



# ==============================
# Generador de Individuos
# ==============================
def generar_individuo(random, args):
    """
    Genera un individuo representado como una lista de usuarios, donde cada usuario tiene una plataforma elegida para cada mes.
    """
    individuo = []
    for user in users:
        user_selection = {
            "name": user.name,  # Cambiar a notación de punto
            "user_id": user.id,  # Cambiar a notación de punto
            "monthly_minutes": user.monthly_minutes,  # Cambiar a notación de punto
            "movies": user.movies,  # Cambiar a notación de punto
            "series": user.series,  # Cambiar a notación de punto
            "months": {}  # Cambiar a notación de punto
        }

        for mes in range(12):
            plataforma = random.choice(list(platforms_indexed.values()))
            user_selection["months"][str(mes)] = plataforma

        individuo.append(user_selection)

    with open("Data/users.json", "w", encoding="utf-8") as f:
        json.dump(individuo, f, ensure_ascii=False, indent=4)

    print(individuo)
    return individuo



def evaluator(candidates, args):
    """
    Función evaluadora que, para cada candidato, calcula dos objetivos:
      - f1: minimizar (-interés_total)  --> en realidad se maximiza el interés.
      - f2: minimizar el costo total.
    Se retorna una lista de tuplas (f1, f2) para cada candidato.
    """
    fitness = []
    return fitness



# ==============================
# Configuración y Ejecución del Algoritmo Genético
# ==============================
def main():
    seed = time()
    prng = random.Random(seed)
    ga = ec.GA(prng)

    bounder = inspyred.ec.Bounder(0, 17)

    # Asignamos observadores y operadores
    ga.observer = ec.observers.stats_observer
    ga.selector = ec.selectors.tournament_selection
    ga.replacer = ec.replacers.generational_replacement
    ga.variator = [
        ec.variators.uniform_crossover,
        ec.variators.gaussian_mutation
    ]
    final_pop = ga.evolve(
        generator=generar_individuo,
        evaluator=evaluator,
        bounder=bounder,
        pop_size=10,
        maximize=False,  # Se minimiza; por ello, el interés se convierte en negativo
        max_generations=100,
        num_elites=2,
        tournament_size=5,
        crossover_rate=0.9,
        mutation_rate=0.2,
        gaussian_stdev=0.5
    )



if __name__ == "__main__":
    main()
