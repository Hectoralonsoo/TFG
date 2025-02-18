import random
from time import time
import json
from inspyred import ec
from inspyred.ec import emo
from Loaders.LoadStreamingPlans import load_streaming_plan_json
from Loaders.LoadPlatforms import load_platforms_json
from Loaders.LoadUsers import load_users_from_json

# Cargar datos
streamingPlans = load_streaming_plan_json("Data/streamingPlans.json")
platformsMovies = load_platforms_json("Data/MoviesPlatform.json")
platformsSeries = load_platforms_json("Data/SeriesPlatform.json")

users = load_users_from_json("Data/users.json")

with open("Data/indice_plataformas_series.json", "r", encoding="utf-8") as f:
    series_platforms = json.load(f)

# Cargar plataformas de películas
with open("Data/indice_plataformas_movies.json", "r", encoding="utf-8") as f:
    movies_platforms = json.load(f)

# Unir ambas listas y eliminar duplicados usando un conjunto
all_platforms = sorted(set(series_platforms) | set(movies_platforms))

filtered_platforms = [p for p in all_platforms if not p.endswith("Amazon Channel")]


indexed_platforms = {platform: idx + 1 for idx, platform in enumerate(filtered_platforms)}


# Guardar el resultado en indice_plataformas.json
with open("Data/indice_plataformas.json", "w", encoding="utf-8") as f:
    json.dump(indexed_platforms, f, indent=4, ensure_ascii=False)

print("Plataformas combinadas guardadas en indice_plataformas.json")



# Cargar películas por plataforma
with open("Data/MoviesPlatform.json", "r") as f:
    movies_by_platform = json.load(f)
print(movies_by_platform)

with open("Data/indice_plataformas_series.json", "w", encoding="utf-8") as f:
    json.dump(sorted(platformsSeries), f, indent=4, ensure_ascii=False)








# Guardar el índice en un archivo JSON
with open("Data/indice_plataformas.json", "r", encoding="utf-8") as f:
    platforms_indexed = json.load(f)

emo.Pareto()

# Generador de individuos mejorado
def generar_individuo(random, args):
    individuo = {}
    for mes in range(12):
        plataforma = random.choice(platforms_indexed.keys())
        individuo[mes] = {
           "plataforma": plataforma,
        }
    print(individuo)
    return individuo

def evaluator(self, candidates, args):
    fitness=[]
    for c in candidates:
        f1 = maximize_contents_interest()
        f2 = minimize_price()





def fitness_function(candidates, args):
    fitness_values = []
    for individuo in candidates:
        total_cost = 0
        total_access = 0

        # 1. Calcular costos por mes
        for mes in range(12):
            plataforma = individuo[mes]["plataforma"]
            # Obtener el plan más barato para la plataforma
            planes = precios_plataformas[plataforma]["plans"]
            plan_elegido = min(planes, key=lambda x: x["price"])
            total_cost += plan_elegido["price"]

        # 2. Calcular acceso al contenido
        for usuario in users:
            for mes in range(12):
                plataforma = individuo[mes]["plataforma"]
                minutos_disponibles = usuario.monthly_minutes[mes]

                # Acceso a películas
                for movie in usuario.movies:
                    if movie["title"] in movies_by_platform.get(plataforma, {}):
                        duracion = movies_by_platform[plataforma][movie["title"]]
                        duracion_vista = min(duracion, minutos_disponibles)
                        porcentaje = duracion_vista / duracion
                        total_access += movie["interest"] * porcentaje
                        minutos_disponibles -= duracion_vista

                # Acceso a series


        # Fitness combinado
        fitness = total_cost - (total_access)
        fitness_values.append(fitness)

    return fitness_values

# Configuración optimizada del algoritmo genético
def main():
    seed = time()
    prng = random.Random(seed)
    ga = ec.GA(prng)

    ga.observer = [ec.observers.stats_observer, ec.observers.best_observer]
    ga.selector = ec.selectors.tournament_selection
    ga.replacer = ec.replacers.generational_replacement
    ga.variator = [
        ec.variators.uniform_crossover,
        ec.variators.gaussian_mutation
    ]

    final_poblacion = ga.evolve(
        generator=generar_individuo,
        evaluator=fitness_function,
        pop_size=10,
        maximize=False,
        max_generations=100,
        num_elites=2,
        tournament_size=5,
        crossover_rate=0.9,
        mutation_rate=0.2,
        gaussian_stdev=0.5
    )

    # Mostrar resultados
    print_platforms_and_plans(precios_plataformas)
    mejor = min(final_poblacion, key=lambda ind: ind.fitness)
    print(f"\nCosto mínimo encontrado: {mejor.fitness:.2f}€")

    print("\nSolución Óptima Encontrada:")
    print("--------------------------------")
    for mes, usuarios_planes in mejor.candidate.items():
        print(f"Mes {mes + 1}:")
        for user_id, datos_plan in usuarios_planes.items():
            print(f"  Usuario {user_id}: {datos_plan['plataforma']} - {datos_plan['plan']}")
    print("--------------------------------")

if __name__ == "__main__":
    main()