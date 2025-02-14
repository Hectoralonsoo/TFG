import random
from time import time
import json
from inspyred import ec
from Loaders.LoadStreamingPlans import load_streaming_plan_json
from Loaders.LoadPlatforms import load_platforms_json
from Loaders.LoadUsers import load_users_from_json

# Cargar datos
streamingPlans = load_streaming_plan_json("Data/streamingPlans.json")
platforms = load_platforms_json("Data/MoviesPlatform.json")
users = load_users_from_json("Data/users.json")

# Agrupar planes por plataforma
plataformas_con_planes = {}
for plan_obj in streamingPlans:
    service_name = plan_obj.service_name
    if service_name not in plataformas_con_planes:
        plataformas_con_planes[service_name] = []
    plataformas_con_planes[service_name].append(plan_obj)

available_plans = list(plataformas_con_planes.keys())
n_plataformas = len(available_plans)
n_usuarios = len(users)

# Crear un índice único para cada plataforma y plan
indice_plataformas = {}
indice_global = 0
for plataforma, planes in plataformas_con_planes.items():
    indice_plataformas[plataforma] = {}
    for plan in planes:
        indice_plataformas[plataforma][plan.name] = indice_global
        indice_global += 1

# Guardar el índice en un archivo JSON
with open("Data/indice_plataformas.json", "w", encoding="utf-8") as f:
    json.dump(indice_plataformas, f, indent=4)

# Estructura para precios y perfiles
precios_plataformas = {
    plataforma: {
        "plans": [
            {"name": plan.name, "profiles": plan.profiles, "price": plan.price}
            for plan in planes
        ]
    }
    for plataforma, planes in plataformas_con_planes.items()
}

def print_platforms_and_plans(precios_plataformas):
    print("Plataformas y Planes Disponibles:")
    print("--------------------------------")
    for plataforma, detalles in precios_plataformas.items():
        print(f"Plataforma: {plataforma}")
        for plan in detalles["plans"]:
            print(f"  - Plan: {plan['name']}")
            print(f"    Perfiles: {plan['profiles']}")
            print(f"    Precio: {plan['price']}€")
        print("--------------------------------")

# Generador de individuos mejorado
def generar_individuo(random, args):
    individuo = {}
    for mes in range(12):
        plataforma = random.choice(platforms)
        individuo[mes] = {
           "plataforma": plataforma,
        }
    print(individuo)
    return individuo


# Función de fitness optimizada
def fitness_function(candidates, args):
    fitness_values = []

    for individuo in candidates:
        total_cost = 0
        total_access = 0

        # Primera pasada: calcular costos
        for mes, usuarios_planes in individuo.items():
            for user_id, datos_plan in usuarios_planes.items():
                plataforma = datos_plan["plataforma"]
                plan_nombre = datos_plan["plan"]
                # Buscar el plan correspondiente

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