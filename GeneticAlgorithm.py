import random
from time import time
import json
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
print(movies_by_platform)

with open("Data/SeriesPlatform.json", "r") as f:
    series_by_platform = json.load(f)
print(series_by_platform)


with open("Data/indice_plataformas.json", "r", encoding="utf-8") as f:
    platforms_indexed = json.load(f)

# Asignamos el archivo Pareto para el manejo de la frontera
pareto_archive = emo.Pareto()


# ==============================
# Funciones de evaluación
# ==============================

# Estas funciones "simulan" la obtención del precio y del interés
# a partir del índice asignado a cada plataforma.
def get_platform_price(platform):
    """
    Asigna un precio mensual a la plataforma.
    Por ejemplo, se parte de un precio base y se incrementa según el índice.
    """
    base_price = 5  # Precio base en alguna unidad
    increment = 1  # Incremento por cada posición
    index = platforms_indexed[platform]
    return base_price + (index - 1) * increment


def get_platform_interest(platform):
    """
    Asigna un valor de interés en contenidos a la plataforma.
    Se parte de un máximo y se reduce según el índice.
    """
    max_interest = 10  # Valor máximo de interés
    decrement = 0.5  # Decrece en función del índice
    index = platforms_indexed[platform]
    return max_interest - (index - 1) * decrement


def maximize_contents_interest(candidate):
    """
    Calcula el interés total en contenidos para el candidato.
    Como se desea maximizar el interés y el algoritmo minimiza,
    se devuelve el negativo del total.
    """
    total_interest = 0
    for mes in candidate:
        plataforma = candidate[mes]["plataforma"]
        total_interest += get_platform_interest(plataforma)
    return -total_interest  # Negativo para transformarlo en un problema de minimización


def minimize_price(candidate):
    """
    Calcula el costo total mensual de las plataformas elegidas en el candidato.
    Se suma el precio de cada plataforma.
    """
    total_price = 0
    for mes in candidate:
        plataforma = candidate[mes]["plataforma"]
        total_price += get_platform_price(plataforma)
    return total_price


def evaluator(candidates, args):
    """
    Función evaluadora que, para cada candidato, calcula dos objetivos:
      - f1: minimizar (-interés_total)  --> en realidad se maximiza el interés.
      - f2: minimizar el costo total.
    Se retorna una lista de tuplas (f1, f2) para cada candidato.
    """
    fitness = []
    for candidate in candidates:
        f1 = maximize_contents_interest(candidate)
        f2 = minimize_price(candidate)
        fitness.append(emo.Pareto([f1, f2]))
    return fitness


# ==============================
# Generador de Individuos
# ==============================
def generar_individuo(random, args):
    """
    Genera un individuo representado como un diccionario con 12 meses.
    Para cada mes se elige aleatoriamente una plataforma de la lista.
    """
    individuo = {}
    for mes in range(12):
        plataforma = random.choice(list(platforms_indexed.keys()))
        individuo[mes] = {"plataforma": plataforma}
    # Se imprime para debug (se puede comentar en producción)
    print("Individuo generado:", individuo)
    return individuo


# ==============================
# Configuración y Ejecución del Algoritmo Genético
# ==============================
def main():
    seed = time()
    prng = random.Random(seed)
    ga = ec.GA(prng)

    # Asignamos observadores y operadores
    ga.observer = [ec.observers.stats_observer, ec.observers.best_observer]
    ga.selector = ec.selectors.tournament_selection
    ga.replacer = ec.replacers.generational_replacement
    ga.variator = [
        ec.variators.uniform_crossover,
        ec.variators.gaussian_mutation
    ]

    # Para problemas multiobjetivo se asigna el archivo Pareto al algoritmo
    ga.archive = pareto_archive

    final_poblacion = ga.evolve(
        generator=generar_individuo,
        evaluator=evaluator,
        pop_size=10,
        maximize=False,  # Se minimiza; por ello, el interés se convierte en negativo
        max_generations=100,
        num_elites=2,
        tournament_size=5,
        crossover_rate=0.9,
        mutation_rate=0.2,
        gaussian_stdev=0.5
    )

    # Se imprime la frontera de Pareto obtenida
    print("Frontera Pareto:")
    for sol in ga.archive:
        print(sol)


if __name__ == "__main__":
    main()
