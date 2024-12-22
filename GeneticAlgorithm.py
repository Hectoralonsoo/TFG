import random

from inspyred import ec, benchmarks
from random import Random

from Loaders.LoadStreamingPlans import load_streaming_plan_json, getServices

streamingPlans = load_streaming_plan_json("Data/streamingPlans.json")
available_plans = getServices(streamingPlans)
"""
def generar_solucion_aleatoria(num_usuarios):
    users = []
    for _ in range(num_usuarios):
        user = random.randint(0, len(available_plans)-1)
        users.append(user)
    return users
"""


def generate_initial_population(streaming_plans, num_users, population_size, max_plans_per_month):
    """
    Genera la población inicial para el algoritmo genético, donde cada usuario
    tiene 12 meses y puede elegir varios planes por mes.

    Args:
        streaming_plans (list): Lista de objetos `StreamingPlan` disponibles.
        num_users (int): Número de usuarios a considerar.
        population_size (int): Tamaño de la población inicial.
        max_plans_per_month (int): Número máximo de planes que un usuario puede elegir por mes.

    Returns:
        list: Lista de individuos, donde cada individuo es un array 3D (num_users x 12 x max_plans_per_month).
    """
    population = []

    # Verificar que hay planes disponibles
    if not streaming_plans:
        raise ValueError("La lista de planes de streaming está vacía.")

    # Generar cada individuo de la población
    for _ in range(population_size):
        # Crear un individuo
        individual = []
        for _ in range(num_users):
            # Crear las elecciones del usuario (12 meses)
            user_choices = []
            for _ in range(12):  # Para cada mes
                # Elegir aleatoriamente los planes del mes
                plans_for_month = random.sample(streaming_plans, k=random.randint(1, max_plans_per_month))
                user_choices.append(plans_for_month)
            individual.append(user_choices)

        # Agregar individuo a la población
        population.append(individual)

    return population

num_users = 3
population_size = 5
max_plans_per_month = 3

# Generar población inicial
initial_population = generate_initial_population(streamingPlans, num_users, population_size, max_plans_per_month)


print("Ejemplo de individuo (solución):")
for i, user in enumerate(initial_population[0]):
    print(f"Usuario {i + 1}:")
    for month, plans in enumerate(user):
        plan_names = [f"{plan.service_name} - {plan.name}" for plan in plans]
        print(f"  Mes {month + 1}: {plan_names}")

