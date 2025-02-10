import random
from time import time
from inspyred import ec
from Loaders.LoadStreamingPlans import load_streaming_plan_json
from Loaders.LoadUsers import load_users_from_json

# Cargar datos
streamingPlans = load_streaming_plan_json("Data/streamingPlans.json")
users = load_users_from_json("Data/users.json")
lambda_ = 0.5

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


# Función para verificar disponibilidad de contenido
def contenido_disponible(contenido, plataforma):
    return contenido.platform == plataforma


# Generador de individuos mejorado
def generar_individuo(random, args):
    individuo = {}
    for plataforma in available_plans:
        for mes in range(12):
            # 50% de probabilidad de incluir la plataforma en el mes
            if random.random() < 0.5:
                planes_disponibles = plataformas_con_planes[plataforma]
                plan = random.choice(planes_disponibles)
                # Seleccionar hasta el máximo de perfiles permitidos
                max_usuarios = plan.profiles
                usuarios = random.sample(
                    range(n_usuarios),
                    k=random.randint(1, max_usuarios)
                ) if n_usuarios > 0 else []
                individuo[(plataforma, mes)] = {
                    "plan": plan.name,
                    "users": usuarios
                }
    return individuo


# Función de fitness optimizada
def fitness_function(candidates, args):
    fitness_values = []

    for individuo in candidates:
        total_cost = 0
        total_access = 0
        user_platforms = {user.id: [set() for _ in range(12)] for user in users}

        # Primera pasada: calcular costos y preparar datos de acceso
        for (plataforma, mes), grupo in individuo.items():
            try:
                plan = next(
                    p for p in plataformas_con_planes[plataforma]
                    if p.name == grupo["plan"]
                )
            except StopIteration:
                continue

            # Validar y calcular costo
            if len(grupo["users"]) > plan.profiles:
                total_cost += 10000
            else:
                total_cost += plan.price

            # Registrar plataformas por usuario
            for user_id in grupo["users"]:
                if 0 <= user_id < len(users):
                    user_platforms[users[user_id].id][mes].add(plataforma)

        # Segunda pasada: calcular acceso
        for user in users:
            for mes in range(12):
                minutos_disponibles = user.monthly_minutes
                plataformas_acceso = user_platforms[user.id][mes]

                for contenido in user.movies + user.series:
                    if minutos_disponibles <= 0:
                        break

                    # Verificar si el contenido está disponible en alguna plataforma
                    for plataforma in plataformas_acceso:
                        if contenido_disponible(contenido, plataforma):
                            duracion = min(contenido.duration, minutos_disponibles)
                            total_access += user.interest * (duracion / contenido.duration)
                            minutos_disponibles -= duracion
                            break  # Solo contar una vez por contenido

        fitness = total_cost - (lambda_ * total_access)
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
        pop_size=100,
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
    for (plataforma, mes), datos in mejor.candidate.items():
        print(f"Mes {mes + 1} - {plataforma}:")
        print(f"  Plan: {datos['plan']}")
        print(f"  Usuarios: {len(datos['users'])}")
    print("--------------------------------")



main()