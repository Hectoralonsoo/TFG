import matplotlib.pyplot as plt
import statistics
import json
import numpy as np
from Loaders.LoadUsers import load_users_from_json
from scripts.User_generator import update_users_json
from utils.evaluation import calcular_minutos_ponderados

generations = []
best_minutes = []
best_cost = []
usuarios_meses = {}
evolucion_minutos = []
evolucion_costo = []


def last_generation_update(population, num_generations, args):
    platforms_indexed = args["platforms_indexed"]

    if num_generations == args.get('max_generations') - 1:
        print("\n📌 **Registrando contenido visto por cada usuario con mes y plataforma...**")

        users_data = load_users_from_json("../Data/users.json")
        best_solution = min(population, key=lambda ind: (ind.objective_values[0], ind.objective_values[1]))
        plataformas_por_usuario = best_solution.candidate

        for i, user in enumerate(users_data):
            user.months = plataformas_por_usuario[i]
            plataformas_mes_dict = {mes: plat_id for mes, plat_id in enumerate(user.months)}

            watched_movies = []
            watched_series = []

            # Películas vistas
            for pelicula in user.movies:
                for mes, plat_id in plataformas_mes_dict.items():
                    if plat_id in pelicula.get('platforms', []):
                        watched_movies.append({
                            "title": pelicula["title"],
                            "mes": mes + 1,
                            "plataforma": platforms_indexed.get(str(plat_id), f"Plataforma {plat_id}")
                        })
                        break

            # Series vistas (por temporada)
            for serie in user.series:
                plataformas_serie = serie.get("platforms", [])
                for temporada in serie.get("season", []):
                    plataformas_temporada = set(temporada.get("platforms", []) + plataformas_serie)

                    for mes, plat_id in plataformas_mes_dict.items():
                        if plat_id in plataformas_temporada:
                            watched_series.append({
                                "title": f"{serie['title']} - T{temporada['season_number']}",
                                "mes": mes + 1,
                                "plataforma": platforms_indexed.get(str(plat_id), f"Plataforma {plat_id}")
                            })
                            break

            user.watched_movies = watched_movies
            user.watched_series = watched_series

        update_users_json(users_data)
'''
def update_user_viewing_for_individual(individual, platforms_indexed, users_path, output_path):
    users_data = load_users_from_json(users_path)
    plataformas_por_usuario = individual.candidate

    final_output = {}

    for i, user in enumerate(users_data):
        user_id = f"user_{i}"
        user.months = plataformas_por_usuario[i]
        final_output[user_id] = {}

        # Inicializar cada mes con su plataforma asignada
        for m in range(12):
            platform_id = user.months[m]
            plataforma = platforms_indexed.get(str(platform_id), f"Plataforma {platform_id}")
            final_output[user_id][f"month_{m+1}"] = {
                "plataforma": plataforma,
                "watched_movies": [],
                "watched_series": [],
                "minutes": 0
            }

    # Recuperar contenidos vistos desde el individuo
    watched_movies = getattr(individual, 'watched_movies', [])
    watched_series = getattr(individual, 'watched_series', [])

    for movie in watched_movies:
        user_id = movie.get("user_id", f"user_0")  # Si incluyes user_id en el futuro
        mes_key = f"month_{movie['mes']}"
        if user_id in final_output and mes_key in final_output[user_id]:
            final_output[user_id][mes_key]["watched_movies"].append({
                "title": movie["title"],
                "minutes": movie["duracion"]
            })
            final_output[user_id][mes_key]["minutes"] += movie["duracion"]

    for serie in watched_series:
        user_id = serie.get("user_id", f"user_0")
        mes_key = f"month_{serie['mes']}"
        if user_id in final_output and mes_key in final_output[user_id]:
            final_output[user_id][mes_key]["watched_series"].append({
                "title": serie["title"],
                "minutes": serie["duracion"]
            })
            final_output[user_id][mes_key]["minutes"] += serie["duracion"]

    # Guardar en archivo JSON
    with open(output_path, "w") as f:
        json.dump(final_output, f, indent=4)

    print(f"✅ Exportado: {output_path}")


'''


def update_user_viewing_for_individual(individual, platforms_indexed, users_path, output_path):

    # Cargar datos de usuarios
    with open(users_path, 'r', encoding='utf-8') as f:
        users_data = json.load(f)

    # Preparar la estructura para el resultado
    result = {
        "users": []
    }

    # Si el individuo tiene los atributos watched_movies y watched_series
    if hasattr(individual, 'watched_movies') and hasattr(individual, 'watched_series'):
        # Agrupar el contenido visto por usuario
        content_by_user = {}

        # Procesar películas vistas
        for movie in individual.watched_movies:
            user_id = movie["user_id"]
            if user_id not in content_by_user:
                content_by_user[user_id] = {}

            mes = movie["mes"]
            if mes not in content_by_user[user_id]:
                content_by_user[user_id][mes] = {
                    "plataforma": movie["plataforma"],
                    "contenido": []
                }

            content_by_user[user_id][mes]["contenido"].append({
                "tipo": "pelicula",
                "titulo": movie["title"],
                "duracion": movie["duracion"]
            })

        # Procesar series vistas
        for serie in individual.watched_series:
            user_id = serie["user_id"]
            if user_id not in content_by_user:
                content_by_user[user_id] = {}

            mes = serie["mes"]
            if mes not in content_by_user[user_id]:
                content_by_user[user_id][mes] = {
                    "plataforma": serie["plataforma"],
                    "contenido": []
                }

            content_by_user[user_id][mes]["contenido"].append({
                "tipo": "serie",
                "titulo": serie["title"],
                "duracion": serie["duracion"]
            })

        # Crear el resultado final
        for user_id, months in content_by_user.items():
            # Extraer el índice de usuario del formato "user_X"
            user_index = int(user_id.split('_')[1])

            # Obtener información del usuario original
            user_data = users_data[user_index]

            user_result = {
                "id": user_data.get("id", user_id),
                "name": user_data.get("name", f"Usuario {user_index}"),
                "monthly_minutes": user_data.get("monthly_minutes", 0),
                "historial": {}
            }

            for mes, data in months.items():
                user_result["historial"][str(mes)] = {
                    "plataforma": data["plataforma"],
                    "contenido": [item["titulo"] for item in data["contenido"]]
                }

            result["users"].append(user_result)
    else:
        print("⚠️ Warning: El individuo no tiene los atributos watched_movies y watched_series")





    # Guardar el resultado en formato JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"✅ Solución guardada en {output_path}")



def observer(population, num_generations, num_evaluations, args):
    """
    Muestra información de cada generación en la evolución del algoritmo.
    """
    global generations, best_minutes, best_cost, usuarios_meses

    users = args["users"]
    print(args)

    print(f"ESTO ES LA POPULATION: {population} ")


    print(f"\n=== Generación {num_generations} ===")
    print(f"Número de evaluaciones: {num_evaluations}")

    # Extraer fitness
    fitness_values = [ind.fitness for ind in population]

    total_minutos = sum(-fitness[0] for fitness in fitness_values)
    total_costo = sum(fitness[1] for fitness in fitness_values)

    # Guardar en la evolución
    evolucion_minutos.append(total_minutos)
    evolucion_costo.append(total_costo)

    # Mejores y peores valores
    mejor_minutos = min(fitness[0] for fitness in fitness_values)
    peor_minutos = max(fitness[0] for fitness in fitness_values)
    mejor_costo = min(fitness[1] for fitness in fitness_values)
    peor_costo = max(fitness[1] for fitness in fitness_values)

    # Actualizar las listas globales para la gráfica
    generations.append(num_generations)
    best_minutes.append(-mejor_minutos)  # Negamos porque estamos minimizando el negativo
    best_cost.append(mejor_costo)

    print(f"  Mejor Minutos Ponderados: {-mejor_minutos:.2f}, Peor: {-peor_minutos:.2f}")
    print(f"  Mejor Costo Total: {mejor_costo:.2f}, Peor: {peor_costo:.2f}")

    # Mostrar todos los individuos de la población
    print("\n--- Todos los individuos ---")
    for i in range(len(population)):
        minutos_ponderados = -population[i].fitness[0]
        costo_total = population[i].fitness[1]
        print(f"Individuo {i + 1}: Minutos ponderados: {minutos_ponderados:.2f}, Costo total: {costo_total:.2f}")
        print(f"  Configuración: {population[i].candidate}")

        # Almacenar la información del historial de plataformas por usuario
        for user_index, config_mensual in enumerate(population[i].candidate):
            for mes, plataforma in enumerate(config_mensual):
                if plataforma not in usuarios_meses:
                    usuarios_meses[plataforma] = {}

                user = users[user_index]
                user_id = user.id if hasattr(user, 'id') else user.name

                if user_id not in usuarios_meses[plataforma]:
                    usuarios_meses[plataforma][user_id] = []

                usuarios_meses[plataforma][user_id].append(mes + 1)

    # Al finalizar la evolución, mostrar el resumen de uso de plataformas por usuario
    if num_generations == args.get('max_generations', 0) - 1:
        print("\n📊 **Resumen de contenido visto por usuario**")

        for user in users:
            user_id = user.id if hasattr(user, 'id') else user.name
            print(f"\n👤 Usuario {user_id}:")

            historial = user.__dict__.get("historial", {})

            for mes_str in sorted(historial, key=lambda x: int(x)):
                entry = historial[mes_str]
                plataforma = entry["plataforma"]
                contenidos = ", ".join(entry["contenido"])
                print(f"  📅 Mes {mes_str}: {plataforma} → {contenidos}")





def plot_evolution():
    """ Genera gráficos de la evolución del algoritmo """
    global generations, best_minutes, best_cost

    if not generations:
        print("No hay datos para graficar. Asegúrate de que el algoritmo haya ejecutado al menos una generación.")
        return

    plt.figure(figsize=(12, 6))

    # Gráfico de minutos ponderados
    plt.subplot(1, 2, 1)
    plt.plot(generations, best_minutes, marker='o', linestyle='-', color='b')
    plt.xlabel("Generación")
    plt.ylabel("Minutos Ponderados")
    plt.title("Evolución de Minutos Ponderados")
    plt.grid(True)

    # Gráfico de costo total
    plt.subplot(1, 2, 2)
    plt.plot(generations, best_cost, marker='o', linestyle='-', color='r')
    plt.xlabel("Generación")
    plt.ylabel("Costo Total")
    plt.title("Evolución del Costo Total")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('evolucion.png')  # Guardar gráfico en un archivo
    plt.show()


def plot_pareto_front(algorithm):
    """
      Dibuja el Frente de Pareto final después de la evolución, con puntos unidos.
    """
    pareto_solutions = algorithm.archive
    for solution in pareto_solutions:
        print(f"fitness: {solution.fitness} | type: {type(solution.fitness)}")
    minutos_ponderados = [-solution.fitness[0] for solution in pareto_solutions]
    costo_total = [solution.fitness[1] for solution in pareto_solutions]

    pareto_data = sorted(zip(costo_total, minutos_ponderados))
    costo_total_sorted, minutos_ponderados_sorted = zip(*pareto_data)

    plt.figure(figsize=(8,6))
    plt.plot(costo_total_sorted, minutos_ponderados_sorted, marker='o', linestyle='-', color='red', label="Frente de Pareto")
    plt.xlabel("Costo total (€)")
    plt.ylabel("Minutos ponderados vistos")
    plt.title("Frente de Pareto de las Soluciones")
    plt.legend()
    plt.grid(True)
    plt.savefig("pareto.png")
    plt.show()




def plot_generation_improve():
    """
    Genera gráficos mostrando la evolución del costo total y los minutos ponderados en el algoritmo.
    """
    generaciones = list(range(len(evolucion_minutos)))  # X-Axis

    plt.figure(figsize=(10, 5))

    # 🔹 Gráfico de Minutos Ponderados
    plt.subplot(1, 2, 1)
    plt.plot(generaciones, evolucion_minutos, marker='o', linestyle='-', color='blue', label="Minutos Ponderados")
    plt.xlabel("Generación")
    plt.ylabel("Total Minutos Ponderados")
    plt.title("Evolución de Minutos Ponderados")
    plt.legend()
    plt.grid(True)

    # 🔹 Gráfico de Costo Total
    plt.subplot(1, 2, 2)
    plt.plot(generaciones, evolucion_costo, marker='o', linestyle='-', color='red', label="Costo Total")
    plt.xlabel("Generación")
    plt.ylabel("Costo Total (€)")
    plt.title("Evolución del Costo Total")
    plt.legend()
    plt.grid(True)

    plt.savefig("evolution_improve.png")
    plt.tight_layout()
    plt.show()

'''
def observador_spea2(algorithm, population, num_generations, num_evaluations, args):
    """
    Observador para SPEA2 with exportación del frente de Pareto.
    """
    if not population:
        return

    fitness_values = [ind.fitness for ind in population]
    objectives_1 = [ind.objective_values[0] for ind in population]  # costo
    objectives_2 = [ind.objective_values[1] for ind in population]  # -minutos

    best = min(population, key=lambda x: x.fitness)

    print_detailed = num_generations % args.get('frequency', 1) == 0 if args else True

    print(f"\n🧬 Generación {num_generations} | Evaluaciones: {num_evaluations}")
    print(f"  Tamaño del archivo: {len(population)}")
    print(f"  Mejor fitness: {best.fitness:.4f}")

    if print_detailed:
        print(f"  Objetivos: Costo={best.objective_values[0]:.2f}, Minutos={-best.objective_values[1]:.2f}")
        print(f"  Promedio costo: {statistics.mean(objectives_1):.2f} (±{statistics.stdev(objectives_1):.2f})")
        print(f"  Promedio minutos: {-statistics.mean(objectives_2):.2f} (±{statistics.stdev(objectives_2):.2f})")
        print(f"  Rango costo: [{min(objectives_1):.2f}, {max(objectives_1):.2f}]")
        print(f"  Rango minutos: [{-max(objectives_2):.2f}, {-min(objectives_2):.2f}]")

    # Exportar resultados del frente de Pareto solo en la última generación
    if num_generations == args.get('max_generations') - 1:

        # EJECUTAR PRIMERO last_generation_update
        print("📦 Ejecutando last_generation_update...")
        last_generation_update(population, num_generations, args)

        # AHORA CALCULAR EL FRENTE DE PARETO (después del update)
        print("📊 Calculando frente de Pareto final...")
        pareto_front = get_non_dominated(algorithm.archive)

        print(f"  - Archive final: {len(algorithm.archive)} individuos")
        print(f"  - Frente de Pareto final: {len(pareto_front)} soluciones")
        print("📦 Exportando soluciones del frente de Pareto...")

        # EXPORTAR EL FRENTE CALCULADO DESPUÉS DEL UPDATE
        for idx, ind in enumerate(pareto_front):
            output_path = f"../results/SPEA2/pareto_solution_{idx}SPEA2.json"
            update_user_viewing_for_individual(
                individual=ind,
                platforms_indexed=args["platforms_indexed"],
                users_path="../Data/users.json",
                output_path=output_path
            )
'''
def observador_spea2(algorithm, population, num_generations, num_evaluations, args):
    """
    Observer adaptado para SPEA2. Muestra evolución y exporta soluciones del frente.
    """
    global generations, best_minutes, best_cost, usuarios_meses

    users = args["users"]

    if num_generations == args.get('max_generations') - 1:
        print("📦 Exportando soluciones del frente de Pareto...")

        archive = algorithm.archive

        for idx, ind in enumerate(archive):
            output_path = f"../results/SPEA2/pareto_solution_{idx}_SPEA2.json"

            # 1. Reconstruir monthly_data para el individuo actual
            calcular_minutos_ponderados(ind.candidate, args)
            ind.monthly_data = args['monthly_data_by_user']

            # 2. Preparar el JSON de salida
            export = {
                "candidate": ind.candidate,
                "objectives": list(ind.objective_values),
                "monthly_data": ind.monthly_data
            }

            # 3. Guardar en JSON
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export, f, ensure_ascii=False, indent=2)

    print(f"\n=== Generación {num_generations} ===")
    print(f"Número de evaluaciones: {num_evaluations}")

    # Extraer objetivos
    fitness_values = [ind.objective_values for ind in population]

    total_minutos = sum(-obj[1] for obj in fitness_values)
    total_costo = sum(obj[0] for obj in fitness_values)

    # Guardar en la evolución
    evolucion_minutos.append(total_minutos)
    evolucion_costo.append(total_costo)

    mejor_minutos = min(obj[1] for obj in fitness_values)
    peor_minutos = max(obj[1] for obj in fitness_values)
    mejor_costo = min(obj[0] for obj in fitness_values)
    peor_costo = max(obj[0] for obj in fitness_values)

    generations.append(num_generations)
    best_minutes.append(-mejor_minutos)
    best_cost.append(mejor_costo)

    print(f"  Mejor Minutos Ponderados: {-mejor_minutos:.2f}, Peor: {-peor_minutos:.2f}")
    print(f"  Mejor Costo Total: {mejor_costo:.2f}, Peor: {peor_costo:.2f}")

    print("\n--- Todos los individuos ---")
    for i, ind in enumerate(population):
        minutos_ponderados = -ind.objective_values[1]
        costo_total = ind.objective_values[0]
        print(f"Individuo {i + 1}: Minutos ponderados: {minutos_ponderados:.2f}, Costo total: {costo_total:.2f}")
        print(f"  Configuración: {ind.candidate}")

        # Almacenar la información del historial de plataformas por usuario
        for user_index, config_mensual in enumerate(ind.candidate):
            for mes, plataforma in enumerate(config_mensual):
                if plataforma not in usuarios_meses:
                    usuarios_meses[plataforma] = {}

                user = users[user_index]
                user_id = user.id if hasattr(user, 'id') else user.name

                if user_id not in usuarios_meses[plataforma]:
                    usuarios_meses[plataforma][user_id] = []

                usuarios_meses[plataforma][user_id].append(mes + 1)

    if num_generations == args.get('max_generations', 0) - 1:
        print("\n📊 **Resumen de contenido visto por usuario**")

        for user in users:
            user_id = user.id if hasattr(user, 'id') else user.name
            print(f"\n👤 Usuario {user_id}:")

            historial = user.__dict__.get("historial", {})

            for mes_str in sorted(historial, key=lambda x: int(x)):
                entry = historial[mes_str]
                plataforma = entry["plataforma"]
                contenidos = ", ".join(entry["contenido"])
                print(f"  📅 Mes {mes_str}: {plataforma} → {contenidos}")

def get_non_dominated(solutions):
    """
    Encuentra las soluciones no dominadas (frente de Pareto) en un conjunto de soluciones.
    """
    pareto = []
    for ind in solutions:
        dominated = False
        for other in solutions:
            if (other != ind and
                    all(x <= y for x, y in zip(other.objective_values, ind.objective_values)) and
                    any(x < y for x, y in zip(other.objective_values, ind.objective_values))):
                dominated = True
                break
        if not dominated:
            pareto.append(ind)
    return pareto


def plot_pareto_front_spea2(algorithm):
    """
    Grafica el frente de Pareto específico para SPEA2.
    Usa objective_values, ya que fitness es escalar.
    """
    pareto_solutions = get_non_dominated(algorithm.archive)
    print('ESTO ES EL FRENTE DE PARETO COMPLETO: ' + str(pareto_solutions))

    # Extraer valores objetivos
    minutos_ponderados = [-s.objective_values[1] for s in pareto_solutions]
    costo_total = [s.objective_values[0] for s in pareto_solutions]

    # Ordenar los puntos para una mejor visualización
    puntos_ordenados = sorted(zip(costo_total, minutos_ponderados), key=lambda x: x[0])
    costo_ordenado, minutos_ordenados = zip(*puntos_ordenados) if puntos_ordenados else ([], [])

    print("LONGITUD FRENTE DE PARETO->>>>>>>>" +  str(len(pareto_solutions)))

    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    plt.scatter(costo_ordenado, minutos_ordenados, marker='o', c='blue')

    # Opcionalmente, conectar los puntos del frente
    plt.plot(costo_ordenado, minutos_ordenados, linestyle='-', c='blue', alpha=0.5)

    plt.xlabel("Costo total (€)")
    plt.ylabel("Minutos ponderados vistos")
    plt.title("Frente de Pareto - SPEA2")
    plt.grid(True)
    plt.savefig("pareto_spea2.png")
    plt.show()



def plot_pareto_front_paco(archive, title="Pareto Front - PACOStreaming", save_path="pareto_paco.png"):
    """
    Dibuja el Frente de Pareto final de PACOStreaming.
    archive: lista de (solution, [minutos_ponderados, costo_total])
    """
    if not archive:
        print("No hay soluciones en el archivo.")
        return

    minutos_ponderados = [objectives[0] for _, objectives in archive]  # Maximizar
    costo_total = [objectives[1] for _, objectives in archive]  # Minimizar

    pareto_data = sorted(zip(costo_total, minutos_ponderados))
    costo_total_sorted, minutos_ponderados_sorted = zip(*pareto_data)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    plt.plot(costo_total_sorted, minutos_ponderados_sorted, marker='o', linestyle='-', color='green', label="Pareto Front")
    plt.xlabel("Costo total (€)")
    plt.ylabel("Minutos ponderados vistos")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.savefig(save_path)
    plt.show()


import matplotlib.pyplot as plt
import numpy as np


def plot_ant_paths_lines(all_solutions, n_months, n_users, platform_options, title="Recorrido de Hormigas (Caminos)",
                         max_ants_to_plot=20):
    """
    Dibuja los recorridos de varias hormigas como caminos.

    all_solutions: lista de soluciones de todas las hormigas
    n_months: número de meses
    n_users: número de usuarios
    platform_options: lista de IDs de plataformas
    max_ants_to_plot: máximo de hormigas a graficar para que no quede saturado
    """

    plt.figure(figsize=(14, 8))

    # Limitar el número de hormigas a graficar
    ants_to_plot = all_solutions[:max_ants_to_plot]

    for i, solution in enumerate(ants_to_plot):
        path = []

        for month in range(n_months):
            for user in range(n_users):
                idx = month * n_users + user
                path.append(solution[idx])

        steps = range(len(path))
        plt.plot(steps, path, marker='o', linestyle='-', label=f"Hormiga {i + 1}", alpha=0.7)

    plt.xlabel("Paso (Mes × Usuarios)")
    plt.ylabel("ID Plataforma Elegida")
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt

def plot_user_platforms_over_time(user_platforms, user_name="Usuario X"):
    """
    Dibuja la evolución de las plataformas asignadas a un usuario a lo largo de los 12 meses.

    user_platforms: lista de plataformas (longitud 12) para el usuario.
    user_name: nombre o identificador del usuario.
    """
    months = list(range(1, 13))  # Meses de 1 a 12

    plt.figure(figsize=(10, 6))
    plt.step(months, user_platforms, where='mid', marker='o', linestyle='-', color='blue')
    plt.xticks(months)
    plt.xlabel("Mes")
    plt.ylabel("ID Plataforma")
    plt.title(f"Evolución de plataformas - {user_name}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()




def observer_paco(paco, iteration, args, export_dir="../results/PACO"):
    """
    Observer para PACO: muestra estado por iteración y exporta archivo final.
    """
    global generations, best_minutes, best_cost

    archive = paco.archive
    print(f"\n=== Iteración {iteration + 1}/{paco.n_iterations} ===")
    print(f"Archivo actual (frente de Pareto): {len(archive)} soluciones")

    if not archive:
        print("⚠️ Archivo vacío. No hay soluciones no dominadas.")
        return

    # Extraer objetivos
    minutos = [obj[0] for _, obj in archive]
    costos = [obj[1] for _, obj in archive]

    mejor_minutos = max(minutos)
    peor_minutos = min(minutos)
    mejor_costo = min(costos)
    peor_costo = max(costos)

    print(f"  ✅ Mejor minutos ponderados: {mejor_minutos:.2f} | Peor: {peor_minutos:.2f}")
    print(f"  💰 Mejor costo total: {mejor_costo:.2f} | Peor: {peor_costo:.2f}")

    # Guardar para graficar evolución
    generations.append(iteration)
    best_minutes.append(mejor_minutos)
    best_cost.append(mejor_costo)

    # Si es la última iteración, exportar soluciones
    if iteration + 1 == paco.n_iterations:
        n_users = len(args["users"])
        n_months = paco.n_months

        print("📦 Exportando soluciones finales del frente de Pareto...")

        for idx, (solution, objectives) in enumerate(archive):
            # Reconstruir candidato
            candidate = [[solution[month * n_users + user] for month in range(n_months)]
                         for user in range(n_users)]

            # Calcular contenidos vistos (esto actualiza monthly_data_by_user en args)
            calcular_minutos_ponderados(candidate, args)
            monthly_data = args.get("monthly_data_by_user", {})

            output = {
                "candidate": candidate,
                "objectives": objectives,
                "monthly_data": monthly_data
            }

            with open(f"{export_dir}/pareto_solution_{idx}_PACO.json", 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"✅ {len(archive)} soluciones exportadas a {export_dir}")
