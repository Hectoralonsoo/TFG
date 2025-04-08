import matplotlib.pyplot as plt
import statistics
import numpy as np
from Loaders.LoadUsers import load_users_from_json
from scripts.User_generator import update_users_json

generations = []
best_minutes = []
best_cost = []
usuarios_meses = {}
evolucion_minutos = []
evolucion_costo = []


def last_generation_update(population, num_generations, args):
    """
    Registra para cada usuario qué contenidos ha visto, en qué mes y plataforma, y actualiza `users.json`.
    """
    platforms_indexed = args["platforms_indexed"]

    if num_generations == args.get('max_generations', 100) - 1:
        print("\n📌 **Registrando contenido visto por cada usuario con mes y plataforma...**")

        users_data = load_users_from_json("Data/users.json")
        best_solution = min(population, key=lambda ind: (ind.fitness[0], ind.fitness[1]))
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
                        break  # Una vez vista, no repetir

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
                            break  # No repetir temporada

            user.watched_movies = watched_movies
            user.watched_series = watched_series

        update_users_json(users_data)








def observer(population, num_generations, num_evaluations, args):
    """
    Muestra información de cada generación en la evolución del algoritmo.
    """
    global generations, best_minutes, best_cost, usuarios_meses

    users = args["users"]


    print(f"ESTO ES LA POPULATION: {population} ")

    if num_generations == args.get('max_generations', 100) - 1:
        last_generation_update(population, num_generations, args)

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

            historial = user.__dict__.get("historial", {})  # Si estás usando clases, accede así

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
    minutos_ponderados = [-solution.fitness[0] for solution in pareto_solutions]  # Maximizar
    costo_total = [solution.fitness[1] for solution in pareto_solutions]  # Minimizar

    # Ordenar los puntos por costo para que la línea sea continua
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






def observador_spea2(population, generation, num_evaluations, args=None):
    """
    Observador para el algoritmo SPEA2.
    Muestra y guarda estadísticas de cada generación.
    """
    if not population:
        return

    # Calcular estadísticas básicas
    fitness_values = [ind.fitness for ind in population]
    objectives_1 = [ind.objective_values[0] for ind in population]  # costo
    objectives_2 = [ind.objective_values[1] for ind in population]  # -minutos (negativo porque queremos maximizar)

    # Encontrar el mejor individuo
    best = min(population, key=lambda x: x.fitness)

    # Identificar soluciones no dominadas (frente de Pareto)
    pareto_front = []
    for ind in population:
        is_dominated = False
        for other in population:
            if other != ind and all(x <= y for x, y in zip(other.objective_values, ind.objective_values)) and any(
                    x < y for x, y in zip(other.objective_values, ind.objective_values)):
                is_dominated = True
                break
        if not is_dominated:
            pareto_front.append(ind)

    # Control de frecuencia de impresión detallada
    print_detailed = generation % args.get('frequency', 1) == 0 if args else True

    # Mostrar información
    print(f"\n🧬 Generación {generation} | Evaluaciones: {num_evaluations}")
    print(f"  Soluciones en el frente de Pareto: {len(pareto_front)}/{len(population)}")
    print(f"  Mejor fitness: {best.fitness:.4f}")

    if print_detailed:
        print(f"  Objetivos: Costo={best.objective_values[0]:.2f}, Minutos={-best.objective_values[1]:.2f}")
        print(f"  Promedio costo: {statistics.mean(objectives_1):.2f} (±{statistics.stdev(objectives_1):.2f})")
        print(f"  Promedio minutos: {-statistics.mean(objectives_2):.2f} (±{statistics.stdev(objectives_2):.2f})")
        print(f"  Rango costo: [{min(objectives_1):.2f}, {max(objectives_1):.2f}]")
        print(f"  Rango minutos: [{-max(objectives_2):.2f}, {-min(objectives_2):.2f}]")

    # Guardar historial si se pasa un archivo
    if args and 'log_file' in args:
        # Primera vez, escribir encabezado
        if generation == 0:
            with open(args['log_file'], 'w') as f:
                f.write(
                    "generation,best_fitness,best_cost,best_minutes,avg_cost,avg_minutes,std_cost,std_minutes,pareto_size\n")

        # Añadir datos de esta generación
        with open(args['log_file'], 'a') as f:
            f.write(f"{generation},{best.fitness:.6f},{best.objective_values[0]:.6f},{-best.objective_values[1]:.6f}," +
                    f"{statistics.mean(objectives_1):.6f},{-statistics.mean(objectives_2):.6f}," +
                    f"{statistics.stdev(objectives_1):.6f},{statistics.stdev(objectives_2):.6f},{len(pareto_front)}\n")


def plot_pareto_front_spea2(algorithm):
    """
    Grafica el frente de Pareto específico para SPEA2.
    Usa objective_values, ya que fitness es escalar.
    """
    pareto_solutions = algorithm.archive
    minutos_ponderados = [-s.objective_values[1] for s in pareto_solutions]
    costo_total = [s.objective_values[0] for s in pareto_solutions]

    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    plt.scatter(costo_total, minutos_ponderados, c='blue')
    plt.xlabel("Costo total (€)")
    plt.ylabel("Minutos ponderados vistos")
    plt.title("Frente de Pareto - SPEA2")
    plt.grid(True)
    plt.savefig("pareto_spea2.png")
    plt.show()
